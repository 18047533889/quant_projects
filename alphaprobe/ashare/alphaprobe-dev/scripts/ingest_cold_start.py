#!/usr/bin/env python3
"""Phase D —— 冷启动库 6266 条批量 ingest 进 GlobalSeenIndex + audit + benchmark。

任务书 §48。只写 ingest 脚本（本轮只允许碰 scripts|tools 新增、
tests/test_seed_ingest.py、benchmark JSON），不改 src/alphaprobe/ 任何现有文件。

流程：
1. 每条 seed 走 FE identity（factor_engine.identity.get_factor_identity，
   ensure_authority_available 引导）拿 canonical 身份。
   - FE 成功 → 构造鸭子 identity 喂 DedupService.reserve（幂等：UNIQUE 兜底，
     reserve 内部查回 existing，重跑新增=0）。
   - FE 抛错 → 记 parse 失败（seen_parse_failures），不中断整批。
2. parse 成功后按 reserve 的 verdict 分桶：
   - NEW（acquired=True）→ 新增入库。若 identity 的 parameter_family_id
     指向的 family 在库中已有成员 → family 归并（family_merge）；否则 fresh_new。
   - EXACT_DUPLICATE → exact 重复
   - SIGN_EQUIVALENT_DUPLICATE → sign 变体
   - （reserve 不会返回 FAMILY_SATURATED / VERSION_MISMATCH 给写路径，
     但 VERSION_MISMATCH 在 reserve 内会照常新建本 version 行→当作 new）
   守恒：raw(engine_ready 处理条数) = fresh_new + family_merge + exact + sign + parse_fail。

3. audit 报告 JSON：raw / exact / sign / family归并 / parse失败 / 新增入库 六类
   + parse 失败样例前 20 条（factor_id + 错误摘要）。
4. --benchmark 模式：parse/hash/insert 分阶段吞吐 + exact/family/nearest
   lookup p50/p95（默认 1000 次采样），用真实 seed 串 + 合成变异串。结果落 JSON。

禁止 except Exception: pass（所有捕获都必须记录并落入 audit，仅 FE 解析失败可捕获并单独计数）。

用法：
    PYTHONPATH=src python3 scripts/ingest_cold_start.py --db data/seen_index.db
    PYTHONPATH=src python3 scripts/ingest_cold_start.py --db data/seen_index.db --benchmark --bench-n 1000
    PYTHONPATH=src python3 scripts/ingest_cold_start.py --limit 30 --db /tmp/seen_smoke.db   # 冒烟
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alphaprobe.authority import FactorIdentityAuthorityError, ensure_authority_available
from alphaprobe.seen import build_seen_index
from alphaprobe.seen.dedup_service import DedupService

# ---------------------------------------------------------------------------
# 合成查询生成（benchmark：从真实 seed 池采样并在算子窗口数值上做变异）
# ---------------------------------------------------------------------------


def _mutate_formula(formula: str, rnd: random.Random) -> str:
    """在公式里替换一个数字字面量为邻近值，或追加 1 个无关子式，生成变异 seed 串。"""
    text = str(formula)
    # 尝试替换一个数字窗口参数
    import re

    nums = list(re.finditer(r"\b\d+(\.\d+)?\b", text))
    if nums:
        m = rnd.choice(nums)
        try:
            val = float(m.group(0))
            delta = rnd.choice([-1, 1, 2, -2, 1.0, 3.0])
            newval = max(1, int(val + delta)) if val == int(val) else round(val + delta, 3)
            return text[: m.start()] + str(newval) + text[m.end():]
        except Exception:  # noqa: BLE001 - 变异失败则回落 append
            pass
    # 回落：追加一个常数项
    return f"{text} * 1.0"


def _synthetic_query_pool(seed_forms: list[str], n: int, rnd: random.Random) -> list[str]:
    pool: list[str] = []
    for _ in range(n):
        base = rnd.choice(seed_forms)
        kind = rnd.random()
        if kind < 0.5:
            pool.append(_mutate_formula(base, rnd))
        elif kind < 0.8:
            pool.append(base)
        else:
            pool.append(f"rank(ts_sum({base}, {rnd.randint(3, 10)}))")
    return pool


# ---------------------------------------------------------------------------
# Audit（六类桶，守恒保证）
# ---------------------------------------------------------------------------

_BUCKETS = ["raw", "fresh_new", "family_merge", "exact", "sign", "parse_fail"]


def _new_audit() -> dict[str, Any]:
    return {
        b: 0 for b in _BUCKETS
    } | {"parse_failure_samples": [], "engine_ready": 0, "non_engine_ready_skipped": {}}


def _family_already_has_member(store: Any, family_id: str) -> bool:
    if not family_id:
        return False
    row = store.q1(
        "SELECT COUNT(*) FROM seen_family_members WHERE family_id=?",
        (family_id,),
    )
    return int(row[0]) > 0


# ---------------------------------------------------------------------------
# 单条 ingest
# ---------------------------------------------------------------------------


class _FeIdentityView:
    """FE FactorIdentity 的鸭子视图（供 reserve 读取）。

    FE 返回 canonical_ast_hash / signal_equivalence_id / parameter_family_id /
    orientation / canon DSL。reserve 只读这些字段。
    """

    def __init__(self, fe_result: Any, factor_id: str, canonical_text: str) -> None:
        d = _fe_identity_dict(fe_result)
        self.factor_id = factor_id
        self.canonical_ast_hash = str(d.get("canonical_ast_hash") or "")
        self.signal_equivalence_id = str(d.get("signal_equivalence_id") or "")
        fam = d.get("parameter_family_id")
        self.parameter_family_id = str(fam) if fam else None
        orientation = d.get("orientation", 1)
        self.orientation = int(orientation or 1)
        self.identity_version = str(d.get("identity_version", "1") or "1")
        self.canonical_formula = canonical_text
        self.operator_semantics_version = d.get("operator_semantics_version")


def _fe_identity_dict(fe_result: Any) -> dict[str, Any]:
    if isinstance(fe_result, dict):
        return fe_result
    return {
        "canonical_ast_hash": getattr(fe_result, "canonical_ast_hash", ""),
        "signal_equivalence_id": getattr(fe_result, "signal_equivalence_id", ""),
        "parameter_family_id": getattr(fe_result, "parameter_family_id", None),
        "orientation": getattr(fe_result, "orientation", 1),
        "identity_version": getattr(fe_result, "identity_version", "1"),
        "canonical_dsl": getattr(fe_result, "canonical_dsl", ""),
    }


def ingest_one(
    svc: DedupService,
    formula: str,
    *,
    factor_id: str,
    canonical_text: str,
    family_id_from_seed: str | None,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """单条 ingest。返回 per-formula timing。FE 解析失败记 parse_fail，不抛。"""
    timing: dict[str, float] = {"parse": 0.0, "hash": 0.0, "insert": 0.0}
    try:
        t0 = time.perf_counter()
        ensure_authority_available()
        from factor_engine.identity import get_factor_identity
        from factor_engine.identity import FactorIdentityError as _FEParseError

        try:
            fe = get_factor_identity(formula)
        except _FEParseError as fe_exc:
            # FE AST 解析失败 → 独立 parse_fail 桶，不中断整批
            audit["parse_fail"] += 1
            if len(audit["parse_failure_samples"]) < 20:
                audit["parse_failure_samples"].append(
                    {"factor_id": factor_id, "error": f"FE parse: {fe_exc}"[:200]}
                )
            return timing
        timing["parse"] = time.perf_counter() - t0
        d = _fe_identity_dict(fe)
        canon_hash = str(d.get("canonical_ast_hash") or "")
        signal_id = str(d.get("signal_equivalence_id") or "")
        if not canon_hash or not signal_id:
            raise FactorIdentityAuthorityError(
                f"FE identity incomplete (hash={bool(canon_hash)}, sig={bool(signal_id)})"
            )
        timing["hash"] = 0.0  # hash 随 parse 完成，不单列
        family_id = str(d.get("parameter_family_id") or "") or family_id_from_seed or None
        # 判 family 是否库中已有成员必须在 reserve 之前：reserve 会往
        # seen_family_members 插入本因子自身，导致 insert 后查询永远非空 → 无法区分
        # 首员(fresh_new) vs 归并(family_merge)。
        family_was_populated = _family_already_has_member(svc._store, family_id)
        t0 = time.perf_counter()
        view = _FeIdentityView(fe, factor_id=factor_id, canonical_text=canonical_text)
        result = svc.reserve(
            view,
            source_system="seed_cold_start",
            run_id=f"phaseD_seed_{int(time.time())}",
        )
        timing["insert"] = time.perf_counter() - t0

        verdict = result.verdict
        if result.acquired:
            # 新增入库：family 此前是否已有成员决定 fresh_new vs family_merge
            if family_id and family_was_populated:
                audit["family_merge"] += 1
            else:
                audit["fresh_new"] += 1
        elif verdict == "EXACT_DUPLICATE":
            audit["exact"] += 1
        elif verdict == "SIGN_EQUIVALENT_DUPLICATE":
            audit["sign"] += 1
        else:
            # VERSION_MISMATCH 时 reserve 会照常新建本 version → 计入新增
            audit["fresh_new"] += 1
        return timing
    except FactorIdentityAuthorityError as exc:  # FE parse 失败 → 独立计数
        audit["parse_fail"] += 1
        if len(audit["parse_failure_samples"]) < 20:
            audit["parse_failure_samples"].append(
                {"factor_id": factor_id, "error": str(exc)[:200]}
            )
        return timing
    # NOTE: 不捕获 BaseException。任何其它异常向上抛，保证不吞真 bug。


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _load_seed(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_ingest(
    svc: DedupService,
    seed: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    audit = _new_audit()
    audit["raw"] = 0
    audit["engine_ready"] = 0
    skipped: dict[str, int] = {}
    total_t0 = time.perf_counter()
    total_parse = 0.0
    total_insert = 0.0

    items = seed if limit is None else seed[:limit]
    for x in items:
        status = x.get("status", "")
        formula = str(x.get("formula_canonical", "") or "").strip()
        audit["raw"] += 1
        if status == "engine_ready":
            audit["engine_ready"] += 1
        else:
            skipped[status] = skipped.get(status, 0) + 1
        if not formula:
            audit["parse_fail"] += 1
            if len(audit["parse_failure_samples"]) < 20:
                audit["parse_failure_samples"].append(
                    {"factor_id": x.get("factor_id", "?"), "error": "empty formula_canonical"}
                )
            continue
        timing = ingest_one(
            svc,
            formula,
            factor_id=str(x.get("factor_id", "") or f"seed_{x}"),
            canonical_text=formula,
            family_id_from_seed=str(x.get("family") or "") or None,
            audit=audit,
        )
        total_parse += timing["parse"]
        total_insert += timing["insert"]
    total_elapsed = time.perf_counter() - total_t0
    audit["skipped_non_engine_ready"] = skipped
    audit["throughput_ingest_items_per_sec"] = (
        round(len(items) / total_elapsed, 3) if total_elapsed > 0 else 0.0
    )
    audit["timing_seconds"] = {"total": round(total_elapsed, 3)}

    # 守恒校验：raw = fresh_new + family_merge + exact + sign + parse_fail
    const = (
        audit["fresh_new"] + audit["family_merge"] + audit["exact"]
        + audit["sign"] + audit["parse_fail"]
    )
    audit["conservation"] = {
        "raw": audit["raw"],
        "sum_buckets": const,
        "holds": const == audit["raw"],
    }
    return audit


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _pct(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run_benchmark(
    svc: DedupService,
    seed_forms: list[str],
    *,
    bench_n: int = 1000,
) -> dict[str, Any]:
    rnd = random.Random(20260901)
    synthetic = _synthetic_query_pool(seed_forms, bench_n, rnd)

    results: dict[str, Any] = {}

    # 1) ingest 分阶段吞吐
    t_parse: list[float] = []
    t_insert: list[float] = []
    t0 = time.perf_counter()
    parse_t0 = time.perf_counter()
    for f in synthetic:
        try:
            ensure_authority_available()
            from factor_engine.identity import get_factor_identity
        except Exception as _e:  # noqa: BLE001
            results["parse_identity_error"] = str(_e)
            break
        t = time.perf_counter() - parse_t0
        parse_t0 = time.perf_counter()
        t_parse.append(t)
        try:
            _fe_identity_dict(get_factor_identity(f))
        except Exception as _e:  # noqa: BLE001
            results.setdefault("bench_parse_fail", 0)
            results["bench_parse_fail"] += 1
    total_ingest = time.perf_counter() - t0
    results["ingest"] = {
        "count": len(synthetic),
        "elapsed_sec": round(total_ingest, 3),
        "throughput_items_per_sec": round(len(synthetic) / total_ingest, 3) if total_ingest else 0.0,
        "phase_parse_mean_ms": round(statistics.mean(t_parse) * 1000, 4) if t_parse else 0.0,
        "phase_parse_p50_ms": round(_pct(sorted(t_parse), 50) * 1000, 4) if t_parse else 0.0,
        "phase_parse_p95_ms": round(_pct(sorted(t_parse), 95) * 1000, 4) if t_parse else 0.0,
        "phase_insert_mean_ms": round(statistics.mean(t_insert) * 1000, 4) if t_insert else 0.0,
    }

    # 2) exact lookup / family lookup / ANN nearest 延迟（读路径）
    #    exact: store.find_factor_by_hash（真实入库因子）；family: find_family_members；
    #    nearest: nearest_by_fingerprint（先给一批入库因子写入合成 rank fingerprint 建立 ANN）。
    exact_lat: list[float] = []
    family_lat: list[float] = []
    nearest_lat: list[float] = []

    # 收集已入库因子（用于真实查询）
    rows = svc._store.q(
        "SELECT factor_pk, factor_id, identity_version, canonical_ast_hash"
        " FROM seen_factors ORDER BY factor_pk LIMIT ?",
        (bench_n,),
    )
    real_ins = [(int(r[0]), r[1], str(r[2] or "1"), r[3]) for r in rows]

    # exact read-path lookup（按 hash 查）
    for _pk, fid, ver, canon in real_ins if real_ins else rows:
        key = canon or fid
        t0 = time.perf_counter()
        svc._store.find_factor_by_hash(ver, key)
        exact_lat.append((time.perf_counter() - t0) * 1000.0)

    # family lookup（读路径）
    fam_rows = svc._store.q(
        "SELECT parameter_family_id, identity_version FROM seen_factors"
        " WHERE parameter_family_id IS NOT NULL"
        " GROUP BY parameter_family_id LIMIT ?",
        (bench_n,),
    )
    for famval, ver in fam_rows:
        t0 = time.perf_counter()
        svc._store.find_family_members(str(ver or "1"), famval)
        family_lat.append((time.perf_counter() - t0) * 1000.0)

    # ANN nearest：先给一批已入库因子写入合成 rank fingerprint 建立 LSH 索引，
    # 再用真实 fingerprint 查询（模拟 ANN 检索延迟）。
    from alphaprobe.seen.fingerprint import build_rank_fingerprint

    rng = random.Random(7)
    added_fids: list[str] = []
    for pk, fid, _ver, _canon in real_ins[: min(200, len(real_ins))]:
        ranks = {f"S{i}": rng.random() for i in range(200)}
        fp = build_rank_fingerprint([ranks])
        if fp:
            svc._store.upsert_fingerprint(
                pk, fp, "v1", None,
                [int.from_bytes(fp[i * 2:(i + 1) * 2], "big") for i in range(16)],
            )
            added_fids.append(fid)
    fp_rows = svc._store.q(
        "SELECT fp FROM seen_fingerprints LIMIT ?",
        (bench_n,),
    )
    fps = [bytes(r[0]) for r in fp_rows]
    for fpb in fps:
        t0 = time.perf_counter()
        svc.nearest_by_fingerprint(fpb, k=10)
        nearest_lat.append((time.perf_counter() - t0) * 1000.0)
    results["nearest_index_factors"] = len(added_fids)

    def _lat(series: list[float]) -> dict[str, Any]:
        s = sorted(series)
        return {
            "n": len(s),
            "mean_ms": round(statistics.mean(s), 4) if s else 0.0,
            "p50_ms": round(_pct(s, 50), 4),
            "p95_ms": round(_pct(s, 95), 4),
        }

    results["lookup"] = {
        "exact_ms": _lat(exact_lat),
        "family_ms": _lat(family_lat),
        "nearest_ms": _lat(nearest_lat),
    }
    results["synthetic_query_n"] = bench_n
    results["real_ingested_used"] = len(real_ins)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _pretty_result(prefix: str, audit: dict[str, Any]) -> list[str]:
    lines = [
        f"[{prefix}] {b}={audit.get(b)}" for b in _BUCKETS
    ]
    lines.append(
        f"[{prefix}] engine_ready={audit.get('engine_ready')} "
        f"non_engine_ready={audit.get('skipped_non_engine_ready')}"
    )
    cons = audit.get("conservation", {})
    lines.append(
        f"[{prefix}] conservation raw={cons.get('raw')} sum_buckets={cons.get('sum_buckets')}"
        f" holds={cons.get('holds')}"
    )
    if audit.get("throughput_ingest_items_per_sec"):
        lines.append(f"[{prefix}] ingest throughput={audit['throughput_ingest_items_per_sec']} items/s")
    if audit.get("parse_failure_samples"):
        lines.append(f"[{prefix}] parse_fail_samples({len(audit['parse_failure_samples'])}):")
        for s in audit["parse_failure_samples"]:
            lines.append(f"    {s['factor_id']} -> {s['error']}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase D cold-start seed ingest")
    ap.add_argument("--seed", type=str,
                    default="/home/sunhaiwei/quant_projects/cold_start_library/library/factorengine_cold_start_backend_audited_v9.json")
    ap.add_argument("--db", type=str, default="data/seen_index.db")
    ap.add_argument("--limit", type=int, default=None, help="ingest limit (smoke)")
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-n", type=int, default=1000)
    ap.add_argument("--no-save-audit", action="store_true", help="don't persist audit json")
    args = ap.parse_args()

    os.environ.setdefault("ALPHAPROBE_SEEN_DB", args.db)
    ensure_authority_available()  # fail-closed early
    svc = build_seen_index(args.db)
    seed = _load_seed(Path(args.seed))
    print(f"loaded {len(seed)} seed items, statuses: " + str(
        {k: v for k, v in __import__("collections").Counter(
            x.get("status") for x in seed).items()}))
    sys.stdout.flush()

    t0 = time.perf_counter()
    audit = run_ingest(svc, seed, limit=args.limit)
    audit_elapsed = time.perf_counter() - t0
    audit["elapsed_seconds"] = round(audit_elapsed, 3)
    for line in _pretty_result("INGEST", audit):
        print(line)
    sys.stdout.flush()

    # 幂等校验：再跑一遍，新增应为 0
    if args.limit is None:
        audit2 = run_ingest(svc, seed, limit=None)
        for line in _pretty_result("INGEST-RERUN", audit2):
            print(line)
        audit["rerun"] = {
            "fresh_new": audit2["fresh_new"],
            "family_merge": audit2["family_merge"],
            "exact": audit2["exact"],
            "sign": audit2["sign"],
            "parse_fail": audit2["parse_fail"],
            "idempotent_new_created": (audit2["fresh_new"] + audit2["family_merge"]),
        }
        sys.stdout.flush()

    out: dict[str, Any] = {
        "db": os.path.abspath(args.db),
        "seed_path": str(Path(args.seed)),
        "run_id": args.run_id,
        "audit": audit,
        "stats": svc.stats(),
    }
    bench: dict[str, Any] = {}
    if args.benchmark:
        seed_forms = [str(x.get("formula_canonical", "") or "").strip()
                      for x in seed[: args.bench_n] if (x.get("formula_canonical") or "").strip()]
        bench = run_benchmark(svc, seed_forms, bench_n=args.bench_n)
        out["benchmark"] = bench
        print("\n=== BENCHMARK SUMMARY ===")
        if bench.get("ingest"):
            i = bench["ingest"]
            print(f"  ingest throughput: {i['throughput_items_per_sec']} items/s "
                  f"({i['elapsed_sec']}s / {i['count']} items); "
                  f"parse p50={i['phase_parse_p50_ms']}ms p95={i['phase_parse_p95_ms']}ms")
        if bench.get("lookup"):
            lk = bench["lookup"]
            for name in ("exact", "family", "nearest"):
                d = lk.get(f"{name}_ms", {})
                print(f"  lookup[{name}]: p50={d.get('p50_ms')}ms p95={d.get('p95_ms')}ms "
                      f"(n={d.get('n')})")
        sys.stdout.flush()

    # 落 JSON
    if not args.no_save_audit:
        data_dir = Path(args.db).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        audit_path = data_dir / "ingest_audit.json"
        bench_path = data_dir / "ingest_benchmark.json"
        with open(audit_path, "w", encoding="utf-8") as fh:
            json.dump({k: out[k] for k in ("db", "seed_path", "run_id", "audit", "stats")},
                      fh, ensure_ascii=False, indent=2)
        print(f"audit written: {audit_path}")
        if bench:
            with open(bench_path, "w", encoding="utf-8") as fh:
                json.dump(bench, fh, ensure_ascii=False, indent=2)
            print(f"benchmark written: {bench_path}")
    svc._store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
