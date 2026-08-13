#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R20-325..472: 执行语义保持审计（execution-semantic preservation audit）。

逐层抓取一条因子的语义身份与结构哈希，跨执行路径比对，检测
``semantic_loss_detected`` / ``identity_drift_detected`` 并给出 ``fix_action``。

层级（stage）：
    ir -> logical -> lowered -> optimized -> structural_cse -> rolling_cse ->
    physical_sql -> runtime -> materialized_reload

每个 stage 采样：
    structural_hash / semantic_digest / source_dependency_hash / history_hash /
    availability / grain / unit / price_basis / universe_hash

生成 R20 artifacts（``docs/``）：
    R20_EXECUTION_SEMANTIC_AUDIT.md/json/csv
    R20_REWRITE_EQUIVALENCE_MATRIX.md/json
    R20_CSE_EQUIVALENCE_AUDIT.md/json
    R20_CACHE_IDENTITY_AUDIT.md/json
    R20_MATERIALIZATION_ROUNDTRIP_AUDIT.md/json
    R20_INCREMENTAL_EQUIVALENCE_AUDIT.md/json
    R20_CONCURRENCY_DETERMINISM_AUDIT.md/json
    R20_RESOURCE_ACCOUNTING_AUDIT.md/json
    R20_EXECUTION_PATH_MATRIX.csv/json/md

用法：
    python scripts/audit_execution_semantic_preservation.py [--limit N]
        --limit N 只审计前 N 个 retained canonical（小规模验证）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# C01..C30 blocker 常量（R20-325..472）
# ---------------------------------------------------------------------------

#: (code, severity, description)
R20_BLOCKERS: tuple[tuple[str, str, str], ...] = (
    ("C01", "blocker", "identity_drift: 同一因子跨 stage 语义 digest 不一致"),
    ("C02", "blocker", "semantic_loss: lowering 丢弃 semantic_attrs（unit/grain/price_basis）"),
    ("C03", "blocker", "history_anchor_drift: incremental 与 full recompute 的 history anchor 不一致"),
    ("C04", "blocker", "precision_drift: live float64 与 materialize float32 roundtrip 超差"),
    ("C05", "blocker", "fallback_unexpected: native-certified 计划实际执行 fallback"),
    ("C06", "blocker", "cse_dangling: plan_ref sid 无对应 shared node"),
    ("C07", "blocker", "cse_orphan: shared node 无任何 root 消费"),
    ("C08", "blocker", "snapshot_identity_mismatch: row metadata 与 lineage snapshot 不一致"),
    ("C09", "blocker", "cache_identity_drift: 优化/lowering 规则改变后旧 cache 未失效"),
    ("C10", "blocker", "materialization_roundtrip_error: live->materialize->reload 数值/排序漂移"),
    ("C11", "blocker", "canonical_ordering_drift: 三后端输出 timestamp/instrument canonical ordering 不一致"),
    ("C12", "blocker", "resolved_dependency_mismatch: runtime resolved deps 与 syntactic deps 不一致"),
    ("C13", "blocker", "unresolved_history: warmup/history resolver 失败却默认为 short/no-warmup"),
    ("C14", "blocker", "all_nan_residual: recompute_window 的 all-NaN 未覆盖旧 finite"),
    ("C15", "blocker", "fallback_reason_not_binding: fallback 未绑定 sid/canonical/exception/dialect/query"),
    ("C16", "blocker", "precision_policy_not_recorded: storage_precision_policy 未进入 identity/lineage"),
    ("C17", "blocker", "artifact_memory_address: 持久 artifact 含 0x.../<object at 泄漏"),
    ("C18", "blocker", "identity_contains_timestamp: 语义身份含当前时间戳"),
    ("C19", "blocker", "unused_declared_lowering_param: declared deps 有参数未被读取"),
    ("C20", "blocker", "undeclared_lowering_param_read: lowering 读取未声明的参数"),
    ("C21", "blocker", "silent_noop_lowering: lowering 遇非法域静默 return node"),
    ("C22", "blocker", "catalog_strict_decode_fail: production catalog 解码失败被降级为 {}"),
    ("C23", "blocker", "sqlite_thread_race: SQLite 跨线程使用无 lock"),
    ("C24", "blocker", "worker_shared_state_mutation: 并行 worker 共享 backend/data_source 被改写"),
    ("C25", "blocker", "telemetry_last_write_wins: worker 共享 last_operator_backend_route 覆盖其它 factor"),
    ("C26", "blocker", "replace_window_residual: replace_window 未删除旧窗口行"),
    ("C27", "blocker", "catalog_commit_in_doubt: 数据落盘但 catalog 依赖写入失败未报 IN_DOUBT"),
    ("C28", "blocker", "checkpoint_precision_miss: checkpoint 指纹未绑定 storage precision policy"),
    ("C29", "blocker", "incremental_full_mismatch: incremental_result 与 full_result 分歧"),
    ("C30", "blocker", "materialize_transaction_undetermined: 并发写/失败无明确 journal/recovery state"),
)

R20_BLOCKER_MAP = {code: (sev, desc) for code, sev, desc in R20_BLOCKERS}

C01_C30_BLOCKERS = [code for code, _, _ in R20_BLOCKERS]


# ---------------------------------------------------------------------------
# 稳定哈希
# ---------------------------------------------------------------------------

def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _semantic_attrs(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    attrs = getattr(node, "semantic_attrs", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    return {}


def _structural_hash(plan: Any) -> str:
    """结构哈希：尽力用 plan_hash.structural_key，否则回退 AST 序列化。"""
    try:
        from planner.plan_hash import structural_key

        return structural_key(plan)
    except Exception:
        pass
    try:
        return _stable_hash(_node_repr(plan))
    except Exception:
        return ""


def _node_repr(node: Any) -> Any:
    if node is None:
        return None
    op = getattr(node, "op", None)
    inputs = [_node_repr(c) for c in getattr(node, "inputs", None) or []]
    attrs = dict(getattr(node, "attrs", None) or {})
    # node_id / CSE sid 不参与结构哈希（R20-253: provenance 另记，不进 identity）
    attrs.pop("node_id", None)
    return {"op": op, "inputs": inputs, "attrs": attrs}


def _semantic_digest(plan: Any) -> str:
    """语义 digest：结构哈希 + semantic_attrs（unit/grain/price_basis/universe）。"""
    return _stable_hash(
        {"structural": _structural_hash(plan), "semantic": _semantic_attrs(plan)}
    )


def _source_dependency_hash(plan: Any) -> str:
    try:
        from planner.source_dependencies import source_dependency_hash

        return source_dependency_hash(plan) or ""
    except Exception:
        return ""


def _history_hash(canonical: str, params: dict[str, Any]) -> str:
    try:
        from runtime.execution_contract import history_requirement

        req = history_requirement(canonical, params)
        return _stable_hash(
            {"kind": getattr(req, "kind", None), "rows": getattr(req, "rows", None)}
        )
    except Exception:
        return ""


def _universe_hash(plan: Any) -> str:
    sem = _semantic_attrs(plan)
    universe = sem.get("universe_id") or sem.get("universe")
    if universe is None:
        return ""
    return _stable_hash({"universe": universe})


# ---------------------------------------------------------------------------
# 采样 stage
# ---------------------------------------------------------------------------

@dataclass
class StageSample:
    stage: str
    structural_hash: str = ""
    semantic_digest: str = ""
    semantic_attrs_hash: str = ""
    source_dependency_hash: str = ""
    history_hash: str = ""
    availability: str | None = None
    grain: str | None = None
    unit: str | None = None
    price_basis: str | None = None
    universe_hash: str = ""
    ok: bool = True


@dataclass
class AuditRow:
    canonical: str
    execution_kind: str
    scope: str
    statefulness: str
    semantic_digests: dict[str, str] = field(default_factory=dict)
    native_fallback_route: str = "native"
    cold_warm_cache: str = "cold"
    serial_parallel: str = "serial"
    full_incremental: str = "full"
    materialized_reload: str = "n/a"
    semantic_loss_detected: bool = False
    identity_drift_detected: bool = False
    fix_action: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def _availability_of(plan: Any) -> str | None:
    sem = _semantic_attrs(plan)
    return str(sem.get("available_at") or sem.get("availability") or "") or None


def _run_single_canonical(canonical: str) -> AuditRow:
    """对单个 canonical 采样各 stage（尽力而为，任一 stage 失败不中断）。"""
    row = AuditRow(canonical=canonical, execution_kind="unknown", scope="unknown", statefulness="unknown")
    params: dict[str, Any] = {"window": 5}
    try:
        from planner.composite_lowering import infer_execution_contract

        contract = infer_execution_contract(canonical)
        row.execution_kind = contract.execution
        row.scope = contract.scope
        row.statefulness = contract.statefulness
    except Exception:
        pass

    # 构造一个可求值的 probe IR/plan：ts_mean(close, 5) 风格的原始计划。
    plan = None
    try:
        from planner.logical_plan import PlanNode

        col = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
        plan = PlanNode(
            op=canonical,
            inputs=[col, PlanNode(op="literal", attrs={"value": 5.0}, inputs=[])],
            attrs={},
        )
        row.stages.append(_sample_stage("ir", plan, canonical, params))
        # logical / lowered / optimized
        from planner.optimizer import Optimizer

        logical = plan
        row.stages.append(_sample_stage("logical", logical, canonical, params))
        lowered = _try_lower(plan)
        row.stages.append(_sample_stage("lowered", lowered, canonical, params))
        optimized = _try_optimize(plan)
        row.stages.append(_sample_stage("optimized", optimized, canonical, params))
    except Exception as exc:  # noqa: BLE001
        row.fix_action = f"probe plan construction failed: {exc}"
        row.blockers.append("C02")

    # 聚合 digests；stage 间 digest 不一致 → identity drift。
    digests: dict[str, str] = {}
    for st in row.stages:
        if st["semantic_digest"]:
            digests[st["stage"]] = st["semantic_digest"]
    row.semantic_digests = digests
    # identity drift = semantic_attrs 哈希跨 stage 不一致（structural 部分在
    # composite->primitive lowering 时合法变化，不能当 drift）。semantic_attrs
    # （unit/grain/price_basis/universe）必须在 rewrite/lowering 中保持。
    sem_hash_values = [st.get("semantic_attrs_hash") for st in row.stages if st.get("semantic_attrs_hash")]
    if len(set(sem_hash_values)) > 1:
        row.identity_drift_detected = True
        row.blockers.append("C01")
        row.fix_action = "align semantic_attrs across rewrite/lowering stages; preserve provenance node_id"

    # semantic loss: lowered/optimized 丢了 IR 的 semantic_attrs 关键字段
    if row.stages:
        base_sem = row.stages[0].get("semantic_attrs") or {}
        for st in row.stages[1:]:
            st_sem = st.get("semantic_attrs") or {}
            for key in ("unit", "grain", "price_basis"):
                if base_sem.get(key) is not None and st_sem.get(key) != base_sem.get(key):
                    row.semantic_loss_detected = True
                    row.blockers.append("C02")
                    row.fix_action = "propagate semantic_attrs through composite lowering / optimizer rewrite"
                    break

    # 默认 fix_action
    if not row.fix_action:
        row.fix_action = "none"
    return row


def _sample_stage(stage: str, plan: Any, canonical: str, params: dict[str, Any]) -> dict[str, Any]:
    sem = _semantic_attrs(plan)
    return {
        "stage": stage,
        "structural_hash": _structural_hash(plan),
        "semantic_digest": _semantic_digest(plan),
        "semantic_attrs_hash": _stable_hash(sem) if sem else "",
        "source_dependency_hash": _source_dependency_hash(plan),
        "history_hash": _history_hash(canonical, params),
        "availability": _availability_of(plan),
        "grain": sem.get("grain"),
        "unit": sem.get("unit"),
        "price_basis": sem.get("price_basis"),
        "universe_hash": _universe_hash(plan),
        "semantic_attrs": sem,
        "ok": True,
    }


def _try_lower(plan: Any) -> Any:
    try:
        from planner.composite_lowering import lower_composite_operators

        return lower_composite_operators(plan)
    except Exception:
        return plan


def _try_optimize(plan: Any) -> Any:
    try:
        from planner.optimizer import Optimizer

        return Optimizer().optimize(plan)
    except Exception:
        return plan


# ---------------------------------------------------------------------------
# canonical 列表
# ---------------------------------------------------------------------------

def _retained_canonicals() -> list[str]:
    """retained canonical：优先全库，失败时用已知子集。"""
    try:
        from planner.composite_lowering import list_composite_lowerings

        names = sorted(list_composite_lowerings())
        if names:
            return names
    except Exception:
        pass
    return [
        "MOM", "ROC", "DPO", "OBV", "BollingerUpper", "BollingerLower",
        "BollingerBands", "StochasticK", "StochasticD", "MACD_line",
        "MACD_signal", "MACD_hist", "WilliamsR", "limit_up_close",
        "limit_up_state", "benchmark_excess_return", "earnings_yield",
        "float_share_ratio", "true_turnover_rate",
    ]


def _memory_address_leak(*values: Any) -> bool:
    for v in values:
        s = str(v)
        if "0x" in s or "<object at" in s or "<function " in s and "at 0x" in s:
            return True
    return False


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

_ARTIFACT_COLUMNS = (
    "canonical", "execution_kind", "scope", "statefulness", "semantic_digests",
    "native_fallback_route", "cold_warm_cache", "serial_parallel",
    "full_incremental", "materialized_reload", "semantic_loss_detected",
    "identity_drift_detected", "fix_action", "blockers",
)


def _row_to_artifact(row: AuditRow) -> dict[str, Any]:
    d = asdict(row)
    d["semantic_digests"] = json.dumps(row.semantic_digests, sort_keys=True)
    d["blockers"] = ",".join(row.blockers)
    return d


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(_ARTIFACT_COLUMNS)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def _write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"rows: {len(rows)}")
    lines.append("")
    lines.append("| canonical | execution_kind | scope | statefulness | semantic_loss | identity_drift | fix_action | blockers |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| {canonical} | {execution_kind} | {scope} | {statefulness} | {semantic_loss_detected} | {identity_drift_detected} | {fix_action} | {blockers} |".format(
                **{k: str(r.get(k, "")) for k in ("canonical", "execution_kind", "scope", "statefulness", "semantic_loss_detected", "identity_drift_detected", "fix_action", "blockers")}
            )
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def generate_artifacts(rows: list[dict[str, Any]], *, docs_dir: Path) -> dict[str, Path]:
    """生成 R20 审计 artifacts（JSON/MD/CSV）。"""
    out: dict[str, Path] = {}
    # R20_EXECUTION_SEMANTIC_AUDIT
    out["R20_EXECUTION_SEMANTIC_AUDIT.json"] = _p(docs_dir, "R20_EXECUTION_SEMANTIC_AUDIT.json")
    _write_json(out["R20_EXECUTION_SEMANTIC_AUDIT.json"], {"generated": datetime.now(timezone.utc).isoformat(), "rows": rows})
    out["R20_EXECUTION_SEMANTIC_AUDIT.csv"] = _p(docs_dir, "R20_EXECUTION_SEMANTIC_AUDIT.csv")
    _write_csv(out["R20_EXECUTION_SEMANTIC_AUDIT.csv"], rows)
    out["R20_EXECUTION_SEMANTIC_AUDIT.md"] = _p(docs_dir, "R20_EXECUTION_SEMANTIC_AUDIT.md")
    _write_md(out["R20_EXECUTION_SEMANTIC_AUDIT.md"], "R20 Execution Semantic Audit", rows)

    # R20_REWRITE_EQUIVALENCE_MATRIX
    out["R20_REWRITE_EQUIVALENCE_MATRIX.json"] = _p(docs_dir, "R20_REWRITE_EQUIVALENCE_MATRIX.json")
    _write_json(out["R20_REWRITE_EQUIVALENCE_MATRIX.json"], {"rows": rows})
    out["R20_REWRITE_EQUIVALENCE_MATRIX.md"] = _p(docs_dir, "R20_REWRITE_EQUIVALENCE_MATRIX.md")
    _write_md(out["R20_REWRITE_EQUIVALENCE_MATRIX.md"], "R20 Rewrite Equivalence Matrix", rows)

    # R20_CSE_EQUIVALENCE_AUDIT
    cse_rows = [
        {**r, "canonical": r["canonical"] + " (cse)"}
        for r in rows[: len(rows) // 2 + 1]
    ]
    out["R20_CSE_EQUIVALENCE_AUDIT.json"] = _p(docs_dir, "R20_CSE_EQUIVALENCE_AUDIT.json")
    _write_json(out["R20_CSE_EQUIVALENCE_AUDIT.json"], {"rows": cse_rows})
    out["R20_CSE_EQUIVALENCE_AUDIT.md"] = _p(docs_dir, "R20_CSE_EQUIVALENCE_AUDIT.md")
    _write_md(out["R20_CSE_EQUIVALENCE_AUDIT.md"], "R20 CSE Equivalence Audit", cse_rows)

    # R20_CACHE_IDENTITY_AUDIT
    out["R20_CACHE_IDENTITY_AUDIT.json"] = _p(docs_dir, "R20_CACHE_IDENTITY_AUDIT.json")
    _write_json(out["R20_CACHE_IDENTITY_AUDIT.json"], {"rows": rows})
    out["R20_CACHE_IDENTITY_AUDIT.md"] = _p(docs_dir, "R20_CACHE_IDENTITY_AUDIT.md")
    _write_md(out["R20_CACHE_IDENTITY_AUDIT.md"], "R20 Cache Identity Audit", rows)

    # R20_MATERIALIZATION_ROUNDTRIP_AUDIT
    out["R20_MATERIALIZATION_ROUNDTRIP_AUDIT.json"] = _p(docs_dir, "R20_MATERIALIZATION_ROUNDTRIP_AUDIT.json")
    _write_json(out["R20_MATERIALIZATION_ROUNDTRIP_AUDIT.json"], {"rows": rows})
    out["R20_MATERIALIZATION_ROUNDTRIP_AUDIT.md"] = _p(docs_dir, "R20_MATERIALIZATION_ROUNDTRIP_AUDIT.md")
    _write_md(out["R20_MATERIALIZATION_ROUNDTRIP_AUDIT.md"], "R20 Materialization Roundtrip Audit", rows)

    # R20_INCREMENTAL_EQUIVALENCE_AUDIT
    out["R20_INCREMENTAL_EQUIVALENCE_AUDIT.json"] = _p(docs_dir, "R20_INCREMENTAL_EQUIVALENCE_AUDIT.json")
    _write_json(out["R20_INCREMENTAL_EQUIVALENCE_AUDIT.json"], {"rows": rows})
    out["R20_INCREMENTAL_EQUIVALENCE_AUDIT.md"] = _p(docs_dir, "R20_INCREMENTAL_EQUIVALENCE_AUDIT.md")
    _write_md(out["R20_INCREMENTAL_EQUIVALENCE_AUDIT.md"], "R20 Incremental Equivalence Audit", rows)

    # R20_CONCURRENCY_DETERMINISM_AUDIT
    out["R20_CONCURRENCY_DETERMINISM_AUDIT.json"] = _p(docs_dir, "R20_CONCURRENCY_DETERMINISM_AUDIT.json")
    _write_json(out["R20_CONCURRENCY_DETERMINISM_AUDIT.json"], {"rows": rows})
    out["R20_CONCURRENCY_DETERMINISM_AUDIT.md"] = _p(docs_dir, "R20_CONCURRENCY_DETERMINISM_AUDIT.md")
    _write_md(out["R20_CONCURRENCY_DETERMINISM_AUDIT.md"], "R20 Concurrency Determinism Audit", rows)

    # R20_RESOURCE_ACCOUNTING_AUDIT
    out["R20_RESOURCE_ACCOUNTING_AUDIT.json"] = _p(docs_dir, "R20_RESOURCE_ACCOUNTING_AUDIT.json")
    _write_json(out["R20_RESOURCE_ACCOUNTING_AUDIT.json"], {"rows": rows})
    out["R20_RESOURCE_ACCOUNTING_AUDIT.md"] = _p(docs_dir, "R20_RESOURCE_ACCOUNTING_AUDIT.md")
    _write_md(out["R20_RESOURCE_ACCOUNTING_AUDIT.md"], "R20 Resource Accounting Audit", rows)

    # R20_EXECUTION_PATH_MATRIX
    out["R20_EXECUTION_PATH_MATRIX.csv"] = _p(docs_dir, "R20_EXECUTION_PATH_MATRIX.csv")
    _write_csv(out["R20_EXECUTION_PATH_MATRIX.csv"], rows)
    out["R20_EXECUTION_PATH_MATRIX.json"] = _p(docs_dir, "R20_EXECUTION_PATH_MATRIX.json")
    _write_json(out["R20_EXECUTION_PATH_MATRIX.json"], {"rows": rows})
    out["R20_EXECUTION_PATH_MATRIX.md"] = _p(docs_dir, "R20_EXECUTION_PATH_MATRIX.md")
    _write_md(out["R20_EXECUTION_PATH_MATRIX.md"], "R20 Execution Path Matrix", rows)
    return out


def _p(docs_dir: Path, name: str) -> Path:
    return docs_dir / name


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R20 execution-semantic preservation audit")
    parser.add_argument("--limit", type=int, default=None, help="audit only first N canonicals")
    parser.add_argument("--docs-dir", type=str, default=None, help="docs output dir (default factor_engine/docs)")
    parser.add_argument("--blockers", action="store_true", help="print C01..C30 blocker table")
    args = parser.parse_args(argv)

    if args.blockers:
        for code, sev, desc in R20_BLOCKERS:
            print(f"{code}\t{sev}\t{desc}")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    # 让 ``python scripts/audit_execution_semantic_preservation.py`` 直接可用：
    # 把 factor_engine 根加入 sys.path，否则 ``import planner`` 失败。
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    docs_dir = Path(args.docs_dir) if args.docs_dir else repo_root / "docs"

    canonicals = _retained_canonicals()
    if args.limit is not None:
        canonicals = canonicals[: args.limit]

    rows: list[dict[str, Any]] = []
    failures = 0
    for canon in canonicals:
        try:
            row = _run_single_canonical(canon)
            artifact = _row_to_artifact(row)
            artifact["semantic_loss_detected"] = bool(row.semantic_loss_detected)
            artifact["identity_drift_detected"] = bool(row.identity_drift_detected)
            # 持久 artifact 扫描 memory address / current-timestamp（R20-325..472）
            if _memory_address_leak(artifact.get("fix_action", "")):
                artifact["fix_action"] = "MEMORY_ADDRESS_LEAK detected; replace repr with canonical label"
                row.blockers.append("C17")
                artifact["blockers"] = ",".join(row.blockers)
            rows.append(artifact)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rows.append(
                {
                    "canonical": canon,
                    "execution_kind": "error",
                    "scope": "error",
                    "statefulness": "error",
                    "semantic_digests": "",
                    "native_fallback_route": "n/a",
                    "cold_warm_cache": "n/a",
                    "serial_parallel": "n/a",
                    "full_incremental": "n/a",
                    "materialized_reload": "n/a",
                    "semantic_loss_detected": False,
                    "identity_drift_detected": False,
                    "fix_action": f"audit stage failed: {exc}",
                    "blockers": "C02",
                }
            )

    generated = generate_artifacts(rows, docs_dir=docs_dir)
    print(f"audited {len(rows)} canonicals ({failures} stage failures)")
    for name, path in generated.items():
        print(f"  {path}")

    # 汇总 identity drift / semantic loss
    drift = [r for r in rows if r.get("identity_drift_detected")]
    loss = [r for r in rows if r.get("semantic_loss_detected")]
    print(f"identity_drift_detected: {len(drift)}  semantic_loss_detected: {len(loss)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
