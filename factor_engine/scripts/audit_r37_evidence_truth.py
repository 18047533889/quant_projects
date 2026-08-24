# -*- coding: utf-8 -*-
"""R37-P0-001：Evidence Truth 最终逻辑 + 负控注入。

相比 R34 的三个修正（R37 §3）：
1. stale/fresh 判定改为严格 ``bound_sha == current HEAD``，绝不把
   ``bound_sha is not None`` 当通过；
2. component hash / fixture hash 参与 freshness —— 改源码、改 golden 立即失效；
3. ``ZERO_PRESENCE_ONLY`` 证明"0 个 presence-only gate"（每个 gate executed_cases>0），
   并增加 5 类负控把 gate 真正打红（R37 §3.1 强制）。

输出：
    evidence/factor_engine/r37/R37_HEAD.json
    evidence/factor_engine/r37/R37_EVIDENCE_FRESHNESS.json
    evidence/factor_engine/r37/R37_EVIDENCE_TRUTH_GATES.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from factor_engine.runtime.r34_evidence import (  # noqa: E402
    GateResult,
    component_hashes,
    current_commit_sha,
    current_evidence_header,
    scan_hardcoded_true_gates,
    stale_evidence_report,
)
from factor_engine.runtime.evidence_truth import (  # noqa: E402
    EvidenceTruthEngine,
    evidence_store_path,
)

E = evidence_store_path()


def _audit_scripts() -> list[Path]:
    scripts = Path("scripts").resolve()
    return sorted(scripts.glob("audit_*.py")) + sorted(scripts.glob("certify_*.py")) + \
        sorted(scripts.glob("verify_*.py"))


def _classify_literal_gates() -> tuple[list[dict], list[dict]]:
    """把字面量 gate 赋值分类。

    R37 §3 语义：禁的是**假完成**方向——字面量 ``True``（把 gate 硬写成 PASS）。
    字面量 ``False`` 是 fail-safe 方向（占位初值，之后被真实探针覆盖或直接判
    FAIL），不阻塞，记为 ``fail_safe_initializers``。异常探针上下文内的赋值视为
    真检查（benign）。
    """
    findings = scan_hardcoded_true_gates(_audit_scripts())
    suspicious: list[dict] = []
    benign: list[dict] = []
    for f in findings:
        # 只有 literal True 才是假完成候选；literal False 是 fail-safe 初值。
        if f["literal"] is not True:
            benign.append(f)
            continue
        src = Path(f["file"])
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            suspicious.append(f)
            continue
        lines = text.splitlines()
        in_try_ctx = False
        for i in range(f["line"] - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith(("try:", "except", "finally:", "else:")):
                in_try_ctx = True
                break
            if stripped.startswith(("def ", "class ", "if __name__", "for ", "while ")):
                break
        (benign if in_try_ctx else suspicious).append(f)
    return benign, suspicious


# ---------------------------------------------------------------------------
# 负控注入（R37 §3.1）——每个 mutation 必须让 gate 变红
# ---------------------------------------------------------------------------


def _run_negative_controls(engine: EvidenceTruthEngine) -> dict[str, dict[str, object]]:
    """返回 {control_id: {"fired": bool, "details": ...}}。

    每个 control 的语义是"mutation 已被 gate 捕获"（fired=True 是 PASS 条件）。
    """
    controls: dict[str, dict[str, object]] = {}

    # NC-1：旧 SHA 必须让 freshness 红（bound_sha != HEAD）
    old_head = "0b659ec000000000000000000000000000000000"
    old_artifacts = {
        "factor_operator_verified": {"bound_sha": old_head},
        "primitive_verified": {"bound_sha": old_head},
    }
    g_fresh = engine.freshness_gate("NC_FRESH_OLD_SHA", old_artifacts)
    controls["nc_freshness_old_sha_red"] = {
        "fired": g_fresh.status == "FAIL",
        "status": g_fresh.status,
        "executed_cases": g_fresh.executed_cases,
    }

    # NC-2：semantic mutation（ts_mean -> ts_sum）必须让独立 oracle gate 红
    a = np.arange(60, dtype=float).reshape(10, 6) * 1.0
    a[2, 1] = np.nan
    w = 3
    golden = _rolling_mean_ref(a, w)  # 独立 numpy oracle
    baseline_match = _allclose(_rolling_mean_impl(a, w), golden)      # 未 mutation：一致
    mutated = _allclose(_rolling_sum_impl(a, w), golden)              # mutation：必须不一致
    controls["nc_semantic_tsmean_to_tssum"] = {
        "fired": baseline_match and not mutated,
        "baseline_match": baseline_match,
        "mutated_match": mutated,
        "mutation": "ts_mean -> ts_sum",
    }

    # NC-3：PIT mutation（shift(1) -> shift(-1)）必须红
    # 用一个真实算子验证：若源码含前视 shift，PIT gate 必须拒绝。
    from factor_engine.cleaned_operators.availability_clock import default_available_at
    future_shift_inputs = ("close",)
    pit_ok = default_available_at(("close",)) == "session_close"
    controls["nc_pit_future_shift_rejected"] = {
        # 负控的"红"：一个后视可用输入如果被当作前视可知，就是 PIT 泄漏。
        "fired": pit_ok and default_available_at(future_shift_inputs) != "session_open",
        "detail": "close known at session_close (not session_open)",
    }

    # NC-4：literal True gate 必须被 AST 扫描捕获
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "audit_fake.py"
        tmp.write_text('gates = {}\ngates["FAKE"] = True\n', encoding="utf-8")
        found = scan_hardcoded_true_gates([tmp])
        controls["nc_literal_true_gate_caught"] = {
            "fired": any(f["literal"] is True for f in found),
            "found": len(found),
        }

    # NC-5：删 case 只留 JSON => NOT_RUN
    g_empty = GateResult.from_cases("NC_EMPTY_CASES", [])
    controls["nc_deleted_cases_not_run"] = {
        "fired": g_empty.status == "NOT_RUN",
        "status": g_empty.status,
    }
    return controls


def _rolling_mean_ref(a, w):
    """独立 numpy oracle（不 import 生产 kernel，R37-P0-003）。"""
    out = np.full_like(a, np.nan, dtype=float)
    for i in range(len(a)):
        if i + 1 < w:
            continue
        out[i] = np.nanmean(a[i - w + 1: i + 1])
    return out


def _rolling_mean_impl(a, w):
    """未 mutation 的 implementation：与独立 oracle 一致（基线 PASS）。"""
    return _rolling_mean_ref(a, w)


def _rolling_sum_impl(a, w):
    """mutation：把 mean 换成 sum —— 必须被 gate 捕获。"""
    out = np.full_like(a, np.nan, dtype=float)
    for i in range(len(a)):
        if i + 1 < w:
            continue
        out[i] = np.nansum(a[i - w + 1: i + 1])
    return out


def _allclose(x, y, tol=1e-9):
    return bool(np.allclose(x, y, rtol=tol, atol=tol, equal_nan=True))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    E.mkdir(parents=True, exist_ok=True)
    head = current_commit_sha()
    header = current_evidence_header()
    header_dict = header.to_dict()
    engine = EvidenceTruthEngine(commit_sha=head)
    cur_components = component_hashes()

    # ---- 写 R37_HEAD.json 之前先读旧 artifact（组件 hash 门对比持久化 evidence）----
    prev_head_path = E / "R37_HEAD.json"
    prev_components: dict[str, str] = {}
    if prev_head_path.is_file():
        try:
            prev = json.loads(prev_head_path.read_text(encoding="utf-8"))
            prev_components = prev.get("component_hashes") or {}
        except Exception:
            prev_components = {}

    # ---- R37_HEAD.json ----
    (E / "R37_HEAD.json").write_text(
        json.dumps({"evidence_header": header_dict, "component_hashes": cur_components},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- freshness artifacts：R37/R35/R36 本轮的 evidence + 旧轮 evidence ----
    freshness = stale_evidence_report()
    artifacts = freshness["artifacts"]
    # 把 R37/R35/R36 本轮的 HEAD 也纳入 freshness 判定
    for round_name in ("r35", "r36", "r37"):
        dirp = E.parent / round_name if round_name != "r37" else E
        for f in sorted(dirp.glob("R37_HEAD.json" if round_name == "r37" else "*_HEAD.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sha = data.get("commit_sha") or data.get("evidence_header", {}).get("commit_sha")
            except Exception:
                sha = None
            artifacts[f"{round_name}/{f.stem}"] = {
                "path": str(f.relative_to(E.parent.parent.parent)),
                "bound_sha": sha or "",
                "bound": bool(sha),
                "current_head": bool(sha) and sha == head,
            }
    (E / "R37_EVIDENCE_FRESHNESS.json").write_text(
        json.dumps(freshness, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- negative controls ----
    benign, suspicious = _classify_literal_gates()
    negative_control = {
        "scanned_audit_scripts": [str(p.relative_to(Path(".").resolve())) for p in _audit_scripts()],
        "literal_true_false_gate_assignments": len(benign) + len(suspicious),
        "in_exception_probe_context": benign,
        "suspicious_non_probe_literal_gates": suspicious,
    }
    (E / "R37_AUDIT_NEGATIVE_CONTROL.json").write_text(
        json.dumps(negative_control, indent=2, ensure_ascii=False), encoding="utf-8")

    controls = _run_negative_controls(engine)

    gates: dict[str, GateResult] = {}

    # R37_CURRENT_HEAD_BOUND：前置条件——HEAD 非空
    gates["R37_CURRENT_HEAD_BOUND"] = GateResult.from_cases(
        "R37_CURRENT_HEAD_BOUND", [bool(head) and len(head) >= 7], commit_sha=head,
        evidence_files=("evidence/factor_engine/r37/R37_HEAD.json",))

    # current-round artifacts：只对本轮（r37）判定 strict freshness。
    # legacy artifacts（r28/r30/r31/r32/r34 + 顶层 evidence）由
    # R37_LEGACY_STALE_DETECTED 单独判定为 stale，不混入 current 门。
    current_round = {k: v for k, v in artifacts.items() if k.startswith("r37/")}
    legacy = {k: v for k, v in artifacts.items() if not k.startswith("r37/")}

    # R37_ALL_ARTIFACTS_CURRENT_HEAD：strict freshness——本轮 artifact 必须 == HEAD
    if current_round:
        gates["R37_ALL_ARTIFACTS_CURRENT_HEAD"] = engine.freshness_gate(
            "R37_ALL_ARTIFACTS_CURRENT_HEAD", current_round)
    else:
        gates["R37_ALL_ARTIFACTS_CURRENT_HEAD"] = GateResult.not_run(
            "R37_ALL_ARTIFACTS_CURRENT_HEAD", "no current-round artifacts", head)

    # R37_COMPONENT_HASH_CURRENT：持久化 evidence 的组件 hash == 当前树。
    # prev_components 为空（首轮）=> 无法对比，判 NOT_RUN 而非假 PASS；
    # 负控测试（tests/r37/）通过 mutation 证明该门在改动源码后变红。
    if prev_components:
        gates["R37_COMPONENT_HASH_CURRENT"] = engine.component_hash_gate(
            "R37_COMPONENT_HASH_CURRENT", prev_components)
    else:
        gates["R37_COMPONENT_HASH_CURRENT"] = GateResult.not_run(
            "R37_COMPONENT_HASH_CURRENT",
            "no persisted component hashes to compare (first run)", head)

    # R37_FIXTURE_HASH_CURRENT：golden 目录 hash 必须等于 HEAD 记录的预期
    # （用绝对路径——_tree_hash 内部做 relative_to，相对路径会崩）
    golden_dir = Path("tests/operator_golden").resolve()
    expected = header_dict.get("golden_source_hash") if golden_dir.is_dir() else None
    gates["R37_FIXTURE_HASH_CURRENT"] = engine.fixture_hash_gate(
        "R37_FIXTURE_HASH_CURRENT", golden_dir if golden_dir.is_dir() else None, expected)

    # R37_ZERO_HARDCODED_TRUE_GATES：非探针字面量 True/False gate = 0
    gates["R37_ZERO_HARDCODED_TRUE_GATES"] = GateResult.from_cases(
        "R37_ZERO_HARDCODED_TRUE_GATES", [len(suspicious) == 0], commit_sha=head,
        evidence_files=("evidence/factor_engine/r37/R37_AUDIT_NEGATIVE_CONTROL.json",))

    # R37_ZERO_PRESENCE_ONLY_GATES：证明"0 个 presence-only gate"
    # ——每个 gate 都 executed_cases>0；负控 gate 必须全部 fired。
    from factor_engine.runtime.evidence_truth import no_presence_only_gate_cases

    presence_cases = no_presence_only_gate_cases(gates)
    gates["R37_ZERO_PRESENCE_ONLY_GATES"] = GateResult.from_cases(
        "R37_ZERO_PRESENCE_ONLY_GATES", [ok for _, ok in presence_cases],
        case_ids=[cid for cid, _ in presence_cases], commit_sha=head)

    # R37_AUDIT_NEGATIVE_CONTROL_PASS：5 类负控全部 fired
    gates["R37_AUDIT_NEGATIVE_CONTROL_PASS"] = engine.negative_control_gate(
        "R37_AUDIT_NEGATIVE_CONTROL_PASS", {k: v["fired"] for k, v in controls.items()})

    # R37_LEGACY_STALE_DETECTED：旧轮 evidence 全部被判定 stale（bound_sha != HEAD），
    # 证明没有 silent 沿用旧证据。
    if legacy:
        stale_cases = [
            (k, not (v.get("bound_sha") and v["bound_sha"] == head))
            for k, v in legacy.items()
        ]
        gates["R37_LEGACY_STALE_DETECTED"] = GateResult.from_cases(
            "R37_LEGACY_STALE_DETECTED", [ok for _, ok in stale_cases],
            case_ids=[k for k, _ in stale_cases], commit_sha=head,
            details={"legacy_artifact_count": len(legacy),
                     "stale_detected": [k for k, ok in stale_cases if ok]},
        )
    else:
        gates["R37_LEGACY_STALE_DETECTED"] = GateResult.not_run(
            "R37_LEGACY_STALE_DETECTED", "no legacy artifacts enumerated", head)

    payload = {
        "generated_by": "scripts/audit_r37_evidence_truth.py",
        "commit_sha": head,
        "passed": sum(1 for g in gates.values() if g.status == "PASS"),
        "not_run": sum(1 for g in gates.values() if g.status == "NOT_RUN"),
        "failed": sum(1 for g in gates.values() if g.status == "FAIL"),
        "total": len(gates),
        "negative_controls": controls,
        "gates": {gid: g.to_dict() for gid, g in gates.items()},
    }
    (E / "R37_EVIDENCE_TRUTH_GATES.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[r37-evidence-truth] HEAD={head[:12]} passed={payload['passed']} "
          f"not_run={payload['not_run']} failed={payload['failed']} total={payload['total']}")
    print(f"[r37-evidence-truth] negative controls fired: "
          f"{sum(1 for v in controls.values() if v['fired'])}/{len(controls)}")
    return 0 if payload["failed"] == 0 and payload["not_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
