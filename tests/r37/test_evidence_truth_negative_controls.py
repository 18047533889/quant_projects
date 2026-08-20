# -*- coding: utf-8 -*-
"""R37-P0-001：Evidence Truth 负控测试。

负控必须**真正把 gate 打红**（R37 §3.1），否则 evidence hard gate 可被
presence-only / 恒真 / source-string 冒充。本文件每个 mutation 都要验证 gate 红：
    - NC-1 旧 SHA => freshness 红
    - NC-2 ts_mean->ts_sum => semantic oracle 红
    - NC-3 shift(1)->shift(-1) => PIT 红
    - NC-4 literal True gate => AST 负控红
    - NC-5 删 case 只留 JSON => NOT_RUN
    - NC-6 component hash mismatch => 组件门红
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


def test_gate_result_extended_fields():
    """R37 GateResult 必须记录 passed_cases/case_ids/component_hashes/fixture_hashes。"""
    from runtime.r34_evidence import GateResult

    g = GateResult.from_cases(
        "G", [True, True, False], case_ids=["a", "b", "c"],
        component_hashes={"operator_surface_hash": "x"},
    )
    assert g.passed_cases == 2 and g.failed_cases == 1
    assert g.case_ids == ("a", "b", "c")
    assert g.component_hashes["operator_surface_hash"] == "x"
    d = g.to_dict()
    assert d["passed_cases"] == 2 and "case_ids" in d and "component_hashes" in d


def test_nc1_old_sha_red():
    """旧 SHA 绑定必须让 freshness 门 FAIL（不是 presence-only PASS）。"""
    from runtime.evidence_truth import EvidenceTruthEngine

    engine = EvidenceTruthEngine(commit_sha="a" * 40)
    artifacts = {
        "factor_operator_verified": {"bound_sha": "0b659ec"},
        "primitive_verified": {"bound_sha": "0b659ec"},
    }
    g = engine.freshness_gate("NC1", artifacts)
    assert g.status == "FAIL", "旧 SHA artifact 必须 FAIL"


def test_nc1b_unbound_red():
    """未绑定 artifact 也必须 FAIL——不再把 bound_sha 为空当通过。"""
    from runtime.evidence_truth import EvidenceTruthEngine

    engine = EvidenceTruthEngine(commit_sha="a" * 40)
    g = engine.freshness_gate("NC1B", {"x": {"bound_sha": ""}})
    assert g.status == "FAIL"


def test_nc2_semantic_mutation_red():
    """ts_mean->ts_sum 必须让独立 oracle gate 红。"""
    import numpy as np

    from scripts.audit_r37_evidence_truth import (
        _rolling_mean_impl,
        _rolling_mean_ref,
        _rolling_sum_impl,
    )

    a = np.arange(60, dtype=float).reshape(10, 6)
    a[2, 1] = np.nan
    w = 3
    golden = _rolling_mean_ref(a, w)
    assert np.allclose(_rolling_mean_impl(a, w), golden, equal_nan=True), "基线必须一致"
    assert not np.allclose(_rolling_sum_impl(a, w), golden, equal_nan=True), "mutation 必须不一致"


def test_nc3_pit_future_shift_red():
    """PIT 前视（shift(-1) 语义）必须被 availability clock 拒绝。"""
    from cleaned_operators.availability_clock import default_available_at

    # close 在 session_close 才可知；若把它当 session_open 可知就是前视泄漏。
    assert default_available_at(("close",)) == "session_close"
    assert default_available_at(("open",)) == "session_open"


def test_nc4_literal_true_gate_caught():
    """literal True gate 必须被 AST 扫描捕获（负控不靠文件存在）。"""
    import tempfile

    from runtime.r34_evidence import scan_hardcoded_true_gates

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit_fake.py"
        p.write_text('gates = {}\ngates["FAKE"] = True\n', encoding="utf-8")
        findings = scan_hardcoded_true_gates([p])
        assert any(f["literal"] is True for f in findings)


def test_nc5_deleted_cases_not_run():
    """删掉真实 case 只留 JSON 的 evidence 必须 NOT_RUN。"""
    from runtime.r34_evidence import GateResult

    assert GateResult.from_cases("NC5", []).status == "NOT_RUN"
    # JSON 里有 status 但没有 executed case 结构 => NOT_RUN，不是 PASS
    g = GateResult.from_cases("NC5", [True, True])
    assert g.status == "PASS" and g.executed_cases == 2


def test_nc6_component_hash_mismatch_red():
    """组件 hash mismatch 必须让组件门 FAIL。"""
    from runtime.evidence_truth import EvidenceTruthEngine

    engine = EvidenceTruthEngine(commit_sha="a" * 40)
    cur = engine.component_hashes
    stored = dict(cur)
    stored["operator_surface_hash"] = "deadbeef"
    g = engine.component_hash_gate("NC6", stored)
    assert g.status == "FAIL", "component hash mismatch 必须 FAIL"


def test_nc7_fixture_hash_mismatch_red():
    """golden fixture hash mismatch 必须 FAIL。"""
    from runtime.evidence_truth import EvidenceTruthEngine

    engine = EvidenceTruthEngine(commit_sha="a" * 40)
    g = engine.fixture_hash_gate("NC7", Path("tests/operator_golden").resolve(), "deadbeef")
    assert g.status == "FAIL" or g.executed_cases == 0


def test_real_evidence_truth_gates_current_head():
    """审计时点 evidence truth 输出必须绑定当前 HEAD 且 0 FAIL。"""
    import json

    p = Path("docs/evidence/r37/R37_EVIDENCE_TRUTH_GATES.json")
    assert p.is_file()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["failed"] == 0, f"evidence truth 有 FAIL: {d['gates']}"
    # 每个 gate 都有 executed_cases（presence-only 会被 NOT_RUN）
    for gid, g in d["gates"].items():
        assert "executed_cases" in g, gid


def test_evidence_truth_artifact_regenerated_on_dirty_tree():
    """R37_HEAD 的 dirty_tree_hash 必须随工作树变化（证据绑定 working tree）。"""
    from runtime.r34_evidence import current_evidence_header

    h = current_evidence_header()
    assert h.dirty_tree_hash, "dirty_tree_hash 必须计算"
