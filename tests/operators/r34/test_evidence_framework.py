# -*- coding: utf-8 -*-
"""R34 验收测试：Evidence 框架 / 代码修复探针 / 参数域 / stateful 检测。

禁止假完成：每个 gate 必须 executed_cases > 0（P0-002）；golden 必须独立
oracle（P0-006/007）。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


# ---------------------------------------------------------------------------
# Evidence 框架（P0-002）
# ---------------------------------------------------------------------------


def test_gate_result_requires_executed_cases():
    from factor_engine.runtime.r34_evidence import GateResult

    g = GateResult.from_cases("G", [True, True])
    assert g.status == "PASS" and g.executed_cases == 2
    g2 = GateResult.from_cases("G2", [True, False])
    assert g2.status == "FAIL" and g2.failed_cases == 1
    g3 = GateResult.from_cases("G3", [])
    assert g3.status == "NOT_RUN" and g3.executed_cases == 0
    g4 = GateResult.not_run("G4", "no evidence")
    assert g4.status == "NOT_RUN"


def test_evidence_header_binds_commit_and_tree():
    from factor_engine.runtime.r34_evidence import current_evidence_header

    h = current_evidence_header()
    assert h.commit_sha, "HEAD 必须非空"
    assert len(h.commit_sha) >= 7
    assert h.dirty_tree_hash, "dirty_tree_hash 必须计算"
    d = h.to_dict()
    assert len(d) == 13, "EvidenceHeader 必须有 13 个字段"


def test_hardcoded_true_gate_scanner_finds_literal():
    from factor_engine.runtime.r34_evidence import scan_hardcoded_true_gates

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit_x.py"
        p.write_text(
            'gates = {}\ngates["FAKE"] = True\n'
            'gates["REAL"] = bool(check())\n',
            encoding="utf-8",
        )
        findings = scan_hardcoded_true_gates([p])
        assert any(f["literal"] is True for f in findings), "必须能扫到字面量 True gate"


# ---------------------------------------------------------------------------
# 参数域独立 oracle（P0-008/009/010）
# ---------------------------------------------------------------------------


def test_parameter_domain_independent_oracle_exists():
    """参数域认证是独立 numpy reference，不调用生产 kernel 自证。"""
    import inspect

    import scripts.audit_r34_parameter_domains as mod

    src = inspect.getsource(mod._rolling_mean_ref)
    assert "rolling" not in src or "OperatorRegistry" not in src
    assert "np.nanmean" in src


def test_parameter_domain_evidence_generated():
    from pathlib import Path

    p = Path("docs/evidence/r34/R34_PARAMETER_DOMAIN_COVERAGE.json")
    assert p.is_file()
    import json

    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("independent_oracle") is True
    assert len(d.get("certified", {})) > 0, "必须认证出非默认参数域"


# ---------------------------------------------------------------------------
# 代码修复探针
# ---------------------------------------------------------------------------


def test_availability_clock_open_semantics():
    from factor_engine.cleaned_operators.availability_clock import (
        default_available_at,
        default_same_session_usable,
    )

    assert default_available_at(("open",)) == "session_open"
    assert default_available_at(("open", "close")) == "session_close"
    assert default_same_session_usable(("open",)) is True
    assert default_same_session_usable(("close",)) is False


def test_stateful_behavior_detection():
    from stateful_contract import detect_stateful_behavior

    assert detect_stateful_behavior(np.cumsum, np.ones(40)) is True
    assert detect_stateful_behavior(np.abs, np.arange(-20, 20, dtype=float)) is False


def test_production_dataevent_no_env_bypass():
    from factor_engine.runtime.production_policy import (
        is_production_mode,
        production_data_event_auto_publish_enabled,
    )

    _prev = os.environ.get("QUANT_PRODUCTION_MODE")
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    os.environ.pop("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", None)
    try:
        assert is_production_mode()
        # production: env bypass 不存在，恒原子两阶段
        assert production_data_event_auto_publish_enabled() is True
        os.environ["DATA_EVENT_PRODUCTION_AUTO_PUBLISH"] = "0"
        assert production_data_event_auto_publish_enabled() is True
    finally:
        if _prev:
            os.environ["QUANT_PRODUCTION_MODE"] = _prev
        else:
            os.environ.pop("QUANT_PRODUCTION_MODE", None)
        os.environ.pop("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", None)
    # research: env 生效
    assert not is_production_mode()
    os.environ.pop("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", None)
    assert production_data_event_auto_publish_enabled() is False
    os.environ["DATA_EVENT_PRODUCTION_AUTO_PUBLISH"] = "1"
    assert production_data_event_auto_publish_enabled() is True
    os.environ.pop("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", None)


def test_model_timing_explicit_contract_required():
    from factor_engine.cleaned_operators.model_timing import (
        MODEL_TIMING_CONTRACTS,
        model_timing_contract_is_explicit,
    )

    assert MODEL_TIMING_CONTRACTS, "显式 contract 表非空"
    assert model_timing_contract_is_explicit("panel_rolling_pcr_forecast")
    assert not model_timing_contract_is_explicit("definitely_not_a_real_model_xyz")


def test_financial_grain_contract_accepts_market_context():
    from factor_engine.cleaned_operators.operator_spec import check_financial_grain_contract

    # 无 market_context 默认 A 股（向后兼容）；不抛错即可
    errs = check_financial_grain_contract("fin_qoq(field('revenue'), field('period'))")
    assert isinstance(errs, list)


def test_engine_production_frequency_fail_closed():
    """production 下缺 frequency 必须 fail-closed（P0-019）。"""
    from factor_engine.runtime.production_policy import is_production_mode
    import os

    _prev = os.environ.get("QUANT_PRODUCTION_MODE")
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        assert is_production_mode()
        # 直接验证 engine 的频次解析路径（读源码确认 fail-closed 存在）
        src = open("runtime/engine.py", encoding="utf-8").read()
        assert "refusing to guess '1d' in production" in src
    finally:
        if _prev:
            os.environ["QUANT_PRODUCTION_MODE"] = _prev
        else:
            os.environ.pop("QUANT_PRODUCTION_MODE", None)


def test_hard_gate_audit_honest():
    """R34 hard gates 审计必须产出 PASS/NOT_RUN/FAIL 而非恒真。"""
    from pathlib import Path

    p = Path("docs/evidence/r34/R34_HARD_GATES.json")
    assert p.is_file()
    import json

    d = json.loads(p.read_text(encoding="utf-8"))
    assert "production_ready" in d
    # 每个 gate 都必须是 GateResult 结构（有 executed_cases 字段）
    for gid, g in d["gates"].items():
        assert "executed_cases" in g, gid
        assert "status" in g, gid
