# -*- coding: utf-8
"""算子层 P0 硬化：PIT fail-closed、forward-fill 黑名单、Tier-1、Polars 白名单。"""

from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from backend.operator_cost import tier1_has_explicit_cost
from cleaned_operators.operator_policy import (
    POLARS_PRODUCTION_SAFE,
    TIER1_ALIASES,
    infer_operator_policy,
    resolve_tier1_canonical,
    tier1_policy_keys,
)
from cleaned_operators.registry import OperatorRegistry
from runtime.pit_audit import audit_ir


@pytest.fixture(scope="module")
def _load():
    ensure_cleaned_loaded()
    yield


def test_infer_operator_policy_fail_closed_without_tags():
    """未显式 policy 且无 pit_safe/causal tag → 默认不安全。"""
    class _Meta:
        name = "unknown_custom_op"
        category = "time_series"
        tags = ["time_series"]

    class _Op:
        metadata = _Meta()

    policy = infer_operator_policy(_Op(), canonical="unknown_custom_op")
    assert policy.pit_safe is False


def test_infer_operator_policy_accepts_pit_safe_tag():
    class _Meta:
        name = "tagged_op"
        category = "time_series"
        tags = ["pit_safe"]

    class _Op:
        metadata = _Meta()

    assert infer_operator_policy(_Op(), canonical="tagged_op").pit_safe is True


def test_tier1_alias_resolves_to_canonical():
    assert resolve_tier1_canonical("SMA") == "ts_mean"
    assert resolve_tier1_canonical("EMA") == "ts_ema"
    assert "SMA" in TIER1_ALIASES


def test_tier1_policy_keys_are_canonical():
    keys = tier1_policy_keys()
    assert "SMA" not in keys
    assert "ts_mean" in keys


def test_audit_flags_missing_runtime(fail_on_missing=True):
    from ir.nodes import IRNode

    ir = IRNode(op="definitely_not_registered_op_xyz", inputs=[], attrs={})
    report = audit_ir(ir, fail_on_missing=True)
    assert not report.passed
    assert any("missing_runtime" in v for v in report.violations)


def test_audit_forbid_forward_fill_includes_ffill():
    from expr.cleaned_call import CleanedCall
    from expr.column import ColumnRef
    from ir.analyzer import Analyzer

    call = CleanedCall("ffill", (ColumnRef("close"),))
    ir = Analyzer().lower(call).ir
    report = audit_ir(ir, forbid_forward_fill=True)
    assert not report.passed
    assert any("forward_fill" in v for v in report.violations)


def test_polars_auto_prefers_whitelist_only():
    op, backend = OperatorRegistry.get_preferred("ts_mean", prefer="auto")
    assert backend == "polars"
    assert op is not None

    _op2, backend2 = OperatorRegistry.get_preferred("rank", prefer="auto")
    assert backend2 == "polars"

    _op3, backend3 = OperatorRegistry.get_preferred("ts_corr", prefer="auto")
    assert backend3 == "polars"

    _op4, backend4 = OperatorRegistry.get_preferred("ts_rank", prefer="auto")
    assert backend4 == "polars"

    _op5, backend5 = OperatorRegistry.get_preferred("MACD", prefer="auto")
    assert backend5 == "pandas_numpy"


def test_tier1_core_has_explicit_cost():
    for name in ("ts_mean", "ts_rank", "micro_realized_vol", "real_turnover_rate"):
        canon = resolve_tier1_canonical(name)
        assert tier1_has_explicit_cost(canon), canon


def test_operator_contracts_script_passes():
    from scripts.check_operator_contracts import check_operator_contracts

    ensure_cleaned_loaded()
    errors = check_operator_contracts(strict_tier1_cost=True)
    assert not errors, errors[:10]
