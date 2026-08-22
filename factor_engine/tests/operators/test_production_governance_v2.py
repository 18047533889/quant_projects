# -*- coding: utf-8 -*-
"""2026-08 production governance rework (review §2, §4, §5, §14).

Covers:
- Six-gate certification is the single ``production_certified`` authority and
  the only thing that may set ``status == "production"`` / ``pit_safe=True``.
- ``allow_in_production`` requires daily/extended classification + six-gate +
  at least one evidence-backed backend + no compat/diagnostic/benchmark flag.
- Dataset-scoped field resolution with anchor preference for the OHLCV family
  and hard ambiguity errors for non-anchor shared names (review §4.3).
- Financial grain contract: single-period change on ``flow_ytd`` fields is
  rejected; YoY (``periods=4``) on cumulative fields is legal (review §5.1).
- New 2026-08 operator families register, classify daily, and run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded

ensure_cleaned_loaded()

from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _date_index(n: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


# ---------------------------------------------------------------------------
# Six-gate governance invariants
# ---------------------------------------------------------------------------

def test_six_gate_is_the_single_production_authority() -> None:
    from cleaned_operators.operator_surface import classify_canonical
    from cleaned_operators.production_hardening import factor_production_targets

    # The invariant binds only the reviewed production targets: unsafe/internal/
    # non-factor canonicals keep a policy ``pit_safe`` flag but are never
    # certified, so status/pit_safe there are not certification results.
    targets = factor_production_targets()
    for canonical in targets:
        catalog = OperatorRegistry._catalog.get(canonical, {})
        certified = catalog.get("production_certified") is True
        op_cert = catalog.get("operator_certification") is True
        status = str(catalog.get("status"))
        pit_safe = catalog.get("pit_safe") is True
        # production_certified and operator_certification must agree.
        assert certified == op_cert, canonical
        # status/pit_safe may only be production/True when certified.
        assert (status == "production") == certified, canonical
        assert pit_safe == certified, canonical
        if certified:
            assert catalog.get("implementation_certified") is True, canonical
            assert catalog.get("backend_passed") is True, canonical
            assert classify_canonical(canonical) in {"daily", "extended"}, canonical


def test_allow_in_production_requires_full_six_gate() -> None:
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.operator_surface import classify_canonical

    for spec in __import__("cleaned_operators.operator_spec", fromlist=["iter_operator_specs"]).iter_operator_specs():
        if not spec.allow_in_production:
            continue
        catalog = OperatorRegistry._catalog.get(spec.canonical, {})
        assert classify_canonical(spec.canonical) in {"daily", "extended"}, spec.canonical
        assert catalog.get("production_certified") is True, spec.canonical
        assert catalog.get("backend_passed") is True, spec.canonical
        for flag in ("compatibility_only", "diagnostic_only", "benchmark_only"):
            assert not catalog.get(flag), f"{spec.canonical} flagged {flag}"


def test_compatibility_only_never_production() -> None:
    from cleaned_operators.operator_spec import build_operator_spec

    for name in ("ts_multi_regression_coeff", "intra_positive_jump_variation"):
        spec = build_operator_spec(name)
        assert spec is not None
        assert spec.allow_in_production is False, name


# ---------------------------------------------------------------------------
# Dataset-scoped field resolution (review §4.3)
# ---------------------------------------------------------------------------

def test_qualified_and_anchor_field_resolution() -> None:
    from api.columns import field
    from fields import FIELD_REGISTRY

    assert field("close").table == "StockDailyBar"
    assert field("volume").table == "StockDailyBar"
    # Table-qualified resolution works for shared physical names.
    assert FIELD_REGISTRY.require("StockDailyBar.close").source_name == "Close"
    assert FIELD_REGISTRY.require("IndexDailyBar.index_close").source_name == "Close"
    assert FIELD_REGISTRY.require("EtfDailyBar.etf_close").source_name == "Close"
    assert FIELD_REGISTRY.require("StockMinuteBar.minute_close").source_name == "Close"
    assert FIELD_REGISTRY.require("StockBalance.pub_date").table == "StockBalance"
    # Bare non-anchor shared name is ambiguous → strict error.
    with pytest.raises(KeyError, match="pub_date"):
        field("pub_date", strict=True)
    # Distinct canonical names are unambiguous.
    assert FIELD_REGISTRY.require("index_close").table == "IndexDailyBar"
    assert FIELD_REGISTRY.require("minute_close").table == "StockMinuteBar"


def test_minute_source_contract_fields_registered() -> None:
    from fields import FIELD_REGISTRY

    for name in ("quote_time", "minute_open", "minute_high", "minute_low",
                 "minute_close", "minute_volume", "minute_amount", "minute_vwap"):
        spec = FIELD_REGISTRY.require(name)
        assert spec.table == "StockMinuteBar", name


# ---------------------------------------------------------------------------
# Financial grain contract (review §5.1 / §5.2)
# ---------------------------------------------------------------------------

def test_financial_grain_contract_rejects_qoq_on_cumulative() -> None:
    from cleaned_operators.operator_spec import check_financial_grain_contract

    assert check_financial_grain_contract("fin_qoq(net_profit, period_id)")
    assert check_financial_grain_contract("fin_pct_change(net_profit, period_id)")
    assert check_financial_grain_contract("fin_log_change(net_profit, period_id, 1)")
    # YoY / multi-period growth on a cumulative flow is legal (same YTD position).
    assert not check_financial_grain_contract("fin_pct_change(net_profit, period_id, 4)")
    assert not check_financial_grain_contract(
        "fin_pct_change(net_profit, period_id, periods=4)"
    )
    # Non-financial fields are not affected.
    assert not check_financial_grain_contract("fin_qoq(close, period_id)")


# ---------------------------------------------------------------------------
# 2026-08 operator families smoke (review §13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "canonical,args,kwargs",
    [
        ("ts_distance_corr", (None, None), {"window": 20}),
        ("ts_permutation_entropy", (None,), {"window": 60, "order": 3}),
        ("ts_expected_shortfall", (None,), {"window": 60, "q": 0.05}),
        ("ts_lower_partial_moment", (None,), {"window": 60, "threshold": 0.0}),
        ("ashare_limit_up_streak", (None, None, None, None), {}),
        ("relation_hhi_change", (None,), {"window": 20}),
        ("group_skewness", (None, None), {}),
        ("intra_slot_volume_surprise", (None,), {"window": 20}),
    ],
)
def test_new_operator_families_run(canonical, args, kwargs) -> None:
    idx = _date_index(6)
    rng = np.random.default_rng(7)
    cols = ["A", "B"]
    x = pd.DataFrame(rng.normal(size=(6, 2)), index=idx, columns=cols)

    def _panel_like(prefix: str) -> pd.DataFrame:
        return pd.DataFrame(rng.normal(size=(6, 2)), index=idx, columns=cols)

    resolved = []
    for arg in args:
        if arg is None:
            resolved.append(_panel_like("y"))
        else:
            resolved.append(arg)
    if canonical == "group_skewness":
        resolved[1] = pd.DataFrame({"A": ["g1"] * 6, "B": ["g2"] * 6}, index=idx)
    if canonical == "ashare_limit_up_streak":
        # close, high_limit, valid_trade, tick_tolerance (scalar)
        close = x.abs() + 10.0
        high_limit = close + 0.1
        valid = pd.DataFrame(np.ones((6, 2)), index=idx, columns=cols)
        resolved = [close, high_limit, valid, 0.005]
    out = OperatorRegistry.get(canonical).calculate(*resolved, **kwargs)
    # Window operators may trim warmup rows; column count must be preserved.
    assert list(out.columns) == list(x.columns)
    assert out.shape[1] == x.shape[1]
    assert out.shape[0] <= x.shape[0]
