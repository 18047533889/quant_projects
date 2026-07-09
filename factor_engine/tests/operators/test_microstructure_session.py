# -*- coding: utf-8
"""微观结构 session-aware rolling 与 production DSL 门禁。"""

from __future__ import annotations

import pandas as pd
import pytest

from api.mining_integration import validate_production_dsl
from backend.cleaned_bridge import build_production_dsl_allowlist, ensure_cleaned_loaded
from cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_pct_change_resets_at_session_boundary():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-03 09:31",
            "2024-01-03 09:32",
        ]
    )
    s = pd.Series([100.0, 101.0, 200.0, 202.0], index=idx)
    out = pct_change_by_session(s)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.01)
    assert pd.isna(out.iloc[2])


def test_rolling_sum_does_not_cross_sessions():
    idx = pd.DatetimeIndex(
        ["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-03 09:31", "2024-01-03 09:32"]
    )
    s = pd.Series([1.0, 2.0, 10.0, 20.0], index=idx)
    out = rolling_by_session(s, 2, "sum", min_periods=1)
    assert out.iloc[2] == pytest.approx(10.0)


def test_micro_realized_vol_session_aware(_loaded):
    op = OperatorRegistry.get("micro_realized_vol")
    idx = pd.DatetimeIndex(
        ["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-03 09:31", "2024-01-03 09:32"]
    )
    close = pd.DataFrame({"A": [100.0, 101.0, 200.0, 204.0]}, index=idx)
    out = op.calculate(close, window=2, min_periods=2)
    assert out.iloc[2, 0] != out.iloc[1, 0] or pd.isna(out.iloc[2, 0])


def test_production_dsl_allowlist_subset(_loaded):
    full = len(__import__("backend.cleaned_bridge", fromlist=["build_cleaned_dsl_allowlist"]).build_cleaned_dsl_allowlist())
    prod = build_production_dsl_allowlist()
    assert 0 < len(prod) <= full


def test_validate_production_dsl_accepts_tier1(_loaded):
    ok, msg = validate_production_dsl("rank(ts_mean(col('close'), 20))")
    assert ok, msg


def test_microstructure_param_names_nonempty(_loaded):
    from cleaned_operators.operator_spec import check_microstructure_param_contracts

    assert not check_microstructure_param_contracts()
