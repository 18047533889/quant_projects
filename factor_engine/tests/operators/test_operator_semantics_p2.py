# -*- coding: utf-8
"""P2 production 核心算子 golden tests：winsorize、group_rank、production allowlist。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy P2 rollout contract superseded by strict fiscal primitives")

from api.mining_integration import validate_production_dsl
from backend.cleaned_bridge import build_production_dsl_allowlist, ensure_cleaned_loaded
from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS, build_operator_spec
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_winsorize_clips_row_quantiles(_loaded):
    op = OperatorRegistry.get("winsorize")
    x = pd.DataFrame({"A": [1.0, 100.0], "B": [2.0, 200.0], "C": [3.0, 300.0]})
    out = op.calculate(x, lower=0.25, upper=0.75)
    row = x.iloc[0]
    lo = row.quantile(0.25)
    hi = row.quantile(0.75)
    expected = row.clip(lower=lo, upper=hi)
    pd.testing.assert_series_equal(out.iloc[0], expected, check_names=False)


def test_group_rank_within_groups_only(_loaded):
    op = OperatorRegistry.get("group_rank")
    x = pd.DataFrame(
        {"A": [10.0, 1.0], "B": [20.0, 2.0], "C": [30.0, 3.0], "D": [40.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    group = pd.DataFrame(
        {"A": ["G1", "G1"], "B": ["G1", "G1"], "C": ["G2", "G2"], "D": ["G2", "G2"]},
        index=x.index,
    )
    out = op.calculate(x, group)
    assert out.loc["2024-01-01", "A"] == pytest.approx(0.5)
    assert out.loc["2024-01-01", "B"] == pytest.approx(1.0)
    assert out.loc["2024-01-01", "C"] == pytest.approx(0.5)
    assert out.loc["2024-01-01", "D"] == pytest.approx(1.0)


def test_cs_demean_row_mean_zero(_loaded):
    op = OperatorRegistry.get("cs_demean")
    x = pd.DataFrame({"A": [1.0, 10.0], "B": [3.0, 30.0], "C": [5.0, 50.0]})
    out = op.calculate(x)
    assert out.mean(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out.mean(axis=1).iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_production_core_canonicals_have_runtime_and_policy(_loaded):
    errors: list[str] = []
    for canon in sorted(PRODUCTION_CORE_CANONICALS):
        if OperatorRegistry.get(canon) is None:
            errors.append(f"{canon} 无 runtime")
            continue
        spec = build_operator_spec(canon)
        if spec is None or not spec.pit_safe:
            errors.append(f"{canon} pit_safe=False")
    assert not errors, errors[:8]


def test_production_dsl_includes_new_core_ops(_loaded):
    prod = build_production_dsl_allowlist()
    for name in ("rolling_beta", "RSI_WILDER", "ATR_WILDER", "ts_rolling_beta", "ts_rsi_wilder"):
        assert name in prod, f"{name} 不在 production allowlist"


def test_validate_production_dsl_rolling_beta(_loaded):
    ok, msg = validate_production_dsl("rolling_beta(col('ret'), col('mkt'), 60)")
    assert ok, msg


def test_validate_production_dsl_rejects_fp_beta(_loaded):
    ok, msg = validate_production_dsl("fp_beta(col('ret'), col('mkt'), 60)")
    assert not ok
    assert "fp_beta" in msg or "rolling_beta_to_market" in msg or "不允许" in msg


def test_avg2_skips_non_consecutive_fiscal_period(_loaded):
    op = OperatorRegistry.get("avg2")
    idx = pd.date_range("2024-01-01", periods=3, freq="QS")
    x = pd.DataFrame({"A": [100.0, 200.0, 350.0]}, index=idx)
    fq = pd.DataFrame({"A": [1, 3, 4]}, index=idx)
    out = op.calculate(x, fq)
    assert out.iloc[1, 0] == pytest.approx(200.0)
    assert out.iloc[2, 0] == pytest.approx((350.0 + 200.0) / 2.0)
