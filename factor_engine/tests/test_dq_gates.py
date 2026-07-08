"""DQ 门禁与算子 policy 单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_policy import (
    compute_operator_catalog_hash,
    effective_lookback,
    infer_operator_policy,
)
from cleaned_operators.registry import OperatorRegistry
from runtime.dq_gates import (
    DQThresholds,
    FactorDQError,
    assert_factor_dq,
    evaluate_factor_dq,
)

pd = pytest.importorskip("pandas")


def _series(values, instruments=("A", "B")):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=len(values), freq="D"), instruments],
        names=["timestamp", "instrument"],
    )
    out = []
    for v in values:
        out.extend([v, v + 1.0])
    return pd.Series(out[: len(idx)], index=idx)


class TestDQGates:
    def test_good_factor_passes(self):
        s = _series([1.0, 2.0, 3.0, 4.0])
        report = evaluate_factor_dq(s)
        assert report.passed

    def test_all_nan_fails_coverage(self):
        s = _series([np.nan, np.nan, np.nan])
        report = evaluate_factor_dq(s, thresholds=DQThresholds(min_coverage=0.5))
        assert not report.passed
        assert any(c.name == "coverage" and not c.passed for c in report.checks)

    def test_inf_fails_by_default(self):
        s = _series([1.0, np.inf, 3.0])
        report = evaluate_factor_dq(s)
        assert not report.passed

    def test_assert_raises(self):
        s = _series([np.nan, np.nan])
        with pytest.raises(FactorDQError):
            assert_factor_dq(
                s, thresholds=DQThresholds(min_coverage=0.5), raise_on_fail=True
            )


class TestOperatorPolicy:
    def test_ts_delay_policy(self):
        op = OperatorRegistry.get("ts_delay")
        p = infer_operator_policy(op, canonical="ts_delay")
        assert p.scope == "ts"
        assert p.pit_safe

    def test_lead_policy(self):
        op = OperatorRegistry.get("Lead")
        p = infer_operator_policy(op, canonical="Lead")
        assert p.lag == -1
        assert not p.pit_safe  # 负 lag = 未来函数，PIT 审计应拦截

    def test_rank_is_cross_section(self):
        op = OperatorRegistry.get("rank")
        p = infer_operator_policy(op, canonical="rank")
        assert p.scope == "cs"

    def test_catalog_hash_stable(self):
        h1 = compute_operator_catalog_hash()
        h2 = compute_operator_catalog_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_effective_lookback(self):
        assert effective_lookback(20) >= 25
        assert effective_lookback(0) >= 5
