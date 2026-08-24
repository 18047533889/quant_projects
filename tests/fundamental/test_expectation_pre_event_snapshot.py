# -*- coding: utf-8 -*-
"""R23-098..104 / R23-299: surprise must be a PRE-EVENT expectation snapshot.

``actual - expected`` is only a valid surprise when expected is frozen strictly
before the actual's knowledge time; a same-day consensus revision after the
print would leak.  The surprise family must DECLARE
``requires:PreEventExpectationSnapshot`` and the expectation-revision family
``requires:ConsensusVintageSource`` on their contracts.
"""
from __future__ import annotations

from factor_engine.cleaned_operators.registry import OperatorRegistry

_PRE_EVENT_CANONICALS = (
    "fin_surprise",
    "fin_surprise_zscore",
    "fin_surprise_event_zscore",
    "fin_surprise_event_percentile",
    "fin_actual_expectation_divergence",
    "fin_beat_streak",
    "fin_miss_streak",
)
_CONSENSUS_CANONICALS = (
    "fin_expectation_revision",
    "fin_expectation_revision_pct",
    "fin_expectation_revision_speed",
    "fin_expectation_revision_count",
    "fin_expectation_revision_magnitude",
    "fin_days_since_expectation_revision",
)


def _tags(canonical):
    ops = OperatorRegistry._operators.get(canonical, {})
    pd_op = ops.get("pandas_numpy")
    if pd_op is not None:
        return list(getattr(getattr(pd_op, "metadata", None), "tags", None) or [])
    return []


def test_surprise_family_declares_pre_event_snapshot_requirement():
    for name in _PRE_EVENT_CANONICALS:
        assert name in OperatorRegistry._catalog, name
        tags = _tags(name)
        assert "requires:PreEventExpectationSnapshot" in tags, name


def test_expectation_revision_family_declares_consensus_vintage_requirement():
    for name in _CONSENSUS_CANONICALS:
        assert name in OperatorRegistry._catalog, name
        tags = _tags(name)
        assert "requires:ConsensusVintageSource" in tags, name


def test_surprise_event_zscore_is_prior_only():
    # R23-107/108: the current event must NOT enter its own reference
    # distribution — the event z-score uses only the prior history.
    import numpy as np
    import pandas as pd

    from factor_engine.cleaned_operators.fundamental.expectation_v2 import fin_surprise_event_zscore

    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # surprises = [0,1,-1,2,-2,3,-3,4,-4,40]; prior std is non-zero.
    actual = pd.DataFrame({"A": [10.0, 11.0, 9.0, 12.0, 8.0, 13.0, 7.0, 14.0, 6.0, 50.0]}, index=idx)
    expected = pd.DataFrame({"A": [10.0] * n}, index=idx)
    scale = pd.DataFrame({"A": [1.0] * n}, index=idx)
    # Ten DISTINCT fiscal periods so the prior-only window is populated.
    pid = pd.DataFrame(
        {"A": [f"2023Q{i + 1}" for i in range(4)] + [f"2024Q{i + 1}" for i in range(4)] + ["2025Q1", "2025Q2"]},
        index=idx,
    )
    out = fin_surprise_event_zscore(actual, expected, scale, pid, periods=8)
    # The final spike (surprise 40) is scored against the PRIOR history only; a
    # prior-only z-score stands out (large positive), never self-compressed by
    # including the spike in its own reference distribution.
    assert np.isfinite(out["A"].iloc[-1]), "prior-only event z-score must be finite"
    assert out["A"].iloc[-1] > 1.0, "spike must stand out against the prior distribution"
