from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.tail_systemic  # noqa: F401
from factor_engine.backend.operator_semantic_version import semantic_version
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import history_requirement


@pytest.mark.parametrize("canonical", ("group_tail_centrality", "group_tail_lead_score"))
@pytest.mark.parametrize("prior_threshold", (True, False))
@pytest.mark.parametrize("lag", (1, 3))
def test_nested_tail_history_declares_both_windows(
    canonical: str, prior_threshold: bool, lag: int
) -> None:
    params = {"window": 20, "prior_threshold": prior_threshold}
    if canonical == "group_tail_lead_score":
        params["lag"] = lag
    requirement = history_requirement(canonical, params)
    assert requirement.kind == "finite"
    assert requirement.rows == 38


@pytest.mark.parametrize("canonical", ("group_tail_centrality", "group_tail_lead_score"))
def test_prior_threshold_impossible_support_is_rejected_statically(canonical: str) -> None:
    op = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    x = pd.DataFrame(np.arange(80.0).reshape(20, 4), columns=list("ABCD"))
    group = pd.DataFrame("G", index=x.index, columns=x.columns)
    kwargs = {"window": 10, "min_periods": 10, "prior_threshold": True}
    if canonical == "group_tail_lead_score":
        kwargs["min_conditioning_events"] = 1
    with pytest.raises(ValueError, match="prior_threshold=True requires min_periods <= window - 1"):
        op.calculate(x, group, **kwargs)

    # The same support is feasible when the current row belongs to the
    # threshold-estimation window; it must not be silently lowered or rejected.
    kwargs["prior_threshold"] = False
    result = op.calculate(x, group, **kwargs)
    assert result.shape == x.shape


@pytest.mark.parametrize("canonical", ("group_tail_centrality", "group_tail_lead_score"))
def test_tail_history_semantic_version_tracks_contract_change(canonical: str) -> None:
    assert semantic_version(canonical) == 2


@pytest.mark.parametrize("canonical", ("group_tail_centrality", "group_tail_lead_score"))
def test_38_row_warmup_restores_nested_tail_replay(canonical: str) -> None:
    rng = np.random.default_rng(20260909)
    values = rng.standard_t(4, size=(180, 6)).cumsum(axis=0)
    x = pd.DataFrame(values, columns=list("ABCDEF"))
    group = pd.DataFrame(
        np.tile(["G1", "G1", "G1", "G2", "G2", "G2"], (len(x), 1)),
        columns=x.columns,
    )
    op = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    kwargs = {
        "window": 20,
        "quantile": 0.25,
        "min_periods": 5,
        "prior_threshold": True,
    }
    if canonical == "group_tail_lead_score":
        kwargs.update(lag=3, min_conditioning_events=1)

    start = 100
    full = op.calculate(x, group, **kwargs).iloc[start:].to_numpy()
    insufficient = op.calculate(
        x.iloc[start - 19 :], group.iloc[start - 19 :], **kwargs
    ).iloc[19:].to_numpy()
    sufficient = op.calculate(
        x.iloc[start - 38 :], group.iloc[start - 38 :], **kwargs
    ).iloc[38:].to_numpy()

    shared = np.isfinite(full) & np.isfinite(insufficient)
    assert shared.any()
    assert np.max(np.abs(full[shared] - insufficient[shared])) > 0.05
    np.testing.assert_allclose(sufficient, full, equal_nan=True)
