from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators.filter_hysteresis as filter_hysteresis
import factor_engine.cleaned_operators.filter_smooth as filter_smooth
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.operator_semantic_version import semantic_version
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import execution_contract, history_requirement
from factor_engine.stateful_contract import StatefulCheckpointRegistry, StatefulContractError


_RECURSIVE_WITHOUT_CHECKPOINT = (
    "ts_super_smoother",
    "ts_kama",
    "state_l1_turnover_prox",
    "state_l2_partial_adjustment",
)


def test_recursive_filters_require_full_history_without_fake_checkpoint_specs() -> None:
    for canonical in _RECURSIVE_WITHOUT_CHECKPOINT:
        contract = execution_contract(canonical)
        assert contract.state_model == "recursive", canonical
        assert contract.chunking == "required_full_history", canonical
        assert contract.checkpoint_schema is None, canonical
        assert history_requirement(canonical).kind == "full_history", canonical
        assert StatefulCheckpointRegistry.get(canonical) is None, canonical


def test_checkpoint_boundary_rejects_declared_full_history_filters() -> None:
    for canonical in _RECURSIVE_WITHOUT_CHECKPOINT:
        StatefulCheckpointRegistry.require_for_segment(
            canonical, starts_at_dataset_origin=True, checkpoint=None
        )
        with pytest.raises(StatefulContractError, match="unsupported without a native checkpoint"):
            StatefulCheckpointRegistry.require_for_segment(
                canonical, starts_at_dataset_origin=False, checkpoint=None
            )


def test_filter_metadata_does_not_claim_unimplemented_checkpoint_restore() -> None:
    assert filter_smooth.TSSuperSmoother.filter_contract.checkpointable is False
    assert filter_smooth.TSKama.filter_contract.checkpointable is False
    assert filter_smooth.TSKama.checkpointable is False
    assert filter_hysteresis.StateL1TurnoverProx.filter_contract.checkpointable is False
    assert filter_hysteresis.StateL2PartialAdjustment.filter_contract.checkpointable is False


def _ehlers_reference(values: np.ndarray, period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return out
    out[finite[0]] = values[finite[0]]
    if finite.size == 1:
        return out
    out[finite[1]] = values[finite[1]]

    angle = np.sqrt(2.0) * np.pi / period
    a = np.exp(-angle)
    c2 = 2.0 * a * np.cos(angle)
    c3 = -(a * a)
    c1 = 1.0 - c2 - c3
    y2 = values[finite[0]]
    y1 = values[finite[1]]
    x1 = values[finite[1]]
    for row in range(finite[1] + 1, values.size):
        current = values[row]
        if not np.isfinite(current):
            continue
        y = c1 * (current + x1) / 2.0 + c2 * y1 + c3 * y2
        out[row] = y
        y2, y1, x1 = y1, y, current
    return out


@pytest.mark.parametrize(
    "values",
    (
        np.r_[1.0, np.zeros(63)],
        np.r_[np.zeros(16), np.ones(48)],
        np.sin(np.arange(64, dtype=float) * 0.23),
        np.full(64, 7.5),
        np.r_[0.0, 1.0, np.nan, 2.0, np.linspace(2.0, 4.0, 59)],
    ),
)
def test_super_smoother_matches_independent_ehlers_recurrence(values: np.ndarray) -> None:
    frame = pd.DataFrame({"A": values})
    actual = OperatorRegistry.get("ts_super_smoother", "pandas_numpy", mode="any").calculate(
        frame, period=10
    )["A"].to_numpy()
    np.testing.assert_allclose(actual, _ehlers_reference(values, 10), equal_nan=True)


def test_super_smoother_semantic_version_bumped_for_new_transfer_function() -> None:
    assert semantic_version("ts_super_smoother") == 2


def test_legacy_polars_candidate_is_an_explicit_exact_delegate() -> None:
    from factor_engine.cleaned_operators.polars_native.ts_advanced_batch4 import (
        TSSuperSmootherPolarsNative,
    )

    candidate = TSSuperSmootherPolarsNative()
    assert candidate.metadata.param_names == ["x", "period"]
    assert candidate._physical_spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
    values = np.r_[np.zeros(8), np.ones(24)]
    actual = candidate.calculate(pl.DataFrame({"A": values}), period=10)["A"].to_numpy()
    expected = _ehlers_reference(values, 10)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
