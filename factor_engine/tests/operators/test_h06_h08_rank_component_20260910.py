from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.common.cross_sectional import RankPctPolars
from factor_engine.cleaned_operators.fundamental.component_score import FinComponentScore


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0],
        [None, 10.0, 20.0],
        [10.0, None, 20.0],
        [10.0, 20.0, None],
        [None, None, None],
        [1.0, float("nan"), 1.0],
        [float("-inf"), 0.0, float("inf")],
        [7.0],
    ],
)
def test_h06_rank_pct_preserves_original_instrument_axis(values) -> None:
    columns = {f"s{i}": [value] for i, value in enumerate(values)}
    actual = RankPctPolars().calculate(pl.DataFrame(columns)).to_pandas()
    expected = pd.DataFrame([values], columns=list(columns)).rank(pct=True, axis=1)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_h06_rank_pct_is_permutation_equivariant() -> None:
    frame = pl.DataFrame({"a": [1.0], "b": [None], "c": [3.0], "d": [2.0]})
    original = RankPctPolars().calculate(frame).to_pandas()
    permuted = RankPctPolars().calculate(frame.select(["d", "b", "a", "c"])).to_pandas()
    pd.testing.assert_frame_equal(original, permuted[["a", "b", "c", "d"]])


def _panel(value: float) -> pd.DataFrame:
    return pd.DataFrame([[value, value]], index=pd.DatetimeIndex(["2025-01-02"]), columns=["A", "B"])


def test_h07_named_and_positional_components_are_equivalent() -> None:
    op = FinComponentScore()
    x, y = _panel(1.0), _panel(-1.0)
    positional = op.calculate(x, y, component_directions=["up", "down"])
    named = op.calculate(component_1=x, component_2=y, component_directions=["up", "down"])
    pd.testing.assert_frame_equal(positional, named)
    assert np.array_equal(
        positional.attrs["effective_component_count"],
        named.attrs["effective_component_count"],
    )


def test_h07_eight_panels_and_trailing_positional_scalars_bind() -> None:
    panels = [_panel(1.0) for _ in range(8)]
    result = FinComponentScore().calculate(
        *panels, "down", [1.0] * 8, "score_available"
    )
    assert (result == 0.0).all().all()
    assert (result.attrs["effective_component_count"] == 8).all()


def test_h07_gapped_named_components_preserve_declared_slots() -> None:
    result = FinComponentScore().calculate(component_1=_panel(1), component_3=_panel(-1))
    assert (result == 1.0).all().all()
    assert (result.attrs["effective_component_count"] == 2).all()


@pytest.mark.parametrize("weights", [[1.0, np.nan], [1.0, np.inf], [1.0, -np.inf]])
def test_h08_nonfinite_weights_are_invalid_parameters(weights) -> None:
    with pytest.raises(ValueError, match="finite"):
        FinComponentScore().calculate(_panel(1), _panel(1), score_weights=weights)


def test_h08_weight_length_and_scalar_rejected_but_finite_negative_allowed() -> None:
    op = FinComponentScore()
    with pytest.raises(ValueError, match="length"):
        op.calculate(_panel(1), _panel(1), score_weights=[1.0])
    with pytest.raises(ValueError, match="sequence"):
        op.calculate(_panel(1), score_weights=1.0)
    result = op.calculate(_panel(1), _panel(1), score_weights=[1.0, -2.0])
    assert (result == -1.0).all().all()


def test_h08_require_full_keeps_real_missing_data_unknown() -> None:
    missing = _panel(1.0)
    missing.iloc[0, 0] = np.nan
    result = FinComponentScore().calculate(
        _panel(1.0), missing, missing_policy="require_full"
    )
    assert np.isnan(result.iloc[0, 0])
    assert result.iloc[0, 1] == 2.0
