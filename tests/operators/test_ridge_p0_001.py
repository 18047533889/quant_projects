"""Focused regressions for R2-P0-016 rolling Ridge ABI and failure behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.operator_errors import OperatorParameterError
from cleaned_operators.common.statistics import Ridge


def _oracle_slope(y: np.ndarray, x: np.ndarray, alpha: float) -> float:
    """Centered paired Ridge slope: intercept is unpenalized."""
    xc = x - x.mean()
    yc = y - y.mean()
    return float(np.dot(xc, yc) / (np.dot(xc, xc) + alpha))


def test_dual_input_all_nan_column_does_not_mask_finite_column() -> None:
    x = pd.DataFrame({"finite": np.arange(1.0, 9.0), "empty": np.nan})
    y = pd.DataFrame({"finite": 2.0 * x["finite"], "empty": np.nan})

    out = Ridge().calculate(y, x, window=5, alpha=0.0)

    assert out["empty"].isna().all()
    assert out.loc[7, "finite"] == pytest.approx(2.0)


def test_positional_alpha_matches_independent_numeric_oracle() -> None:
    x = pd.DataFrame({"A": np.array([1.0, 2.0, 4.0, 7.0, 11.0])})
    y = pd.DataFrame({"A": np.array([3.0, 5.0, 10.0, 16.0, 24.0])})
    alpha = 0.5

    out = Ridge().calculate(y, x, 5, alpha)

    assert out.loc[4, "A"] == pytest.approx(
        _oracle_slope(y["A"].to_numpy(), x["A"].to_numpy(), alpha)
    )


def test_malformed_alpha_raises_instead_of_silent_nan() -> None:
    x = pd.DataFrame({"A": np.arange(1.0, 6.0)})

    with pytest.raises(OperatorParameterError, match="alpha"):
        Ridge().calculate(x, 5, object())
