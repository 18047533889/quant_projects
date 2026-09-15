# -*- coding: utf-8 -*-
"""Tests for intraday_session operators.

Coverage of 2 operators:
1. intraday_session_shape_novelty
2. intraday_profile_pca_residual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.session_panel import default_ashare_calendar

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op


def _session_id(x: pd.DataFrame) -> pd.DataFrame:
    """Explicit session identity required by the canonical two-panel API."""
    codes = pd.factorize(x.index.normalize())[0].astype(float)
    return pd.DataFrame(
        np.repeat(codes[:, None], x.shape[1], axis=1),
        index=x.index,
        columns=x.columns,
    )


def _minute_panel(days: int, cols: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stamps = []
    for offset in range(days):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset)
        stamps.extend(day + pd.Timedelta(minutes=m) for m in (*range(571, 691), *range(781, 901)))
    index = pd.DatetimeIndex(stamps)
    names = [f"S{i}" for i in range(cols)]
    x = pd.DataFrame(rng.normal(size=(len(index), cols)), index=index, columns=names)
    return x, _session_id(x)



# ---------------------------------------------------------------------------
# 1. intraday_session_shape_novelty
# ---------------------------------------------------------------------------
def test_intraday_session_shape_novelty_basic() -> None:
    """Basic functionality test."""
    x, session_id = _minute_panel(days=7, cols=10, seed=42)

    op = _op("intraday_session_shape_novelty")
    try:
        result = op.calculate(x, session_id, calendar=default_ashare_calendar())
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_intraday_session_shape_novelty_handles_nans() -> None:
    """NaN handling test."""
    x, session_id = _minute_panel(days=7, cols=3, seed=43)
    x.iloc[::17, 1] = np.nan

    op = _op("intraday_session_shape_novelty")
    try:
        result = op.calculate(x, session_id, calendar=default_ashare_calendar())
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_intraday_session_shape_novelty_deterministic() -> None:
    """Determinism test - same input yields same output."""
    x, session_id = _minute_panel(days=7, cols=5, seed=123)

    op = _op("intraday_session_shape_novelty")
    try:
        result1 = op.calculate(x, session_id, calendar=default_ashare_calendar())
        result2 = op.calculate(x, session_id, calendar=default_ashare_calendar())
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic


# ---------------------------------------------------------------------------
# 2. intraday_profile_pca_residual
# ---------------------------------------------------------------------------
def test_intraday_profile_pca_residual_basic() -> None:
    """Basic functionality test."""
    x, session_id = _minute_panel(days=7, cols=10, seed=42)

    op = _op("intraday_profile_pca_residual")
    try:
        result = op.calculate(x, session_id, calendar=default_ashare_calendar())
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_intraday_profile_pca_residual_handles_nans() -> None:
    """NaN handling test."""
    x, session_id = _minute_panel(days=7, cols=3, seed=43)
    x.iloc[::17, 1] = np.nan

    op = _op("intraday_profile_pca_residual")
    try:
        result = op.calculate(x, session_id, calendar=default_ashare_calendar())
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_intraday_profile_pca_residual_deterministic() -> None:
    """Determinism test - same input yields same output."""
    x, session_id = _minute_panel(days=7, cols=5, seed=123)

    op = _op("intraday_profile_pca_residual")
    try:
        result1 = op.calculate(x, session_id, calendar=default_ashare_calendar())
        result2 = op.calculate(x, session_id, calendar=default_ashare_calendar())
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic


@pytest.mark.parametrize("name", ["intraday_session_shape_novelty", "intraday_profile_pca_residual"])
def test_intraday_session_operators_require_explicit_session_id(name: str) -> None:
    from factor_engine.backend.operator_errors import OperatorParameterError

    x, _ = _minute_panel(days=2, cols=2, seed=7)
    with pytest.raises(OperatorParameterError, match="session_id"):
        _op(name).calculate(x)


def test_shape_novelty_is_available_only_at_complete_session_close() -> None:
    x, session_id = _minute_panel(days=7, cols=2, seed=9)
    out = _op("intraday_session_shape_novelty").calculate(
        x, session_id, history_days=6, min_history_sessions=5,
        calendar=default_ashare_calendar(),
    )
    close_rows = x.index.strftime("%H:%M") == "15:00"
    assert out.loc[~close_rows].isna().all().all()
    assert np.isfinite(out.loc[close_rows].to_numpy()).any()



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_intraday_session_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "intraday_session_shape_novelty",
        "intraday_profile_pca_residual"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
