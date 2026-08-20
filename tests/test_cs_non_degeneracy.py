"""CrossSectionalNonDegeneracy certification regression tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cleaned_operators import load_all  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


# ---------------------------------------------------------------------------
# Helpers (mirror the certification script's logic inline for test isolation)
# ---------------------------------------------------------------------------

N_ROWS = 120
N_COLS = 20
EPSILON = 1e-12
WARMUP_FRAC = 0.40
MIN_PASS_FRAC = 0.50


def _make_panels():
    idx = pd.date_range("2024-06-01", periods=N_ROWS, freq="D")
    rng = np.random.default_rng(99)
    drift = rng.uniform(-0.002, 0.005, size=N_COLS)
    noise = rng.standard_normal((N_ROWS, N_COLS)) * 0.02
    log_price = np.cumsum(drift[None, :] + noise, axis=0) + np.log(100.0)
    close = pd.DataFrame(np.exp(log_price), index=idx,
                         columns=[f"S{i:03d}" for i in range(N_COLS)])
    return close


def _cs_std(frame: pd.DataFrame) -> pd.Series:
    return frame.astype(float).std(axis=1, ddof=1)


def _pass_fraction(frame: pd.DataFrame) -> float:
    cs = _cs_std(frame.dropna())
    n = int(cs.count())
    if n == 0:
        return 0.0
    start = int(len(cs) * WARMUP_FRAC)
    cs_pw = cs.iloc[start:]
    total = int(cs_pw.count())
    passing = int((cs_pw > EPSILON).sum())
    return passing / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCSNonDegeneracyRSIWilder:
    """RSI_WILDER on heterogeneous panel must pass cs non-degeneracy."""

    def test_pass_fraction_above_threshold(self):
        from cleaned_operators.registry import OperatorRegistry
        from scripts.audit_all_factor_production import _build_call

        close = _make_panels()
        high = close + 1.0
        low = close - 1.0
        volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)

        panels = {
            "x": close, "y": close, "close": close, "open": close,
            "high": high, "low": low, "volume": volume, "amount": volume,
            "vwap": close, "turnover": volume, "ret": close.pct_change(),
            "signal": close.pct_change(),
            "condition": (close.pct_change() > 0).astype(float),
            "weight": pd.DataFrame(1.0 / N_COLS, index=close.index, columns=close.columns),
            "returns": close.pct_change(),
        }
        op = OperatorRegistry.get("RSI_WILDER", backend="pandas_numpy")
        assert op is not None, "RSI_WILDER not available"
        args, kwargs = _build_call("RSI_WILDER", op, panels)
        out = op.calculate(*args, **kwargs)
        if isinstance(out, pd.Series):
            out = pd.DataFrame(out)
        assert isinstance(out, pd.DataFrame), f"expected DataFrame, got {type(out)}"
        fp = _pass_fraction(out)
        assert fp >= MIN_PASS_FRAC, f"RSI_WILDER cs pass fraction {fp:.2%} < {MIN_PASS_FRAC:.0%}"


class TestCSNonDegeneracyGlobalState:
    """cs_count returns identical value per date across stocks -> GLOBAL_STATE."""

    def test_cs_count_is_global_state(self):
        from cleaned_operators.registry import OperatorRegistry
        from scripts.audit_all_factor_production import _build_call

        close = _make_panels()
        panels = {
            "x": close, "y": close, "close": close, "open": close,
            "high": close + 1.0, "low": close - 1.0,
            "volume": pd.DataFrame(1e6, index=close.index, columns=close.columns),
            "amount": close,
            "vwap": close, "turnover": close,
            "ret": close.pct_change(), "signal": close.pct_change(),
            "condition": (close.pct_change() > 0).astype(float),
            "weight": pd.DataFrame(1.0 / N_COLS, index=close.index, columns=close.columns),
            "returns": close.pct_change(),
        }
        op = OperatorRegistry.get("cs_count", backend="pandas_numpy")
        assert op is not None, "cs_count not available"
        args, kwargs = _build_call("cs_count", op, panels)
        out = op.calculate(*args, **kwargs)
        if isinstance(out, pd.Series):
            out = pd.DataFrame(out)
        assert isinstance(out, pd.DataFrame)
        cs = _cs_std(out.dropna())
        assert bool((cs < EPSILON).all()), (
            "cs_count should produce constant cross-sectional output (GLOBAL_STATE)"
        )


class TestCSNonDegeneracySyntheticFails:
    """A uniform (no heterogeneity) panel should fail cs non-degeneracy."""

    def test_uniform_panel_fails(self):
        idx = pd.date_range("2024-06-01", periods=N_ROWS, freq="D")
        uniform = pd.DataFrame(100.0, index=idx, columns=[f"S{i:03d}" for i in range(N_COLS)])
        cs = _cs_std(uniform.dropna())
        start = int(len(cs) * WARMUP_FRAC)
        cs_pw = cs.iloc[start:]
        total = int(cs_pw.count())
        passing = int((cs_pw > EPSILON).sum())
        frac = passing / total if total > 0 else 0.0
        assert frac == 0.0, f"uniform panel should have zero pass fraction, got {frac:.2%}"
