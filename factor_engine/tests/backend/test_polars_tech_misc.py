# -*- coding: utf-8 -*-
"""Parity tests for native Polars Keltner/Donchian/Bollinger/supertrend/PSAR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(1234)
    index = pd.date_range("2022-01-01", periods=150, freq="D")
    close = pd.DataFrame(
        {
            "A": 100.0 + np.cumsum(rng.normal(0.1, 1.0, len(index))),
            "B": 50.0 + np.cumsum(rng.normal(-0.03, 0.8, len(index))),
        },
        index=index,
    )
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    return high, low, close


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_native_polars_tech_misc_matches_pandas(panels):
    high, low, close = panels
    cases = [
        ("KeltnerMid", (close,), {"ema_window": 20}),
        ("KeltnerUpper", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("KeltnerLower", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("KeltnerPosition", (high, low, close), {"ema_window": 20, "atr_window": 14, "multiplier": 2.0}),
        ("donchian_upper", (high,), {"window": 20}),
        ("donchian_lower", (low,), {"window": 20}),
        ("donchian_mid", (high, low), {"window": 20}),
        ("donchian_position", (close, high, low), {"window": 20}),
        ("bollinger_pct_b", (close,), {"window": 20, "std_dev": 2.0}),
        ("bollinger_width", (close,), {"window": 20, "std_dev": 2.0}),
        ("efficiency_ratio", (close,), {"window": 10}),
        ("choppiness_index", (high, low, close), {"window": 14}),
        ("Supertrend", (high, low, close), {"atr_window": 14, "multiplier": 3.0}),
        ("SupertrendDirection", (high, low, close), {"atr_window": 14, "multiplier": 3.0}),
        ("PSAR", (high, low), {"acceleration": 0.02, "maximum": 0.2}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_tech_misc_nan_warmup_matches_pandas():
    rng = np.random.default_rng(77)
    index = pd.date_range("2023-01-01", periods=100, freq="D")
    mask = rng.random((100, 2)) < 0.1
    close = pd.DataFrame(np.cumsum(rng.normal(0.0, 1.0, (100, 2)), axis=0) + 100, index=index, columns=["A", "B"])
    close[mask] = np.nan
    spread = pd.DataFrame(rng.uniform(0.5, 2.0, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    high[mask] = np.nan
    low[mask] = np.nan
    cases = [
        ("KeltnerMid", (close,), {"ema_window": 10}),
        ("KeltnerUpper", (high, low, close), {"ema_window": 10, "atr_window": 8, "multiplier": 2.0}),
        ("KeltnerPosition", (high, low, close), {"ema_window": 10, "atr_window": 8, "multiplier": 2.0}),
        ("donchian_upper", (high,), {"window": 10}),
        ("donchian_position", (close, high, low), {"window": 10}),
        ("bollinger_pct_b", (close,), {"window": 10, "std_dev": 2.0}),
        ("efficiency_ratio", (close,), {"window": 8}),
        ("choppiness_index", (high, low, close), {"window": 8}),
        ("Supertrend", (high, low, close), {"atr_window": 8, "multiplier": 3.0}),
        ("SupertrendDirection", (high, low, close), {"atr_window": 8, "multiplier": 3.0}),
        ("PSAR", (high, low), {"acceleration": 0.02, "maximum": 0.2}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


# ---------------------------------------------------------------------------
# R21-DEF4 guards: Supertrend warmup boundary, post-gap UNKNOWN, Red-Team
# sensitivities — all against pandas-direct oracles (the implementation under
# test is never the oracle).
# ---------------------------------------------------------------------------


def _supertrend_reference_loader():
    """Load indicators_v2.Supertrend source without registry side effects."""
    import ast
    import os
    import types

    src = open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "cleaned_operators",
            "technical",
            "indicators_v2.py",
        ),
        encoding="utf-8",
    ).read()
    tree = ast.parse(src)
    keep = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_pi", "_pf", "_wilder", "_tr", "Supertrend"}
    ]
    module = types.ModuleType("iv2_supertrend_def4_ref")
    exec("import numpy as np\nimport pandas as pd\n" + "\n".join(keep), module.__dict__)
    return module.Supertrend


_PdSupertrend = None


def _pd_supertrend(high, low, close, w, m):
    global _PdSupertrend
    if _PdSupertrend is None:
        _PdSupertrend = _supertrend_reference_loader()
    return _PdSupertrend(high, low, close, w, m)


def test_supertrend_clean_panel_warmup_boundary_exact():
    # R21-DEF4 guard 1: on a clean panel the pandas reference emits its first
    # finite Supertrend at row w+1: Wilder ATR (min_periods=window, TR row 0 is
    # NaN because prev-close is missing) publishes at row w, that first
    # finite-band row is an UNKNOWN (trend=0, NaN — warmup acts like a gap),
    # and row w+1 re-asserts direction.  Pre-fix the polars kernel emitted one
    # row early (row 14 vs pandas row 15 for w=14) via the off-by-one polars
    # ewm warmup.  Pinned against pandas direct, boundary exact.
    rng = np.random.default_rng(4242)
    n = 60
    c = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    h = c + rng.uniform(0.3, 1.5, n)
    l = c - rng.uniform(0.3, 1.5, n)
    w = 14
    high, low, close = (pd.DataFrame({"A": h}), pd.DataFrame({"A": l}), pd.DataFrame({"A": c}))
    expected = _pd_supertrend(high, low, close, w, 3.0)["A"].to_numpy()
    got = OperatorRegistry.get("Supertrend", backend="polars").calculate(
        _polars(high), _polars(low), _polars(close), atr_window=w, multiplier=3.0
    )["A"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert np.all(np.isnan(got[: w + 1])), "warmup rows 0..w must be NaN (pandas parity)"
    assert np.isfinite(got[w + 1]), "first finite Supertrend must be exactly row w+1 on a clean panel"
    assert int(np.argmax(np.isfinite(got))) == w + 1
    # And the warmup boundary must track the window, not a constant.
    for w2 in (5, 8):
        exp2 = _pd_supertrend(high, low, close, w2, 3.0)["A"].to_numpy()
        got2 = OperatorRegistry.get("Supertrend", backend="polars").calculate(
            _polars(high), _polars(low), _polars(close), atr_window=w2, multiplier=3.0
        )["A"].to_numpy()
        np.testing.assert_allclose(got2, exp2, rtol=1e-9, atol=1e-9, equal_nan=True)
        assert int(np.argmax(np.isfinite(got2))) == w2 + 1, f"warmup boundary must be w+1={w2 + 1}"


def test_supertrend_post_gap_unknown_then_reassert():
    # R21-DEF4 guard 2: after a NaN gap the first valid bar publishes UNKNOWN
    # (NaN output), direction is re-asserted at gap+2, and the whole path is
    # pinned against the pandas reference.  Pre-fix the polars kernel re-seeded
    # trend=1 (a manufactured long bias out of a data hole) on the first
    # post-gap bar.
    rng = np.random.default_rng(4243)
    n = 80
    c = np.cumsum(rng.normal(0.0, 1.0, n)) + 50
    h = c + rng.uniform(0.2, 1.2, n)
    l = c - rng.uniform(0.2, 1.2, n)
    gap_lo, gap_hi = 30, 33  # 3-bar gap; row 33 is the first valid bar after it
    for arr in (h, l, c):
        arr[gap_lo:gap_hi] = np.nan
    high, low, close = (pd.DataFrame({"A": h}), pd.DataFrame({"A": l}), pd.DataFrame({"A": c}))
    expected = _pd_supertrend(high, low, close, 7, 2.0)["A"].to_numpy()
    got = OperatorRegistry.get("Supertrend", backend="polars").calculate(
        _polars(high), _polars(low), _polars(close), atr_window=7, multiplier=2.0
    )["A"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert np.all(np.isnan(got[gap_lo:gap_hi])), "gap rows must be NaN"
    assert np.isnan(got[gap_hi]), "first valid bar after the gap must be NaN (UNKNOWN, no fake direction)"
    assert np.isfinite(got[gap_hi + 1]), "direction must be re-asserted at gap+2"
    assert np.isfinite(got[gap_hi + 2]), "machine must stay live after re-asserting direction"


def test_supertrend_redteam_not_midpoint_window_and_multiplier_move_output():
    # R21-DEF4 guard 3 (Red Team): the output must NOT be (high+low)/2, and
    # BOTH window and multiplier must move it (pre-collapse regression: dead
    # parameters).  Oracle: pandas reference direct call.
    rng = np.random.default_rng(4244)
    n = 100
    c = np.cumsum(rng.normal(0.03, 1.0, n)) + 80
    h = c + rng.uniform(0.3, 1.5, n)
    l = c - rng.uniform(0.3, 1.5, n)
    high, low, close = (pd.DataFrame({"A": h}), pd.DataFrame({"A": l}), pd.DataFrame({"A": c}))
    mid = (h + l) / 2.0

    def _got(w, m):
        return OperatorRegistry.get("Supertrend", backend="polars").calculate(
            _polars(high), _polars(low), _polars(close), atr_window=w, multiplier=m
        )["A"].to_numpy()

    base = _got(10, 3.0)
    exp_base = _pd_supertrend(high, low, close, 10, 3.0)["A"].to_numpy()
    np.testing.assert_allclose(base, exp_base, rtol=1e-9, atol=1e-9, equal_nan=True)
    live = np.isfinite(base)
    assert live.sum() >= 30, "non-vacuous: enough live rows to judge"
    assert not np.allclose(base[live], mid[live], rtol=1e-6, atol=1e-6), (
        "Supertrend collapsed to (high+low)/2 — dead window/multiplier regression"
    )
    # Window sensitivity (multiplier fixed): different windows must move the
    # shared-finite output, and each variant must match the pandas reference.
    for w in (4, 14):
        variant = _got(w, 3.0)
        exp = _pd_supertrend(high, low, close, w, 3.0)["A"].to_numpy()
        np.testing.assert_allclose(variant, exp, rtol=1e-9, atol=1e-9, equal_nan=True)
        both = np.isfinite(variant) & live
        assert both.sum() >= 20
        assert not np.allclose(variant[both], base[both], rtol=1e-8, atol=1e-8), (
            f"window={w} left the Supertrend output unchanged — dead parameter"
        )
    # Multiplier sensitivity (window fixed).
    for m in (1.0, 4.0):
        variant = _got(10, m)
        exp = _pd_supertrend(high, low, close, 10, m)["A"].to_numpy()
        np.testing.assert_allclose(variant, exp, rtol=1e-9, atol=1e-9, equal_nan=True)
        both = np.isfinite(variant) & live
        assert both.sum() >= 20
        assert not np.allclose(variant[both], base[both], rtol=1e-8, atol=1e-8), (
            f"multiplier={m} left the Supertrend output unchanged — dead parameter"
        )
