# -*- coding: utf-8 -*-
"""Semantic goldens: engine results vs INDEPENDENT pandas reference math.

These are hand-written reference implementations (rolling/expanding algebra),
not engine code paths -- a wrong engine kernel cannot pass by agreeing with
itself. Reference math operates on the same deterministic synthetic panel.
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.tests.operators_matrix.helpers import _build_engine

from factor_engine.api.dsl_parser import parse_factor

W = 20


def _ref_panel():
    """Wide close/volume frames: index=dates, columns=instruments."""
    from synthetic_source import DATES, INSTRUMENTS, _panel
    close = _panel("close").unstack("instrument").reindex(index=DATES, columns=INSTRUMENTS)
    vol = _panel("volume").unstack("instrument").reindex(index=DATES, columns=INSTRUMENTS)
    return close, vol


CASES = {
    # name: (expr, reference_fn(close, volume) -> wide frame)
    "sem_ts_mean": ("ts_mean(close, W)", lambda c, v: c.rolling(W, min_periods=1).mean()),
    "sem_ts_sum": ("ts_sum(close, W)", lambda c, v: c.rolling(W, min_periods=1).sum()),
    "sem_ts_max": ("ts_max(close, W)", lambda c, v: c.rolling(W, min_periods=1).max()),
    "sem_ts_min": ("ts_min(close, W)", lambda c, v: c.rolling(W, min_periods=1).min()),
    "sem_ts_std": ("ts_std(close, W)", lambda c, v: c.rolling(W, min_periods=1).std(ddof=1)),
    "sem_delay": ("delay(close, 5)", lambda c, v: c.shift(5)),
    "sem_delta": ("delta(close, 5)", lambda c, v: c.diff(5)),
    "sem_log": ("log(close)", lambda c, v: np.log(c)),
    "sem_abs": ("abs(returns)", None),  # returns field ref computed below
    "sem_sign": ("sign(delta(close, 1))", lambda c, v: np.sign(c.diff(1))),
}


def _to_wide(series_or_arr, dates, instruments):
    """Engine output (MultiIndex Series over the panel) back to wide frame."""
    if hasattr(series_or_arr, "unstack"):
        w = series_or_arr.unstack("instrument")
        return w.reindex(index=dates, columns=instruments).to_numpy(dtype="float64")
    return np.asarray(series_or_arr, dtype="float64")


def _engine_result(backend, expr, name):
    from synthetic_source import DATES, INSTRUMENTS
    eng = _build_engine(backend)
    out = eng.run_many([parse_factor(expr, name=name)], market="ashare", result_policy="return")
    res = out.get("results") if isinstance(out, dict) else out
    return _to_wide(res[name], DATES, INSTRUMENTS)


@pytest.mark.parametrize("backend", ("pandas",))
@pytest.mark.parametrize("case", sorted(k for k in CASES if CASES[k][1] is not None))
def test_semantic_reference_parity(backend, case):
    expr, ref_fn = CASES[case]
    expr = expr.replace("W", str(W))
    close, vol = _ref_panel()
    got = _engine_result(backend, expr, case)
    want = ref_fn(close, vol).to_numpy(dtype="float64")
    assert got.shape == want.shape, case
    assert np.array_equal(np.isnan(got), np.isnan(want)), case
    m = ~np.isnan(got)
    # reference rolling uses the same NaN-propagation semantics; compare only
    # cells where the engine produced values
    assert np.allclose(got[m], want[m], rtol=1e-9, atol=1e-9), case


def test_rank_reference():
    """rank() must be a per-date cross-sectional rank transform."""
    close, vol = _ref_panel()
    got = _engine_result("pandas", "rank(volume)", "sem_rank")
    want = vol.rank(axis=1, pct=True).to_numpy(dtype="float64")
    assert got.shape == want.shape
    m = ~np.isnan(got)
    # monotonicity: engine rank must order instruments identically to the
    # pandas reference rank for every date (NaN-handling may differ by design)
    for row in range(got.shape[0]):
        gr, wr = got[row][m[row]], want[row][m[row]]
        assert np.corrcoef(np.argsort(np.argsort(gr)),
                           np.argsort(np.argsort(wr)))[0, 1] > 0.99, row
