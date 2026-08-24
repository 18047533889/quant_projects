# -*- coding: utf-8 -*-
"""Round-14 P0 regression tests for ``ts_delay_intrinsic_dimension``.

Two correctness bugs were fixed in ``cleaned_operators/intrinsic_dimension.py``:

BUG 1 -- Theiler window used the *compressed* post-NaN row ordinal.  After the
NaN embedding vectors were dropped, ``idx = arange(n_pts)`` made two surviving
points that are far apart in *real* time look temporally adjacent, so the
temporal-exclusion window wrongly removed (or kept) neighbour pairs.
``_delay_points`` now returns ``(points, original_time_index)`` and the Theiler
exclusion is evaluated on real-time distance only.

BUG 2 -- Distinct-point detection used ``np.unique(pts.round(10), axis=0)``, a
fixed absolute decimal precision.  ``f(x)``, ``f(1000*x)`` and ``f(1e-6*x)``
produced different distinct-point counts (and therefore different
``n_unique >= k+1`` gates), corrupting the estimate for returns / spreads /
log-returns.  The duplicate definition is now scale-robust: each embedding
dimension is normalised by its robust per-dimension scale (MAD, std fallback)
and points are duplicates only when their normalised co-ordinates agree to a
relative tolerance.

The operator embeds each field independently (a single 1D delay embedding), so
a genuine 2-D grid manifold is not expressible through this operator; the
sanity section therefore uses deterministic 1-D sequences with a known answer
(a smooth line -> ID ~ 1, iid noise -> ID ~ embedding_dim).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.intrinsic_dimension as im
from factor_engine.cleaned_operators.registry import OperatorRegistry

CANONICAL = "ts_delay_intrinsic_dimension"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _operator():
    op = OperatorRegistry.get(CANONICAL, "pandas_numpy")
    assert op is not None, f"{CANONICAL} pandas_numpy backend missing"
    return op


def _eval(values: np.ndarray, **params) -> float:
    """Last-window intrinsic-dimension estimate for ``values`` (one column)."""
    df = pd.DataFrame({"A": np.asarray(values, dtype=float)})
    out = _operator().calculate(df, **params)
    return float(out["A"].iloc[-1])


def _ref_embed(chunk: np.ndarray, dim: int, delay: int):
    """Independent re-derivation of the surviving embedding points.

    Returns ``(points, original_time_index)`` computed directly from the chunk,
    so a regression in ``_delay_points`` (e.g. returning the compressed ordinal)
    is caught by comparing against the operator output.
    """
    n = int(chunk.shape[0])
    lag = delay * (dim - 1)
    if n < lag + 1:
        return None, None
    pts = np.stack([chunk[s - lag : s + 1 : delay] for s in range(lag, n)], axis=0)
    orig = np.arange(lag, n, dtype=np.int64)
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite], orig[finite]


def _ref_id(pts: np.ndarray, orig: np.ndarray, k: int, theiler_window: int, use_original_time: bool) -> float:
    """Reference Levina-Bickel ID with a selectable Theiler time basis.

    ``use_original_time=True`` evaluates the Theiler exclusion on real time;
    ``False`` reproduces the old (buggy) compressed post-NaN ordinal.
    """
    if pts.shape[0] < 2:
        return np.nan
    # scale-robust distinct-point gate (BUG 2; independent of the Theiler logic)
    if im._distinct_count_scale_robust(pts) < k + 1:
        return np.nan
    if pts.shape[0] < k + 1:
        return np.nan
    d = np.sqrt(np.maximum(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, np.inf)
    if theiler_window > 0:
        times = orig if use_original_time else np.arange(len(orig), dtype=np.int64)
        temporal = np.abs(times[:, None] - times[None, :]) <= theiler_window
        np.fill_diagonal(temporal, False)
        d[temporal] = np.inf
    d_sorted = np.sort(d, axis=1)[:, :k]
    t_k = d_sorted[:, -1]
    t_j = d_sorted[:, :-1]
    valid = (
        (t_k > 0.0)
        & (t_j[:, 0] > 0.0)
        & np.isfinite(t_k)
        & np.isfinite(t_j).all(axis=1)
    )
    if not valid.any():
        return np.nan
    log_term = np.sum(np.log(t_k[valid, None] / t_j[valid]), axis=1)
    dims = (k - 1) / log_term
    dims = dims[np.isfinite(dims) & (dims > 0.0)]
    if dims.size == 0:
        return np.nan
    return float(np.median(dims))


# Each row: a window with a block of 1 / 3 / 10 NaN values placed so the
# surviving embedding points straddle a real-time gap larger than the Theiler
# window.  ``_delay_points(chunk, dim=2, delay=1)`` keeps vectors ``s`` whose
# co-ordinates ``x[s-1], x[s]`` are both finite.
_NAN_CHUNKS = [
    # 1 NaN at position 2
    np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]),
    # 3 consecutive NaNs at positions 2..4
    np.array([0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0]),
    # 10 consecutive NaNs at positions 2..11
    np.array([0.0, 1.0] + [np.nan] * 10 + [12.0, 13.0, 14.0, 15.0]),
]
_NaN_IDS = ["1_nan", "3_nan", "10_nan"]


@pytest.fixture(scope="module", autouse=True)
def _operator_ready():
    # Importing the module registers the operator directly; no full
    # ``load_all()`` sweep is required to exercise this one canonical.
    op = OperatorRegistry.get(CANONICAL, "pandas_numpy")
    assert op is not None, f"{CANONICAL} not registered on module import"
    return op


# --------------------------------------------------------------------------- #
# BUG 1 -- Theiler window must use real time, not the compressed ordinal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", _NAN_CHUNKS, ids=_NaN_IDS)
def test_delay_points_tracks_original_time(chunk: np.ndarray) -> None:
    pts, orig = im._delay_points(chunk, dim=2, delay=1)
    assert pts is not None
    assert pts.shape[1] == 2
    # Surviving vectors keep their *original* integer time positions, with a
    # gap (real-time distance > theiler_window=2) across the NaN block.
    assert orig[1] - orig[0] > 2, f"no real-time gap: {orig.tolist()}"
    # The latest coordinate of each surviving vector is the chunk value at that
    # original time position -- proves the mapping is not lost.
    np.testing.assert_allclose(pts[:, 1], chunk[orig])


@pytest.mark.parametrize("chunk", _NAN_CHUNKS, ids=_NaN_IDS)
def test_theiler_exclusion_uses_real_time_not_compressed_index(chunk: np.ndarray) -> None:
    params = dict(window=len(chunk), embedding_dim=2, k=3, delay=1, theiler_window=2)
    got = _eval(chunk, **params)

    pts, orig = im._delay_points(chunk, dim=2, delay=1)
    assert pts is not None and orig is not None
    assert len(orig) >= 4  # enough survivors for k=3 (need k+1 distinct)

    # Independent references: real-time Theiler vs the old compressed ordinal.
    real_time_ref = _ref_id(pts, orig, k=3, theiler_window=2, use_original_time=True)
    compressed_ref = _ref_id(pts, orig, k=3, theiler_window=2, use_original_time=False)

    # The cross-gap pair (first survivor, first post-gap survivor) is further
    # apart in real time than the Theiler window, so it must NOT be excluded.
    assert orig[1] - orig[0] > 2
    assert np.isfinite(real_time_ref)

    # Corrected behaviour: operator output equals the real-time reference.
    assert np.isfinite(got), f"operator produced NaN; real-time ref={real_time_ref}"
    assert np.isclose(got, real_time_ref, rtol=1e-9), f"got={got} real_time_ref={real_time_ref}"

    # The compressed-ordinal variant wrongly excludes the cross-gap pair and
    # must give a different (here NaN) answer -- this is what the bug caused.
    if np.isnan(compressed_ref):
        assert np.isfinite(real_time_ref), "corrected path should stay finite"
    else:
        assert not np.isclose(real_time_ref, compressed_ref, rtol=1e-9), (
            "compressed and real-time Theiler should differ for this chunk"
        )


# --------------------------------------------------------------------------- #
# BUG 2 -- distinct-point detection must be scale-invariant
# --------------------------------------------------------------------------- #
def test_distinct_count_scale_invariant() -> None:
    rng = np.random.default_rng(42)
    # Tiny relative fluctuation: 1e-5 of the level.  The old ``round(10)`` rule
    # merges every point once the series is scaled to ~1e-6 (differences drop
    # below 1e-10), collapsing the distinct count to 1.
    v = 1.0 + 1e-5 * rng.normal(0.0, 1.0, 80)
    pts, _ = im._delay_points(np.asarray(v, dtype=float), dim=3, delay=1)
    assert pts is not None and pts.shape[0] >= 6

    counts = [im._distinct_count_scale_robust(pts * f) for f in (1.0, 1000.0, 1e-6)]
    assert counts[0] == counts[1] == counts[2], f"MAD-distinct counts differ: {counts}"
    assert counts[0] >= 6

    # Show the removed rule is genuinely broken for this series (guard against
    # the test silently passing on a scale where the old rule happened to work).
    old = [np.unique(np.round(pts * f, 10), axis=0).shape[0] for f in (1.0, 1000.0, 1e-6)]
    assert old[0] > old[2], f"old round(10) counts should collapse at small scale: {old}"


def _scale_processes() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 120
    t = np.arange(n, dtype=float)
    rw = np.cumsum(rng.normal(0.0, 1.0, n))
    return {
        "random_walk": rw,
        "gaussian_noise": rng.normal(0.0, 1.0, n),
        "absolute_spread": np.abs(rng.normal(0.0, 1.0, n)) + 0.1,
        "tiny_fluctuation": 1.0 + 1e-5 * rng.normal(0.0, 1.0, n),
        "sine_drift": np.sin(2 * np.pi * t / 12.0) + t * 0.02,
    }


@pytest.mark.parametrize(
    "name",
    ["random_walk", "gaussian_noise", "absolute_spread", "tiny_fluctuation", "sine_drift"],
)
def test_scale_metamorphic_operator(name: str) -> None:
    v = _scale_processes()[name]
    params = dict(window=60, embedding_dim=3, k=5, delay=1)  # default Theiler
    ids = [_eval(v * f, **params) for f in (1.0, 1000.0, 1e-6)]
    assert all(np.isfinite(x) for x in ids), f"{name}: not all finite: {ids}"
    span = max(ids) - min(ids)
    # f(x) vs f(1000x) vs f(1e-6x) must be essentially identical.
    assert span < 1e-3, f"{name}: ID not scale-invariant: {ids} (span={span})"


# --------------------------------------------------------------------------- #
# sanity -- deterministic sequences with a known answer
# --------------------------------------------------------------------------- #
def test_sanity_smooth_1d_line_has_low_dimension() -> None:
    n = 120
    line = np.arange(n, dtype=float)
    params = dict(window=n, embedding_dim=3, k=10, delay=1, theiler_window=0)
    id_line = _eval(line, **params)
    # A 1-D smooth curve delay-embeds onto a 1-D curve -> ID ~ 1.
    assert 0.8 <= id_line <= 1.8, f"1-D smooth line ID={id_line} not ~1"


def test_sanity_noise_has_high_dimension() -> None:
    n = 120
    rng = np.random.default_rng(7)
    line = np.arange(n, dtype=float)
    noise = rng.normal(0.0, 1.0, n)
    params = dict(window=n, embedding_dim=3, k=10, delay=1, theiler_window=0)
    id_line = _eval(line, **params)
    id_noise = _eval(noise, **params)
    assert id_noise > 2.5, f"iid noise ID={id_noise} should approach embedding_dim=3"
    assert id_noise - id_line > 1.5, f"noise({id_noise}) not clearly above line({id_line})"


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_smoke_import_and_eval_shape() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"A": np.linspace(0.0, 1.0, 40), "B": rng.normal(0.0, 1.0, 40)}
    )
    out = _operator().calculate(df, window=20, embedding_dim=3, k=5, delay=1, theiler_window=2)
    assert out.shape == df.shape
    assert list(out.columns) == ["A", "B"]
    # a trailing window produces a finite value once enough rows have accrued
    assert np.isfinite(out["A"].iloc[-1])
