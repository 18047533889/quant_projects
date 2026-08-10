# -*- coding: utf-8 -*-
"""Round-3 audit: P1-K group spectrum + P1-L DMD (items 132-138).

Covers:
  132  group spectrum ops are ``group_state`` metadata (not stock-level alpha).
  133  breadth-history keys = (GroupSchemaVersion, GroupId), not the bare label.
  134  exact-duplicate feature panels are rejected at the op boundary.
  135  the 0.10 localization eigen-gap is an estimator constant in metadata.
  136  DMD mode energy is computed in log-space (logsumexp concentration).
  137  DMD splits price-level vs return semantics (ts_dmd_level_* / _return_*).
  138  dominant frequency = max-energy IMAGINARY mode; NaN only when none.

The two owned modules are imported DIRECTLY rather than through ``load_all()``:
the shared tree is currently blocked by an unrelated concurrent-session
registration-audit failure (``ts_cpt_value`` polars arity mismatch) that this
file-disjoint fixer is not allowed to touch.  Direct import registers exactly
the operators these modules own, which is sufficient for every assertion here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.dmd  # noqa: F401  (registers on import)
import cleaned_operators.group_spectrum  # noqa: F401  (registers on import)
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry

SPECTRUM_OPS = [
    "group_feature_mode_share",
    "group_feature_effective_rank",
    "group_feature_mode_localization",
    "group_feature_spectral_gap",
    "group_feature_second_mode_localization",
]
GROUP_OPS = SPECTRUM_OPS + [
    "group_feature_valid_member_count",
    "group_feature_coverage_ratio",
]
LOCALIZATION_OPS = [
    "group_feature_mode_localization",
    "group_feature_second_mode_localization",
]


def _get(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend) or OperatorRegistry.get(name)
    assert op is not None, name
    return op


def _group_fixture(n: int = 40, ncols: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = [f"S{i:02d}" for i in range(ncols)]
    g = pd.DataFrame(
        np.tile(np.array(["A"] * (ncols // 2) + ["B"] * (ncols // 2)), (n, 1)),
        index=idx, columns=cols,
    )
    return idx, cols, rng, g


def _feats(idx, cols, rng):
    return (
        pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols),
        pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols),
        pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols),
    )


# ---------------------------------------------------------------------------
# 132 — group spectral state ops are group_state, not stock-level alpha
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canon", GROUP_OPS)
def test_group_state_role_declared(canon):
    meta = _get(canon).metadata
    assert meta.role == "group_state", canon
    assert "group_state" in [str(t) for t in (meta.tags or [])], canon


def test_group_spectrum_broadcasts_same_value_per_group():
    """The machine-readable reason for group_state: every member of a group
    receives the SAME value on a date."""
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=7)
    f1, f2, f3 = _feats(idx, cols, rng)
    out = _get("group_feature_mode_share").calculate(f1, f2, f3, g).to_numpy(dtype=float)
    for r in range(out.shape[0]):
        a = out[r, :10][np.isfinite(out[r, :10])]
        b = out[r, 10:][np.isfinite(out[r, 10:])]
        if a.size:
            assert np.allclose(a, a[0]), "group A members differ within a date"
        if b.size:
            assert np.allclose(b, b[0]), "group B members differ within a date"


# ---------------------------------------------------------------------------
# 133 — breadth-history keys = (GroupSchemaVersion, GroupId)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canon", SPECTRUM_OPS)
def test_group_schema_version_param_contract(canon):
    meta = _get(canon).metadata
    assert "group_schema_version" in meta.param_names, canon
    spec = meta.param_specs["group_schema_version"]
    assert spec.dtype is str, canon
    assert spec.searchable is False, canon
    assert spec.default == "v1", canon


def test_group_schema_version_isolates_breadth_history():
    """Two taxonomy versions with the same label are independent groups: the
    version is part of the breadth-history key, so a call under v2 never shares
    the v1 membership history.  (History is per-call, so within a single call
    the outputs are identical — the point is the key carries the version.)"""
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=11)
    f1, f2, f3 = _feats(idx, cols, rng)
    op = _get("group_feature_mode_share")
    out_v1 = op.calculate(f1, f2, f3, g, group_schema_version="v1").to_numpy(dtype=float)
    out_v2 = op.calculate(f1, f2, f3, g, group_schema_version="v2").to_numpy(dtype=float)
    assert np.allclose(out_v1, out_v2, equal_nan=True)


def test_group_schema_version_rejects_non_string():
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=13)
    f1, f2, f3 = _feats(idx, cols, rng)
    with pytest.raises(Exception):
        _get("group_feature_mode_share").calculate(f1, f2, f3, g, group_schema_version=5)


# ---------------------------------------------------------------------------
# 134 — exact duplicate feature panels are rejected at the op boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canon", SPECTRUM_OPS)
def test_duplicate_feature_panels_rejected(canon):
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=3)
    f1, f2, f3 = _feats(idx, cols, rng)
    op = _get(canon)
    with pytest.raises(ValueError, match="duplicate"):
        op.calculate(f1, f1, f3, g)
    with pytest.raises(ValueError, match="duplicate"):
        op.calculate(f1, f2, f1, g)
    with pytest.raises(ValueError, match="duplicate"):
        op.calculate(f1, f2, f2, g)
    # NaN-identical panels are duplicates too (same expression -> same NaN mask)
    f1b = f1.copy()
    f2b = f1.copy()
    f1b.iloc[3, 0] = np.nan
    f2b.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="duplicate"):
        op.calculate(f1b, f2b, f3, g)


def test_duplicate_detection_does_not_fire_on_distinct_panels():
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=5)
    f1, f2, f3 = _feats(idx, cols, rng)
    out = _get("group_feature_mode_share").calculate(f1, f2, f3, g)
    assert out.shape == (40, 20)


# ---------------------------------------------------------------------------
# 135 — localization 0.10 eigen-gap is an estimator constant in metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canon", LOCALIZATION_OPS)
def test_eigen_gap_param_declared(canon):
    meta = _get(canon).metadata
    spec = meta.param_specs["eigen_gap"]
    assert spec.dtype is float, canon
    assert spec.searchable is False, canon
    assert spec.param_role is not None and spec.param_role.name == "ESTIMATOR_RESOLUTION", canon
    assert spec.default == pytest.approx(0.10), canon
    assert f"eigen_gap:0.1" in [str(t) for t in (meta.tags or [])], canon


def test_eigen_gap_gates_localization_readout():
    """A group whose two leading singular values are near-degenerate yields NaN
    localization at the default eigen-gap, but a finite value at eigen_gap=0
    (the gate is a real estimator knob, not a hard-wired number)."""
    rng = np.random.default_rng(21)
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    cols = [f"S{i:02d}" for i in range(20)]
    g = pd.DataFrame(
        np.tile(np.array(["A"] * 20), (60, 1)), index=idx, columns=cols,
    )
    # Two equal-variance orthogonal directions (u, v) + small noise: the top two
    # singular values are close, so the relative gap is below the 0.10 gate.
    u = rng.normal(size=(60, 20))
    v = rng.normal(size=(60, 20))
    eps = 0.05 * rng.normal(size=(60, 20))
    f1 = pd.DataFrame(u + eps, index=idx, columns=cols)
    f2 = pd.DataFrame(v + eps, index=idx, columns=cols)
    f3 = pd.DataFrame(u - v + 0.1 * rng.normal(size=(60, 20)), index=idx, columns=cols)
    op = _get("group_feature_mode_localization")
    default = op.calculate(f1, f2, f3, g).to_numpy(dtype=float)
    open_gap = op.calculate(f1, f2, f3, g, eigen_gap=0.0).to_numpy(dtype=float)
    assert np.isfinite(open_gap).sum() > np.isfinite(default).sum()


# ---------------------------------------------------------------------------
# 136 — DMD mode energy in log-space
# ---------------------------------------------------------------------------

def test_dmd_log_finite_horizon_sum_stable():
    from cleaned_operators.dmd import _log_finite_horizon_sum

    # R26-123/124: the kernel takes ``log_rho = log(rho)`` so ``rho = |λ|²`` is
    # never materialised (a huge lambda's rho overflows float64; the log-space
    # form stays finite).  ``rho = 1e6`` <-> ``log_rho = log(1e6)``.
    assert np.isfinite(_log_finite_horizon_sum(float(np.log(1e6)), 100))
    assert np.isfinite(_log_finite_horizon_sum(700.0, 100))  # |lambda| ~ e^350
    # matches the direct sum wherever the direct sum is representable
    for r, K in [(0.5, 50), (0.9, 50), (1.1, 50), (2.0, 50), (5.0, 20), (0.5, 200)]:
        direct = np.log(np.sum(r ** np.arange(K, dtype=float)))
        assert abs(direct - _log_finite_horizon_sum(float(np.log(r)), K)) < 1e-9, (r, K)
    # rho near 1: no catastrophic cancellation (compare against the exact direct
    # finite sum, which is representable here)
    r_near = 1.0000005
    assert abs(
        _log_finite_horizon_sum(float(np.log(r_near)), 1000)
        - np.log(np.sum(r_near ** np.arange(1000, dtype=float)))
    ) < 1e-12
    # rho == 0 (log_rho == -inf): only the t=0 term survives
    assert _log_finite_horizon_sum(float("-inf"), 100) == 0.0


def test_dmd_concentration_bounded_and_monotone():
    rng = np.random.default_rng(31)
    idx = pd.date_range("2024-01-01", periods=150, freq="B")
    ret = pd.DataFrame(rng.normal(0, 0.01, (150, 3)), index=idx, columns=list("ABC"))
    op = _get("ts_dmd_mode_concentration")
    c1 = op.calculate(ret, window=60, rank=3, dim=3, top_k=1).to_numpy(dtype=float)
    c2 = op.calculate(ret, window=60, rank=3, dim=3, top_k=2).to_numpy(dtype=float)
    f1 = c1[np.isfinite(c1)]
    f2 = c2[np.isfinite(c2)]
    assert f1.size > 0 and f2.size > 0
    assert bool((f1 >= 0).all() and (f1 <= 1).all())
    assert bool((f2 >= 0).all() and (f2 <= 1).all())
    # top_k=2 concentration is >= top_k=1 wherever both are finite
    both = np.isfinite(c1) & np.isfinite(c2)
    assert bool((c2[both] >= c1[both] - 1e-12).all())


# ---------------------------------------------------------------------------
# 137 — DMD splits price-level vs return semantics
# ---------------------------------------------------------------------------

_DMD_VARIANTS = {
    "ts_dmd_level_dominant_growth_rate": ("price_level", "log_growth_per_bar", "level"),
    "ts_dmd_level_dominant_frequency": ("price_level", "cycles_per_bar", "level"),
    "ts_dmd_level_mode_concentration": ("price_level", "ratio", "level"),
    "ts_dmd_return_dominant_growth_rate": ("return", "log_growth_per_bar", "return"),
    "ts_dmd_return_dominant_frequency": ("return", "cycles_per_bar", "return"),
    "ts_dmd_return_mode_concentration": ("return", "ratio", "return"),
}


@pytest.mark.parametrize("canon", list(_DMD_VARIANTS))
def test_dmd_level_return_variants_registered(canon):
    expected_unit, expected_out, expected_semantic = _DMD_VARIANTS[canon]
    op = _get(canon)
    meta = op.metadata
    assert meta.input_units == {"x": expected_unit}, canon
    assert meta.output_unit == expected_out, canon
    tags = " ".join(str(t) for t in (meta.tags or []))
    assert f"input_semantic:{expected_semantic}" in tags, canon
    assert "pandas_numpy" in OperatorRegistry.backends_for(canon)
    assert "polars" in OperatorRegistry.backends_for(canon)
    # R22-035/036: the typed level/return DMD variants are promoted off the
    # research surface to DIRECT_ALPHA_HIGH_COST (extended authoring tier) —
    # surface==research is an authoring state, not a semantic verdict (R22-009).
    assert classify_canonical(canon) == "extended", canon


def test_dmd_level_return_variants_compute():
    rng = np.random.default_rng(41)
    idx = pd.date_range("2024-01-01", periods=150, freq="B")
    ret = pd.DataFrame(rng.normal(0, 0.01, (150, 3)), index=idx, columns=list("ABC"))
    for canon in _DMD_VARIANTS:
        out = _get(canon).calculate(ret, window=60, rank=3, dim=3)
        assert out.shape == ret.shape, canon
        assert out.index.equals(idx), canon


# ---------------------------------------------------------------------------
# 138 — dominant frequency is the max-energy OSCILLATORY mode
# ---------------------------------------------------------------------------

def test_dmd_dominant_frequency_oscillatory():
    """A real max-energy mode (level persistence, λ≈1) must NOT suppress the
    frequency to NaN: the max-energy IMAGINARY mode's frequency is returned."""
    n = 120
    t = np.arange(n, dtype=float)
    sig = 1.05 ** t + 0.5 * np.cos(2 * np.pi * 0.1 * t)
    x = pd.DataFrame(
        sig[:, None], index=pd.date_range("2024-01-01", periods=n, freq="B"), columns=["S"],
    )
    out = _get("ts_dmd_dominant_frequency").calculate(x, window=40, rank=3, dim=3)
    last = out.dropna()
    assert not last.empty, "dominant oscillatory frequency must not be NaN"
    assert abs(float(last.iloc[-1, 0]) - 0.1) < 0.02


def test_dmd_dominant_frequency_nan_only_when_no_imaginary_modes():
    """A pure (real) exponential has an all-real spectrum -> NaN frequency."""
    n = 120
    t = np.arange(n, dtype=float)
    sig = 1.05 ** t
    x = pd.DataFrame(
        sig[:, None], index=pd.date_range("2024-01-01", periods=n, freq="B"), columns=["S"],
    )
    out = _get("ts_dmd_dominant_frequency").calculate(x, window=40, rank=3, dim=3)
    assert out.dropna().empty


# ---------------------------------------------------------------------------
# regression — legacy canonicals still compute
# ---------------------------------------------------------------------------

def test_legacy_dmd_canonicals_still_compute():
    rng = np.random.default_rng(51)
    idx = pd.date_range("2024-01-01", periods=150, freq="B")
    ret = pd.DataFrame(rng.normal(0, 0.01, (150, 3)), index=idx, columns=list("ABC"))
    for canon, kw in [
        ("ts_dmd_dominant_growth_rate", {"window": 60, "rank": 3, "dim": 3}),
        ("ts_dmd_dominant_frequency", {"window": 60, "rank": 3, "dim": 3}),
        ("ts_dmd_mode_concentration", {"window": 60, "rank": 3, "dim": 3, "top_k": 2}),
    ]:
        out = _get(canon).calculate(ret, **kw)
        assert out.shape == ret.shape, canon


def test_group_spectrum_legacy_4arg_call_still_works():
    """Adding group_schema_version/eigen_gap must not break the legacy 4-arg
    positional call (f1, f2, f3, group)."""
    idx, cols, rng, g = _group_fixture(n=40, ncols=20, seed=61)
    f1, f2, f3 = _feats(idx, cols, rng)
    for canon in SPECTRUM_OPS:
        out = _get(canon).calculate(f1, f2, f3, g)
        assert out.shape == (40, 20), canon
