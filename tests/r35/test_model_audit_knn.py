# -*- coding: utf-8 -*-
"""R35 model-audit KNN tests (M-110 / M-111 / M-112 / M-113 / M-114 / M-115 /
M-116 / M-240).

Pins the kernel + documentation-side fixes for the KNN cross-section operators
in ``cleaned_operators/dynamic_knn.py`` and
``cleaned_operators/cross_section_local.py``.  It does NOT touch the
reconciler-owned shared files (model_timing.py / model_contract.py /
model_lane.py / operator_surface.py / layer_governance.py) — reconciler
obligations are only asserted (read-only) where the kernel itself is
responsible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.base import OperatorParameterError, ParamRole, searchable_param_names  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _load():
    load_all()


def _panel(n, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=[chr(ord("A") + i) for i in range(cols)])


def _get(name):
    _load()
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name} not registered"
    return op


DYNAMIC_KNN_OPS = [
    "cs_knn_peer_mean_ex_self",
    "cs_knn_neighbor_retention",
    "cs_knn_graph_dirichlet_energy",
]
LOCAL_OPS = [
    "cs_knn_local_linear_residual",
    "cs_knn_local_gradient_norm",
    "cs_knn_tangent_residual",
]
ALL_KNN_OPS = DYNAMIC_KNN_OPS + LOCAL_OPS


def _local_neighbors():
    """Import the private tie-inclusive neighbour kernel (read-only)."""
    from factor_engine.cleaned_operators.cross_section_local import _neighbors

    return _neighbors


# ---------------------------------------------------------------------------
# M-110: SameTimeCrossSection timing — documented + self exclusion enforced
# ---------------------------------------------------------------------------

def test_m110_local_linear_residual_documented_sametime():
    op = _get("cs_knn_local_linear_residual")
    desc = op.metadata.description
    assert "as_of=0" in desc, desc
    assert "self_excluded=True" in desc, desc
    assert "peer_feature_available=0" in desc, desc
    assert "peer_target_available=0" in desc, desc
    assert "fit-through-t-1" in desc or "NOT fit-through-t-1" in desc, desc
    # module docstring carries the same SameTimeCrossSection block
    import factor_engine.cleaned_operators.cross_section_local as csl

    assert "SameTimeCrossSection" in csl.__doc__
    assert "self_excluded=True" in csl.__doc__
    assert "peer_target_available=0" in csl.__doc__
    # the reconciler contract (read-only assertion) is fit_cutoff_offset=0
    from factor_engine.cleaned_operators.model_timing import MODEL_TIMING_CONTRACTS

    c = MODEL_TIMING_CONTRACTS["cs_knn_local_linear_residual"]
    assert c.fit_cutoff_offset == 0
    assert c.descriptive


def test_m110_self_excluded_in_neighbour_kernel():
    """Self must never be in its own neighbourhood (``dist[i]=inf``)."""
    _neighbors = _local_neighbors()
    U = np.array(
        [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]
    )
    valid = np.ones(6, dtype=bool)
    for i in range(6):
        nbrs = _neighbors(U, valid, 3, i)
        assert i not in nbrs, f"self index {i} included in its own neighbourhood"


def test_m110_local_linear_self_not_in_fit():
    """If the query row were in its own fit, the residual would be exactly zero
    (y_i in the fit set).  With self excluded the residual is generically
    non-zero, and the kernel never emits a self-collapse row."""
    op = _get("cs_knn_local_linear_residual")
    rng = np.random.default_rng(13)
    n = 25
    tgt = pd.DataFrame(rng.standard_normal((60, n)), columns=[f"S{i}" for i in range(n)])
    f1 = pd.DataFrame(rng.standard_normal((60, n)), columns=[f"S{i}" for i in range(n)])
    f2 = pd.DataFrame(rng.standard_normal((60, n)), columns=[f"S{i}" for i in range(n)])
    f3 = pd.DataFrame(rng.standard_normal((60, n)), columns=[f"S{i}" for i in range(n)])
    out = op.calculate(tgt, f1, f2, f3, k=20, ridge=1e-3)
    v = out.to_numpy(dtype=float)
    fin = np.isfinite(v)
    assert fin.any(), "no finite residual — test fixture is degenerate"
    # a self-inclusive fit would give an exact 0 residual (perfect self-fit);
    # assert the finite residuals are not all mechanically zero.
    assert np.abs(v[fin]).max() > 1e-12


# ---------------------------------------------------------------------------
# M-111: target availability / VWAP decision clock + missing-target peers
# ---------------------------------------------------------------------------

def test_m111_vwap_decision_clock_documented():
    desc = _get("cs_knn_local_linear_residual").metadata.description
    assert "不可回测同日 VWAP" in desc, desc
    assert "prior-label" in desc, desc
    # dynamic_knn peer-mean also uses same-day peer target_t
    pm_desc = _get("cs_knn_peer_mean_ex_self").metadata.description
    assert "不可回测同日 VWAP" in pm_desc, pm_desc
    assert "prior-label" in pm_desc, pm_desc


def test_m111_neighbors_exclude_missing_target_peers():
    """A peer with a missing date-t target is excluded by the ``peer_mask``
    (feature-valid AND target-observed, R4-82)."""
    _neighbors = _local_neighbors()
    U = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.5, 0.0], [0.6, 0.0]])
    valid = np.ones(5, dtype=bool)
    # stock 1 is the nearest peer to stock 0 but its target is missing today
    peer_mask = np.array([True, False, True, True, True])
    nbrs = _neighbors(U, valid, 2, 0, peer_mask)
    assert 0 not in nbrs, "self must be excluded"
    assert 1 not in nbrs, "missing-target peer must be excluded"
    assert 2 in nbrs, "finite-target peer must be selectable"


def test_m111_missing_target_peer_excluded_from_fit():
    """End-to-end: a NaN-target peer cannot enter the local ridge fit; the
    kernel output for a query equals a manual fit computed on the same
    peer_mask-excluded neighbourhood."""
    from factor_engine.cleaned_operators.cross_section_local import _local_linear_series, _neighbors, _rank_features

    rng = np.random.default_rng(9)
    rows, n, d = 1, 30, 3
    feats = rng.standard_normal((rows, n, d))
    target = rng.standard_normal((rows, n))
    target[0, 2] = np.nan  # stock 2 target missing on the only date
    # make stock 2 feature-adjacent to stock 0 so it WOULD be selected were its
    # target observed — proving the mask is what keeps it out
    feats[0, 2] = feats[0, 0] + 0.001
    k, ridge = 20, 1e-3
    out = _local_linear_series(target, feats, k, ridge)
    assert np.isfinite(out[0, 0]), "query with enough valid peers must be finite"
    # manual replication of the kernel for stock 0
    min_neigh = max(10, 5 * (d + 1))
    target_k = max(min_neigh, int(k))
    U, valid = _rank_features(feats, 0)
    y_t = target[0]
    peer_mask = valid & np.isfinite(y_t)
    nbrs = _neighbors(U, valid, target_k, 0, peer_mask)
    assert 2 not in nbrs, "missing-target peer leaked into the fit"
    Z = U[nbrs]
    y = y_t[nbrs]
    Xd = np.column_stack([np.ones(len(y)), Z])
    pen = np.zeros((d + 1, d + 1))
    pen[1:, 1:] = float(ridge) * np.eye(d)
    beta = np.linalg.solve(Xd.T @ Xd + pen, Xd.T @ y)
    resid = y - Xd @ beta
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    scale = (1.4826 * mad) if mad > 1e-12 else float(np.std(resid))
    yhat = float(beta[0] + np.dot(U[0], beta[1:]))
    manual = float((y_t[0] - yhat) / scale)
    assert out[0, 0] == pytest.approx(manual, rel=1e-8)


# ---------------------------------------------------------------------------
# M-112: exactly 3 features (docs, not the API)
# ---------------------------------------------------------------------------

def test_m112_docs_exactly_three_features():
    for name in ALL_KNN_OPS:
        desc = _get(name).metadata.description
        assert "2..4" not in desc, f"{name} still claims 2..4 features"
    import factor_engine.cleaned_operators.dynamic_knn as dk

    assert "2..4" not in dk.__doc__
    assert "exactly 3" in dk.__doc__ or "exactly 3:" in dk.__doc__, dk.__doc__[:400]
    assert "3 个" in _get("cs_knn_peer_mean_ex_self").metadata.description


# ---------------------------------------------------------------------------
# M-113 / M-240: strict operator-boundary k / lag validation + documented cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,has_target",
    [
        ("cs_knn_peer_mean_ex_self", True),
        ("cs_knn_graph_dirichlet_energy", True),
    ],
)
def test_m113_dynamic_knn_k_rejected_invalid(name, has_target):
    op = _get(name)
    t = _panel(40, 3)
    for bad in (0, -1, 1.5, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            op.calculate(t, t, t, t, k=bad)


def test_m113_retention_k_and_lag_rejected_invalid():
    op = _get("cs_knn_neighbor_retention")
    t = _panel(40, 3)
    for bad in (0, -1, 1.5, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            op.calculate(t, t, t, k=bad)
        with pytest.raises(ValueError):
            op.calculate(t, t, t, k=5, lag=bad)


def test_m240_retention_lag_zero_rejected_not_clamped():
    """M-240: ``lag=0`` was silently clamped to 1 by ``max(1, int(lag))``; the
    operator boundary now rejects it instead of manufacturing a false search
    value."""
    op = _get("cs_knn_neighbor_retention")
    t = _panel(40, 3)
    with pytest.raises(ValueError):
        op.calculate(t, t, t, k=5, lag=0)
    # and the fractional-lag silent truncation is gone
    with pytest.raises(ValueError):
        op.calculate(t, t, t, k=5, lag=1.9)


@pytest.mark.parametrize(
    "name,has_target",
    [
        ("cs_knn_local_linear_residual", True),
        ("cs_knn_local_gradient_norm", True),
        ("cs_knn_tangent_residual", False),
    ],
)
def test_m113_local_ops_k_rejected_below_floor_and_invalid(name, has_target):
    op = _get(name)
    t = _panel(40, 25)
    args = (t, t, t, t) if has_target else (t, t, t)
    # k below the DOF floor -> audited _KNN_MIN_K raise
    with pytest.raises(ValueError):
        op.calculate(*args, k=19)
    # non-integer / bool / NaN / Inf -> strict contract error
    for bad in (19.5, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            op.calculate(*args, k=bad)


def test_m113_peer_count_cap_documented():
    """The internal ``peer_count-1`` breadth cap is legitimate but must be
    documented and exposed, not silent (M-113)."""
    from factor_engine.cleaned_operators.cross_section_local import _neighbors

    assert "peer_count - 1" in (_neighbors.__doc__ or "")
    desc = _get("cs_knn_local_linear_residual").metadata.description
    assert "peer_count-1" in desc, desc


# ---------------------------------------------------------------------------
# M-114: tie-inclusive kth-radius documented
# ---------------------------------------------------------------------------

def test_m114_tie_inclusive_documented():
    for name in ALL_KNN_OPS:
        desc = _get(name).metadata.description
        assert "tie-inclusive" in desc or "kth-distance" in desc, f"{name}: {desc}"
        assert "k 是最小邻居数" in desc or "kth-distance radius" in desc, f"{name}: {desc}"


# ---------------------------------------------------------------------------
# M-115: dynamic_knn declares estimator-resolution ParamSpecs (searchable=False)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,params",
    [
        ("cs_knn_peer_mean_ex_self", ("k",)),
        ("cs_knn_graph_dirichlet_energy", ("k",)),
        ("cs_knn_neighbor_retention", ("k", "lag")),
    ],
)
def test_m115_dynamic_knn_param_specs_declared(name, params):
    md = _get(name).metadata
    specs = md.param_specs
    for p in params:
        spec = specs.get(p)
        assert spec is not None, f"{name} missing ParamSpec for {p!r}"
        assert spec.dtype is int, f"{name}.{p} dtype must be int"
        assert spec.min == 1, f"{name}.{p} min must be 1"
        assert spec.searchable is False, f"{name}.{p} must be searchable=False"
        assert spec.param_role == ParamRole.ESTIMATOR_RESOLUTION, f"{name}.{p} role"


def test_m115_dynamic_knn_estimator_params_excluded_from_search():
    for name in DYNAMIC_KNN_OPS:
        grades = searchable_param_names(_get(name).metadata)
        for p in ("k", "lag"):
            assert p not in grades["full"], f"{name}.{p} leaked into full search"
            assert p not in grades["coarse"], f"{name}.{p} leaked into coarse search"


# ---------------------------------------------------------------------------
# M-116: as-of universe / tradable mask is a caller responsibility
# ---------------------------------------------------------------------------

def test_m116_universe_note_documented():
    import factor_engine.cleaned_operators.dynamic_knn as dk

    assert "universe" in dk.__doc__ and "tradable" in dk.__doc__, dk.__doc__[:500]
    desc = _get("cs_knn_local_linear_residual").metadata.description
    assert "universe" in desc and "tradable" in desc, desc
    # the operator does NOT invent a universe parameter
    md = _get("cs_knn_local_linear_residual").metadata
    assert "universe" not in md.param_names


# ---------------------------------------------------------------------------
# sanity: valid defaults still run (ParamSpec additions did not break calls)
# ---------------------------------------------------------------------------

def test_dynamic_knn_valid_k_still_runs():
    t = _panel(40, 3)
    op = _get("cs_knn_peer_mean_ex_self")
    out = op.calculate(t, t, t, t, k=2)
    assert out.shape == t.shape
    op2 = _get("cs_knn_neighbor_retention")
    out2 = op2.calculate(t, t, t, k=2, lag=1)
    assert out2.shape == t.shape
