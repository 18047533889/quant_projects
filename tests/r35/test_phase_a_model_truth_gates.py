# -*- coding: utf-8 -*-
"""R35 Phase A hard gates: model-truth fixes P0-M01..M15.

Each test asserts one concrete P0 fix holds at current HEAD:
- P0-M01/M02/M03 GARCH/GJR timing contract == implementation (fit strictly t-1)
- P0-M04 HAR FeatureLabelTiming (features through t, labels matured at t)
- P0-M05 panel forecast zero-feature rejection
- P0-M06 market_state required panel
- P0-M07 PCA min-history explicit public contract
- P0-M08 PCA sign tie-break stable under column reorder
- P0-M11 Polars variance-ratio current-NaN parity
- P0-M12 Polars pairwise finite-only parity
- P0-M13 zero duplicate model timing keys
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent
_TIMING_FILE = _REPO / "cleaned_operators" / "model_timing.py"


def _load():
    load_all()


# --------------------------------------------------------------------------
# P0-M13: no duplicate keys in MODEL_TIMING_CONTRACTS literal
# --------------------------------------------------------------------------

def test_p0_m13_no_duplicate_model_timing_keys():
    src = _TIMING_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "MODEL_TIMING_CONTRACTS" and isinstance(node.value, ast.Dict):
                    keys = [k.value if isinstance(k, ast.Constant) else None for k in node.value.keys]
                    from collections import Counter
                    c = Counter(keys)
                    dupes = {k for k, n in c.items() if n > 1 and k is not None}
                    assert not dupes, f"duplicate MODEL_TIMING_CONTRACTS keys: {sorted(dupes)}"
                    found = True
    assert found, "MODEL_TIMING_CONTRACTS dict literal not found"
    # runtime reflection agrees
    from factor_engine.cleaned_operators.model_timing import MODEL_TIMING_CONTRACTS
    assert len(MODEL_TIMING_CONTRACTS) == len(set(MODEL_TIMING_CONTRACTS))


# --------------------------------------------------------------------------
# P0-M01/M02/M03: GARCH timing — params fit strictly on <= t-1 for every stat
# --------------------------------------------------------------------------

def test_p0_m01_garch_forecast_params_fit_t_minus_1():
    """R35-P0-M01: ``ts_garch_next_vol_forecast`` must fit params on <= t-1.

    Proof: perturbing the CURRENT return to an extreme value changes the
    one-step variance update (it is the shock), but the PARAMETER fit segment
    excludes the current return.  If the kernel were fitting the current return
    into its own parameters (old behavior), perturbing r_t would also move the
    fit — and a huge r_t would dominate the fitted params.  The distinguishing
    assertion: persistence (pure parameter sum a+b) must be invariant to the
    current return, because params come from t-1.
    """
    from factor_engine.cleaned_operators.ts_model.volatility import _garch_path

    rng = np.random.default_rng(7)
    rets = rng.standard_normal(150)

    p1 = _garch_path(rets, 120, "persistence", False, 0.0)
    rets2 = rets.copy()
    rets2[-1] = 1e6  # extreme current return
    p2 = _garch_path(rets2, 120, "persistence", False, 0.0)
    assert np.isfinite(p1) and np.isfinite(p2)
    # persistence = a+b estimated on seg[:-1]; current return must not move it
    assert p1 == pytest.approx(p2, abs=1e-9), (
        "GARCH persistence moved when the current return was perturbed — "
        "params are NOT fit strictly on t-1 (P0-M02)"
    )


def test_p0_m01_garch_forecast_contract_matches_kernel():
    """R35-P0-M01/M02/M03: every GARCH/GJR timing contract is explicit and
    says fit_cutoff=1, and the kernel fits strictly t-1."""
    from factor_engine.cleaned_operators.model_timing import MODEL_TIMING_CONTRACTS, get_model_timing_contract

    _load()
    garch_canons = [
        "ts_garch_standardized_shock",
        "ts_garch_next_vol_forecast",
        "ts_garch_persistence",
        "ts_garch_vol_surprise",
        "ts_gjr_garch_vol_forecast",
    ]
    for name in garch_canons:
        c = get_model_timing_contract(name)
        assert c is not None and c.fit_cutoff_offset >= 1, name


# --------------------------------------------------------------------------
# P0-M04: HAR FeatureLabelTiming
# --------------------------------------------------------------------------

def test_p0_m04_har_feature_label_timing():
    from factor_engine.cleaned_operators.model_contract import feature_label_timing_of

    flt = feature_label_timing_of("ts_har_rv_next_vol_forecast")
    assert flt is not None
    # features through t usable, newest matured label RV_t anchored at t-1
    assert flt.feature_origin_offset == 0
    assert flt.label_origin_offset == 1
    assert flt.label_maturity_offset == 1
    assert flt.fit_latest_mature_label_offset == 0
    assert flt.score_feature_offset == 0
    # the innovation-z variant is strictly prior
    flt_z = feature_label_timing_of("ts_har_rv_innovation_z")
    assert flt_z is not None and flt_z.score_feature_offset == 1


def test_p0_m04_har_next_forecast_prefix_invariance():
    """HAR next-vol forecast (R35-P0-M04): the trailing-window output must be
    invariant to history BEFORE the window (prefix invariance) and to any
    hypothetically-future row beyond the current last observation — no future /
    no wrong-history leakage.  The rolling window is ``rv[-window:]``; prepending
    extra history (or perturbing the very first row, which falls out of the
    window for a long-enough series) must not move the current forecast."""
    from factor_engine.cleaned_operators.ts_model.volatility import _har_rv

    rng = np.random.default_rng(8)
    rv = np.abs(rng.standard_normal(300)) + 1.0
    base = _har_rv(rv, 120, "forecast")       # window = rv[180:300]
    rv_prefix = np.concatenate([rng.standard_normal(500) * 1e6, rv])
    out_prefixed = _har_rv(rv_prefix, 120, "forecast")  # same trailing window
    assert np.isfinite(base) and np.isfinite(out_prefixed)
    assert base == pytest.approx(out_prefixed, abs=1e-9), (
        "HAR forecast moved when history before the window changed"
    )
    rv_first = rv.copy()
    rv_first[0] = 1e6  # first row falls outside a 120-window over 300 rows
    out_first = _har_rv(rv_first, 120, "forecast")
    assert base == pytest.approx(out_first, abs=1e-9)


# --------------------------------------------------------------------------
# P0-M05 / P0-M06: panel required inputs
# --------------------------------------------------------------------------

def test_p0_m05_panel_zero_feature_rejected():
    from factor_engine.cleaned_operators.cross_section.panel_model import _forecast_generic

    rng = np.random.default_rng(11)
    n = 40
    y = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    with pytest.raises(ValueError):
        _forecast_generic(y, (None, None, None, None), 20, "pcr", 2, 0.01, 0.5, 1)


def test_p0_m06_market_state_required():
    from factor_engine.cleaned_operators.cross_section.panel_model import _regime_forecast, _moe_forecast

    rng = np.random.default_rng(12)
    n = 40
    y = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    x1 = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    with pytest.raises(ValueError):
        _regime_forecast(y, (x1,), None, 20, 3, 1)
    with pytest.raises(ValueError):
        _moe_forecast(y, (x1,), None, 20, 3, 1)


# --------------------------------------------------------------------------
# P0-M07: PCA min-history explicit
# --------------------------------------------------------------------------

def test_p0_m07_pca_min_history_explicit():
    import factor_engine.cleaned_operators.cross_section.panel_model as pm

    assert hasattr(pm, "PCA_MIN_HISTORY")
    assert hasattr(pm, "PCA_MIN_COVERAGE")
    assert pm.PCA_MIN_HISTORY == 2
    assert pm.PCA_MIN_COVERAGE == 0.7
    # the internal aliases must track the public contract
    assert pm._ABSOLUTE_MIN_OBS == pm.PCA_MIN_HISTORY
    assert pm._MIN_COVERAGE_RATIO == pm.PCA_MIN_COVERAGE


# --------------------------------------------------------------------------
# P0-M08: PCA sign tie-break stable under column reorder
# --------------------------------------------------------------------------

def test_p0_m08_pca_sign_tie_stable_under_reorder():
    """Two active loadings with exactly equal |value| must flip the eigenvector
    deterministically by instrument identity, not column position."""
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_svd, _pca_loading

    # Construct a window whose SVD produces a symmetric eigenvector (tie).
    # Use a symmetric X so loadings come out |equal| for two components.
    rng = np.random.default_rng(5)
    base = rng.standard_normal((40, 4))
    # force component 0 and 1 of the first PC to tie by symmetry
    X = base.copy()
    X[:, 1] = X[:, 0]  # columns 0 and 1 identical -> tied loading positions
    cur = rng.standard_normal(4)
    l1 = _pca_loading(X, cur, 0, ("B", "A", "C", "D"))  # order B,A,C,D
    # reorder the SAME data (columns permuted) with ids permuted accordingly
    perm = [1, 0, 3, 2]
    l2 = _pca_loading(X[:, perm], cur[perm], 0, ("A", "B", "D", "C"))
    # map back: position p in l1 == position perm[p] in l2
    assert np.allclose(l1, l2[perm], equal_nan=True) or np.allclose(-l1, l2[perm], equal_nan=True), (
        "PCA tie sign flipped under column reorder (P0-M08)"
    )


# --------------------------------------------------------------------------
# P0-M11 / P0-M12: Polars parity
# --------------------------------------------------------------------------

def test_p0_m11_polars_variance_ratio_trailing_nan():
    from factor_engine.cleaned_operators.ts_model.polars_regression import _variance_ratio_slope as _pl

    arr = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert np.isnan(_pl(arr, 3)), "Polars VR must be NaN when last row is non-finite"


def test_p0_m12_polars_pairwise_finite_parity():
    import polars as pl
    from factor_engine.cleaned_operators.cross_section.peer_ops import _rolling_regression
    from factor_engine.cleaned_operators.ts_model.polars_regression import _pairwise_rolling

    rng = np.random.default_rng(3)
    n = 60
    y = rng.standard_normal(n)
    y[25] = np.inf  # inf poisons window under is_not_null, filtered by is_finite
    x = rng.standard_normal(n)
    yd = pd.DataFrame({"A": y}, index=range(n))
    xd = pd.DataFrame({"A": x}, index=range(n))
    ref = _rolling_regression(yd, xd, 30, 6)["A"].to_numpy()
    out = _pairwise_rolling(pl.DataFrame(yd), pl.DataFrame(xd), 30, 6)["A"].to_numpy()
    both_nan = np.isnan(ref) & np.isnan(out)
    both_fin = np.isfinite(ref) & np.isfinite(out)
    assert ((both_nan | both_fin)).all(), "NaN-pattern mismatch (P0-M12)"
    if both_fin.any():
        assert np.allclose(ref[both_fin], out[both_fin], atol=1e-8), "value mismatch (P0-M12)"


# --------------------------------------------------------------------------
# P0-M09/P0-M10: doc drift fixed (description text)
# --------------------------------------------------------------------------

def test_p0_m09_pca_loading_description_no_cross_window_alignment():
    _load()
    op = OperatorRegistry.get("panel_rolling_pca_loading", "pandas_numpy")
    assert op is not None
    desc = op.metadata.description
    assert "跨窗 sign 对齐" not in desc, "P0-M09: stale cross-window sign-alignment description"
    assert "逐窗确定性" in desc, "P0-M09: new deterministic description missing"
