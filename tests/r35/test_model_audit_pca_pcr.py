# -*- coding: utf-8 -*-
"""M-020..M-036 model-audit regression tests for the panel PCA / PCR family.

Covers (see FactorEngine_Model_Operators_Full_Audit_and_Remediation_20260811.md):
- M-020: PCA full-history floor (no silent expanding warmup)
- M-021: reconstruction vs regression rank policy split
- M-022: PCR with a single feature degrades to standardized linear regression
- M-026: industry PCA fits once per (date, unique industry)
- M-027: PCA downstream rolling warmup two-stage maturity
- M-030: panel operators are documented as per-symbol (not pooled)
- M-033: regime / MoE require at least one true predictor
- M-035: regime description documents current-state routing
- M-036: regime / MoE fit-quality telemetry accessor
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
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _load():
    load_all()


def _panel(n: int, cols: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, cols)), columns=list("ABCD")[:cols])


# --------------------------------------------------------------------------
# M-020: PCA full-history floor
# --------------------------------------------------------------------------

def test_m020_rolling_pca_full_history_floor():
    """window=120 -> rows with fit_end < 119 are NaN; row 120 (fit_end=119) is
    the first eligible.  ``warmup_policy="expanding"`` opts in to early output."""
    from factor_engine.cleaned_operators.cross_section.panel_model import _rolling_pca

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((200, 5)), columns=list("ABCDE"))
    out = _rolling_pca(X, 120, lambda w, c: np.ones(5))
    arr = out.to_numpy()
    # fit_end = row - 1 ; floor is fit_end < window-1 == 119  ->  row < 120
    assert np.isnan(arr[:120]).all(), "rows 0..119 must be NaN (full-history floor)"
    assert np.isfinite(arr[120]).all(), "row 120 (fit_end=119) is first eligible"
    # expanding warmup explicitly opts in to partial-window fits
    out_exp = _rolling_pca(X, 120, lambda w, c: np.ones(5), warmup_policy="expanding")
    assert np.isfinite(out_exp.to_numpy()[60]).all(), "expanding warmup gave no early output"


def test_m020_pca_resid_operator_full_history_floor():
    """The public panel_rolling_pca_resid operator honours the floor: first
    finite row at index window (fit_end = window-1), all prior rows NaN."""
    _load()
    op = OperatorRegistry.get("panel_rolling_pca_resid", "pandas_numpy")
    assert op is not None
    rng = np.random.default_rng(2)
    ret = pd.DataFrame(rng.standard_normal((170, 5)), columns=list("ABCDE"))
    out = op.calculate(ret, window=120, n_components=2)
    arr = out.to_numpy()
    assert np.isnan(arr[:120]).all()
    assert np.isfinite(arr[120]).all()


# --------------------------------------------------------------------------
# M-021: reconstruction vs regression rank policy
# --------------------------------------------------------------------------

def test_m021_rank_policy_split():
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_svd
    from factor_engine.cleaned_operators.cross_section.pca_state import _fit

    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 5))
    rec = _pca_svd(X, 5)                      # default reconstruction
    reg = _pca_svd(X, 5, rank_policy="regression")
    assert rec is not None and reg is not None
    assert rec["k"] == 4, "reconstruction must keep k <= p-1 (avoid full-rank residual=0)"
    assert reg["k"] == 5, "regression must allow k <= p (all components)"
    # the shared authoritative state honours the same split
    s_rec = _fit(X, 5)
    s_reg = _fit(X, 5, rank_policy="regression")
    assert s_rec["k"] == 4 and s_reg["k"] == 5


def test_m021_pcr_uses_regression_policy_full_rank():
    """PCR with n_components == p (full rank) must equal OLS on the
    standardized features — this only holds if the PCR path allows k == p."""
    from factor_engine.cleaned_operators.cross_section.panel_model import _model_predict

    rng = np.random.default_rng(4)
    n = 80
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
    x_cur = rng.standard_normal(2)
    pcr = _model_predict(X, y, x_cur, "pcr", 2, 0.01, 0.5)
    mu = X.mean(axis=0)
    sd = np.std(X, axis=0)
    Xs = (X - mu) / sd
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), Xs]), y, rcond=None)
    ref = beta[0] + beta[1:] @ ((x_cur - mu) / sd)
    assert np.isfinite(pcr) and np.isfinite(ref)
    assert pcr == pytest.approx(ref, abs=1e-6), (
        "full-rank PCR must equal OLS on standardized features (regression policy)"
    )


# --------------------------------------------------------------------------
# M-022: PCR single-feature degradation
# --------------------------------------------------------------------------

def test_m022_pcr_single_feature_finite_for_mature_rows():
    from factor_engine.cleaned_operators.cross_section.panel_model import _forecast_generic

    rng = np.random.default_rng(5)
    n = 140
    y = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    x1 = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    out = _forecast_generic(y, (x1, None, None, None), 80, "pcr", 3, 0.01, 0.5, 1)
    v = out.to_numpy()
    fin = np.where(np.isfinite(v))[0]
    assert fin.size >= 20, f"p=1 PCR must produce finite mature rows (got {fin.size})"
    assert np.isfinite(v[np.isfinite(v)]).all()
    # the last row (fully-matured training window) must be finite
    assert np.isfinite(v[-1]).any(), "last row should have a mature finite forecast"


def test_m022_pcr_single_feature_equals_ols():
    """p=1 PCR degrades to a legal standardized linear regression."""
    from factor_engine.cleaned_operators.cross_section.panel_model import _model_predict

    rng = np.random.default_rng(6)
    n = 60
    X = rng.standard_normal((n, 1))
    y = 1.5 * X[:, 0] + 0.1 * rng.standard_normal(n)
    x_cur = rng.standard_normal(1)
    pcr = _model_predict(X, y, x_cur, "pcr", 1, 0.01, 0.5)
    mu = X.mean(axis=0)
    sd = np.std(X, axis=0)
    Xs = (X - mu) / sd
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), Xs]), y, rcond=None)
    ref = beta[0] + beta[1:] @ ((x_cur - mu) / sd)
    assert np.isfinite(pcr) and np.isfinite(ref)
    assert pcr == pytest.approx(ref, abs=1e-8)


# --------------------------------------------------------------------------
# M-026: industry PCA fit once per (date, unique industry)
# --------------------------------------------------------------------------

def _naive_industry_pca_loading(ret, group, window, component):
    """Reference per-(row, col) implementation (pre-M-026 behaviour)."""
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_loading

    rv = ret.to_numpy(dtype=float)
    gv = group.to_numpy()
    col_ids = tuple(ret.columns)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        fit_end = row - 1
        if fit_end < window - 1:
            continue
        start = max(0, fit_end - window + 1)
        for col in range(cols):
            g = gv[row, col]
            mask = gv[row] == g
            members = np.flatnonzero(mask)
            if len(members) < 4:
                continue
            local = int(np.where(members == col)[0][0])
            X = rv[start:fit_end + 1][:, members]
            member_ids = tuple(col_ids[int(i)] for i in members)
            loading = _pca_loading(X, rv[row][members], int(component), member_ids)
            out[row, col] = loading[local] if np.isfinite(loading).any() else np.nan
    return out


def test_m026_industry_pca_fit_once_per_date_industry():
    """The M-026 cache must fit the SVD exactly once per (date, industry) —
    not once per (row, col)."""
    import factor_engine.cleaned_operators.cross_section.panel_model as pm

    rng = np.random.default_rng(7)
    rows, cols = 50, 8
    ret = pd.DataFrame(
        rng.standard_normal((rows, cols)), columns=[f"c{i}" for i in range(cols)]
    )
    grp = pd.DataFrame(
        [["G"] * 4 + ["X"] * 4] * rows,
        columns=[f"c{i}" for i in range(cols)], dtype=object,
    )
    calls = {"n": 0}
    orig = pm._pca_loading

    def counting(X, cur, component, ids=None):
        calls["n"] += 1
        return orig(X, cur, component, ids)

    pm._pca_loading = counting
    try:
        pm._industry_pca_loading(ret, grp, 30, 0)
    finally:
        pm._pca_loading = orig
    # rows 30..49 (20 dates with a full 30-row window) x 2 industries
    assert calls["n"] == 20 * 2, f"expected 40 fits, got {calls['n']}"


def test_m026_industry_pca_cached_matches_naive():
    """The cached (date x industry) broadcast must be bit-identical to the old
    per-(row, col) refit."""
    import factor_engine.cleaned_operators.cross_section.panel_model as pm

    rng = np.random.default_rng(8)
    rows, cols = 50, 8
    ret = pd.DataFrame(
        rng.standard_normal((rows, cols)), columns=[f"c{i}" for i in range(cols)]
    )
    grp = pd.DataFrame(
        [["G"] * 4 + ["X"] * 4] * rows,
        columns=[f"c{i}" for i in range(cols)], dtype=object,
    )
    cached = pm._industry_pca_loading(ret, grp, 30, 0).to_numpy()
    naive = _naive_industry_pca_loading(ret, grp, 30, 0)
    np.testing.assert_array_equal(cached, naive)


# --------------------------------------------------------------------------
# M-027: PCA downstream rolling two-stage maturity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn_name", ["_pca_resid_vol", "_pca_resid_momentum"])
def test_m027_downstream_first_output_timing(fn_name):
    """First output of the downstream rolling stat must be at
    (PCA full-history floor) + (downstream min_periods=10), not a short-window
    stat from the first few residual observations."""
    import factor_engine.cleaned_operators.cross_section.panel_model as pm

    rng = np.random.default_rng(9)
    n = 300
    ret = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("ABCDE"))
    window = 100
    fn = getattr(pm, fn_name)
    out = fn(ret, window, 2).to_numpy()
    first = np.where(np.isfinite(out).any(axis=1))[0]
    assert first.size > 0, f"{fn_name} produced no finite output"
    # PCA full-history floor: first residual at row `window` (fit_end=window-1).
    # downstream min_periods=10 residual observations -> row (window-1)+10.
    assert first[0] == (window - 1) + 10, (
        f"{fn_name} first output at {first[0]}, expected {(window - 1) + 10}"
    )


def test_m027_resid_vol_description_documents_two_stage():
    _load()
    for name in ("panel_rolling_pca_resid_vol", "panel_rolling_pca_resid_momentum"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None
        desc = op.metadata.description
        assert "两级成熟" in desc, name
        assert "window-1)+10" in desc, name


# --------------------------------------------------------------------------
# M-030: panel operators are per-symbol (not pooled)
# --------------------------------------------------------------------------

def test_m030_module_docstring_is_per_symbol():
    import factor_engine.cleaned_operators.cross_section.panel_model as pm

    assert "per-symbol" in pm.__doc__ and "not pooled" in pm.__doc__


def test_m030_supervised_operator_descriptions_are_per_symbol():
    _load()
    for name in (
        "panel_rolling_pcr_forecast",
        "panel_rolling_pls_forecast",
        "panel_rolling_elastic_net_forecast",
        "panel_regime_conditioned_forecast",
        "panel_mixture_of_experts_score",
    ):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None
        assert "逐股票独立滚动时序模型，非 pooled stock×date 模型" in op.metadata.description, name


# --------------------------------------------------------------------------
# M-033: regime / MoE require at least one true predictor
# --------------------------------------------------------------------------

def test_m033_regime_moe_require_true_predictor():
    from factor_engine.cleaned_operators.cross_section.panel_model import _moe_forecast, _regime_forecast

    rng = np.random.default_rng(10)
    n = 40
    y = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    ms = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("ABC"))
    # market_state alone does NOT count as a predictor
    with pytest.raises(ValueError, match="at least one predictor"):
        _regime_forecast(y, (None, None, None, None), ms, 20, 3, 1)
    with pytest.raises(ValueError, match="at least one predictor"):
        _moe_forecast(y, (None, None, None, None), ms, 20, 3, 1)


# --------------------------------------------------------------------------
# M-035: regime description documents current-state routing
# --------------------------------------------------------------------------

def test_m035_regime_description_documents_current_state_routing():
    _load()
    op = OperatorRegistry.get("panel_regime_conditioned_forecast", "pandas_numpy")
    assert op is not None
    desc = op.metadata.description
    assert "market_state_t 选择当前 regime" in desc
    assert "严格截至 t-1" in desc


# --------------------------------------------------------------------------
# M-036: regime / MoE fit-quality telemetry
# --------------------------------------------------------------------------

def test_m036_regime_telemetry():
    from factor_engine.cleaned_operators.cross_section.panel_model import _regime_forecast, last_fit_telemetry

    rng = np.random.default_rng(11)
    n = 140
    y = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    x1 = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    x2 = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    ms = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    _regime_forecast(y, (x1, x2), ms, 60, 3, 1)
    t = last_fit_telemetry()
    assert t.get("model") == "regime"
    for key in (
        "effective_train_obs", "effective_regime_obs",
        "condition_number", "convergence", "gate_entropy",
    ):
        assert key in t, f"regime telemetry missing {key}"
    assert isinstance(t["effective_train_obs"], int) and t["effective_train_obs"] >= 8
    assert isinstance(t["effective_regime_obs"], int)
    assert t["condition_number"] == np.inf or np.isfinite(t["condition_number"])
    assert t["gate_entropy"] >= 0.0


def test_m036_moe_telemetry():
    from factor_engine.cleaned_operators.cross_section.panel_model import _moe_forecast, last_fit_telemetry

    rng = np.random.default_rng(12)
    n = 140
    y = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    x1 = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    x2 = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    ms = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("ABCD"))
    _moe_forecast(y, (x1, x2), ms, 60, 3, 1)
    t = last_fit_telemetry()
    assert t.get("model") == "moe"
    for key in (
        "effective_train_obs", "active_expert_count",
        "condition_number", "convergence", "gate_entropy", "expert_weight_max",
    ):
        assert key in t, f"moe telemetry missing {key}"
    assert t["active_expert_count"] >= 1
    assert 0.0 <= t["expert_weight_max"] <= 1.0 + 1e-12
    assert 0.0 <= t["gate_entropy"]
    # telemetry is a defensive copy — mutating it must not corrupt the module
    last_fit_telemetry().clear()
    assert last_fit_telemetry()
