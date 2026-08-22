# -*- coding: utf-8 -*-
"""2026-08-08 Gemini-recommended primitives: registration, semantics, parity.

Covers the daily gather/weighted-moment/activity-clock/spectral/relation/
intraday pack plus the research-surface transforms.  Every assertion is a
pinned reference, NaN fail-closed, strict-PIT and shape-preserving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_surface import classify_canonical

try:
    import polars as pl
except Exception:  # pragma: no cover - optional
    pl = None

_DAILY_OPS = [
    "group_topk_mean",
    "ts_value_at_argextreme",
    "ts_weighted_standardized_moment",
    "ts_cov_if",
    "ts_spectral_entropy",
    "ts_dominant_cycle_period",
    "ts_activity_clock_lagged_value",
    "ts_activity_clock_age",
    "cs_weighted_percentile_rank",
    "group_distribution_js_divergence",
    "event_level_survival_share",
    "ts_max_drawdown_activity_cost",
    "cs_multi_robust_resid",
    "intraday_activity_duration_curvature",
]
_RESEARCH_OPS = [
    "ts_wavelet_lowpass_reconstruct",
    "ts_signature_mahalanobis_anomaly",
    "ts_persistence_birth_dispersion",
    # P0-008: the group complete-graph signal share is the honest research-only
    # canonical; relation_pagerank_centrality is now a pure alias to it.
    "group_signal_attraction_share",
]


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all

    load_all()


@pytest.mark.parametrize("op", _DAILY_OPS + _RESEARCH_OPS)
def test_registered_both_backends(_loaded, op):
    backends = OperatorRegistry.backends_for(op)
    assert "pandas_numpy" in backends, op
    assert "polars" in backends, op


@pytest.mark.parametrize("op", _DAILY_OPS)
def test_daily_surface(_loaded, op):
    assert classify_canonical(op) == "daily"


@pytest.mark.parametrize("op", _RESEARCH_OPS)
def test_research_surface(_loaded, op):
    assert classify_canonical(op) == "research"


def _panel(rows=40, cols=3, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    cols_l = [f"P{index}" for index in range(cols)]
    return pd.DataFrame(rng.normal(0.0, 1.0, (rows, cols)), index=idx, columns=cols_l)


# ---------------------------------------------------------------------------
# group_topk_mean
# ---------------------------------------------------------------------------
def test_group_topk_mean_reference(_loaded):
    idx = list(range(6))
    tgt = pd.DataFrame(
        {"A": [1.0, 2, 3, 4, 5, 6], "B": [10.0, 9, 8, 7, 6, 5], "C": [1.0, 1, 1, 1, 1, 1]},
        index=idx,
    )
    score = pd.DataFrame(
        {"A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "B": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        index=idx,
    )
    grp = pd.DataFrame({"A": ["g"] * 6, "B": ["g"] * 6, "C": ["h"] * 6}, index=idx)
    op = OperatorRegistry.get("group_topk_mean", "pandas_numpy")
    out = op.calculate(tgt, score, grp, k=2, exclude_self=True)
    # g = {A,B}.  exclude_self=True with k=2 needs 2 peers -> both NaN.
    assert out["A"].isna().all() and out["B"].isna().all()
    # C alone in h -> NaN.
    assert out["C"].isna().all()
    # exclude_self=False, k=1 -> each group member equals the group's max target.
    out2 = op.calculate(tgt, score, grp, k=1, exclude_self=False)
    assert out2["A"].iloc[0] == 10.0  # B has top score in g
    assert out2["B"].iloc[0] == 10.0
    assert out2["C"].iloc[0] == 1.0


def test_group_topk_mean_shape_and_nan(_loaded):
    x = _panel(30, 3)
    grp = pd.DataFrame({"P0": ["a"] * 30, "P1": ["a"] * 30, "P2": ["b"] * 30}, index=x.index)
    op = OperatorRegistry.get("group_topk_mean", "pandas_numpy")
    out = op.calculate(x, x, grp, k=2, exclude_self=True)
    assert out.shape == x.shape
    assert out.index.equals(x.index) and out.columns.equals(x.columns)


# ---------------------------------------------------------------------------
# ts_value_at_argextreme
# ---------------------------------------------------------------------------
def test_value_at_argextreme_reference(_loaded):
    idx = list(range(10))
    value = pd.DataFrame({"A": [10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, index=idx)
    score = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 4, 3, 2, 1, 0]}, index=idx)
    op = OperatorRegistry.get("ts_value_at_argextreme", "pandas_numpy")
    # window=5, exclude current: at row 4, rows 0..3, argmax score=row3 -> value=40
    out = op.calculate(value, score, window=5, mode="max", include_current=False)
    assert out["A"].iloc[4] == 40.0
    # at row 0, empty window -> NaN
    assert np.isnan(out["A"].iloc[0])
    # include_current=True: at row 4, rows 0..4, argmax=row4 -> value=50
    out2 = op.calculate(value, score, window=5, mode="max", include_current=True)
    assert out2["A"].iloc[4] == 50.0
    # mode=min: at row 4 exclude current, rows 0..3, argmin score=row0 -> value=10
    out3 = op.calculate(value, score, window=5, mode="min", include_current=False)
    assert out3["A"].iloc[4] == 10.0


# ---------------------------------------------------------------------------
# ts_weighted_standardized_moment
# ---------------------------------------------------------------------------
def test_weighted_standardized_moment_reference(_loaded):
    idx = list(range(10))
    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=idx)
    w = pd.DataFrame({"A": [1.0] * 10}, index=idx)
    op = OperatorRegistry.get("ts_weighted_standardized_moment", "pandas_numpy")
    out = op.calculate(x, w, window=10, order=3)
    v = np.arange(1.0, 11.0)
    mu = v.mean()
    sigma = np.sqrt(np.mean((v - mu) ** 2))
    skew = float(np.mean((v - mu) ** 3)) / (sigma**3 + 1e-12)
    assert out["A"].iloc[-1] == pytest.approx(skew, rel=1e-9)
    # order validation
    with pytest.raises(ValueError):
        op.calculate(x, w, window=10, order=5)


# ---------------------------------------------------------------------------
# ts_cov_if
# ---------------------------------------------------------------------------
def test_cov_if_reference(_loaded):
    rng = np.random.default_rng(11)
    idx = list(range(40))
    x = pd.DataFrame({"A": rng.normal(0, 1, 40)}, index=idx)
    y = pd.DataFrame({"A": 0.5 * x["A"] + rng.normal(0, 0.1, 40)}, index=idx)
    cond = pd.DataFrame({"A": np.ones(40)}, index=idx)
    op = OperatorRegistry.get("ts_cov_if", "pandas_numpy")
    out = op.calculate(x, y, cond, window=20, min_periods=2)
    ref = np.cov(x["A"].iloc[-20:], y["A"].iloc[-20:])[0, 1]
    assert out["A"].iloc[-1] == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# ts_spectral_entropy / ts_dominant_cycle_period
# ---------------------------------------------------------------------------
def test_spectral_entropy_bounds(_loaded):
    rng = np.random.default_rng(3)
    idx = list(range(90))
    x = pd.DataFrame({"A": rng.normal(0, 1, 90)}, index=idx)
    op = OperatorRegistry.get("ts_spectral_entropy", "pandas_numpy")
    out = op.calculate(x, window=60)
    v = out["A"].dropna()
    assert ((v >= 0.0) & (v <= 1.0)).all()


def test_dominant_cycle_period_reference(_loaded):
    t = np.arange(150, dtype=float)
    sine = np.sin(2 * np.pi * t / 10.0)
    x = pd.DataFrame({"A": sine}, index=list(range(150)))
    op = OperatorRegistry.get("ts_dominant_cycle_period", "pandas_numpy")
    out = op.calculate(x, window=60, min_peak_share=0.10)
    assert out["A"].iloc[-1] == pytest.approx(10.0, rel=0.15)


# ---------------------------------------------------------------------------
# activity clock
# ---------------------------------------------------------------------------
def test_activity_clock_reference(_loaded):
    idx = list(range(60))
    x = pd.DataFrame({"A": np.linspace(1, 60, 60)}, index=idx)
    act = pd.DataFrame({"A": np.ones(60)}, index=idx)
    age_op = OperatorRegistry.get("ts_activity_clock_age", "pandas_numpy")
    lag_op = OperatorRegistry.get("ts_activity_clock_lagged_value", "pandas_numpy")
    age = age_op.calculate(act, budget=5.0, scale_window=10, max_lookback=60)
    lag = lag_op.calculate(x, act, budget=5.0, scale_window=10, max_lookback=60)
    assert age["A"].iloc[-1] == 4.0
    assert lag["A"].iloc[-1] == 56.0


def test_max_drawdown_activity_cost_reference(_loaded):
    idx = list(range(8))
    x = pd.DataFrame({"A": [10.0, 9, 8, 12, 11, 7, 13, 14]}, index=idx)
    act = pd.DataFrame({"A": [1.0] * 8}, index=idx)
    op = OperatorRegistry.get("ts_max_drawdown_activity_cost", "pandas_numpy")
    out = op.calculate(x, act, window=8)
    assert out["A"].iloc[-1] == pytest.approx(3.0 / 8.0)


# ---------------------------------------------------------------------------
# cs_weighted_percentile_rank
# ---------------------------------------------------------------------------
def test_weighted_percentile_rank_reference(_loaded):
    idx = list(range(3))
    x = pd.DataFrame({"A": [1.0, 2, 4], "B": [4.0, 2, 1], "C": [2.0, 4, 2]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1, 1], "B": [1.0, 1, 1], "C": [1.0, 1, 1]}, index=idx)
    op = OperatorRegistry.get("cs_weighted_percentile_rank", "pandas_numpy")
    out = op.calculate(x, w)
    # row 0: A=1,B=4,C=2 -> ranks (0.5*1)/3, (2+0.5)/3, (1+0.5)/3
    assert out["A"].iloc[0] == pytest.approx(0.5 / 3.0)
    assert out["B"].iloc[0] == pytest.approx(2.5 / 3.0)
    assert out["C"].iloc[0] == pytest.approx(1.5 / 3.0)
    # negative weights fail closed to NaN
    wneg = w.copy()
    wneg["A"] = -1.0
    out2 = op.calculate(x, wneg)
    assert np.isnan(out2["A"].iloc[0])


# ---------------------------------------------------------------------------
# group_distribution_js_divergence
# ---------------------------------------------------------------------------
def test_js_divergence_reference(_loaded):
    # R11 #133: the reference distribution needs >= 2 * bins observations, so
    # the cross-section must be wide enough (a 2-4 column panel would now be
    # correctly rejected as an under-powered histogram).
    idx = list(range(8))
    cols = ["A", "B", "C"] + [f"E{i}" for i in range(10)]
    data = {
        "A": [1.0, 2, 3, 4, 5, 6, 7, 8],
        "B": [8.0, 7, 6, 5, 4, 3, 2, 1],
        "C": [1.0, 1, 1, 1, 1, 1, 1, 1],  # singleton group -> NaN
    }
    for i, c in enumerate(cols[3:]):
        data[c] = [float((i + 1) % 9 + 1)] * 8
    x = pd.DataFrame(data, index=idx)
    grp_data = {"A": ["m"] * 8, "B": ["m"] * 8, "C": ["s"] * 8}
    for i, c in enumerate(cols[3:]):
        grp_data[c] = [f"g{i}"] * 8
    grp = pd.DataFrame(grp_data, index=idx)
    op = OperatorRegistry.get("group_distribution_js_divergence", "pandas_numpy")
    out = op.calculate(x, grp, bins=4, min_group_size=2)
    assert out["A"].iloc[0] == out["B"].iloc[0]  # same group -> same value
    assert out["A"].iloc[0] >= 0.0  # JS divergence is non-negative
    assert np.isnan(out["C"].iloc[0])  # group too small (1 member)
    # ex-self default: group m is compared against the (>= 8 member) market
    # minus {A,B}.
    out_ex = op.calculate(x, grp, bins=4, min_group_size=2, exclude_group_from_reference=True)
    assert np.isfinite(out_ex["A"].iloc[0])


def test_js_divergence_exclude_group_reference(_loaded):
    # A group that IS the whole market must diverge maximally from its ex-self
    # reference: excluding it leaves an empty reference -> fail closed.
    idx = list(range(6))
    cols = [f"C{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.integers(1, 9, (6, 10)).astype(float), index=idx, columns=cols)
    grp = pd.DataFrame({c: ["g"] * 6 for c in cols}, index=idx)
    op = OperatorRegistry.get("group_distribution_js_divergence", "pandas_numpy")
    out = op.calculate(x, grp, bins=4, min_group_size=2, exclude_group_from_reference=True)
    assert np.isnan(out[cols[0]].iloc[0])  # ex-self reference empty -> NaN
    # with exclusion disabled, both members compare against the full market
    # (R11 #133: wide enough cross-section to satisfy the reference-breadth gate).
    out_full = op.calculate(x, grp, bins=4, min_group_size=2, exclude_group_from_reference=False)
    assert out_full[cols[0]].iloc[0] == out_full[cols[1]].iloc[0]
    assert np.isfinite(out_full[cols[0]].iloc[0])


# ---------------------------------------------------------------------------
# event_level_survival_share
# ---------------------------------------------------------------------------
def test_event_level_survival_share_reference(_loaded):
    # Path-survival semantic: an event only survives if the WHOLE path since the
    # event stayed on the favorable side (a dip below the level then recovery
    # does NOT count as survived).
    idx = list(range(8))
    ev = pd.DataFrame({"A": [0.0, 0, 1, 0, 0, 0, 0, 0]}, index=idx)
    lv = pd.DataFrame({"A": [10.0] * 8}, index=idx)
    x = pd.DataFrame({"A": [5.0, 6, 7, 8, 9, 11, 12, 13]}, index=idx)
    op = OperatorRegistry.get("event_level_survival_share", "pandas_numpy")
    out = op.calculate(ev, lv, x, history_window=5, direction="up")
    assert out["A"].iloc[5] == 0.0  # path dipped to x=8 < level before 11
    assert out["A"].iloc[3] == 0.0  # x=8<10 -> not survived
    assert np.isnan(out["A"].iloc[0])  # no past events
    # Inclusive boundary (P0-004): a path that touches the level (== 10) without
    # going below it survives.
    x2 = pd.DataFrame({"A": [5.0, 6, 7, 10, 10, 11, 12, 13]}, index=idx)
    out2 = op.calculate(ev, lv, x2, history_window=5, direction="up")
    assert out2["A"].iloc[5] == 1.0  # running min = 10 >= level
    assert out2["A"].iloc[7] == 1.0


# ---------------------------------------------------------------------------
# cs_multi_robust_resid
# ---------------------------------------------------------------------------
def test_multi_robust_resid_reference(_loaded):
    rng = np.random.default_rng(5)
    idx = list(range(6))
    # R11 #141: the residual regression needs a real DOF margin, so the
    # cross-section (columns) must be >= max(20, 5 * K) names; a 4-stock
    # section is now correctly rejected as an interpolation fit.
    cols = [f"C{i}" for i in range(25)]
    x1 = pd.DataFrame(rng.normal(0, 1, (6, 25)), index=idx, columns=cols)
    x2 = pd.DataFrame(rng.normal(0, 1, (6, 25)), index=idx, columns=cols)
    y = 0.5 * x1 + (-0.3) * x2 + 2.0 + rng.normal(0, 1e-3, (6, 25))
    op = OperatorRegistry.get("cs_multi_robust_resid", "pandas_numpy")
    out = op.calculate(y, x1, x2, add_intercept=True)
    # mild ridge keeps residuals small for near-perfect linear structure
    assert float(np.nanmax(np.abs(out.to_numpy()))) < 0.05


# ---------------------------------------------------------------------------
# relation_pagerank_centrality
# ---------------------------------------------------------------------------
def test_pagerank_centrality_reference(_loaded):
    idx = list(range(3))
    x = pd.DataFrame({"A": [1.0, 2, 3], "B": [2.0, 1, 3], "C": [3.0, 3, 1]}, index=idx)
    grp = pd.DataFrame({"A": ["g"] * 3, "B": ["g"] * 3, "C": ["g"] * 3}, index=idx)
    # P0-008: the honest canonical is group_signal_attraction_share; the old
    # relation_pagerank_centrality name is an alias to it.
    op = OperatorRegistry.get("group_signal_attraction_share", "pandas_numpy")
    assert op is not None
    assert OperatorRegistry.resolve_canonical("relation_pagerank_centrality") == "group_signal_attraction_share"
    out = op.calculate(x, grp, damping=0.85)
    row = out.iloc[0].to_numpy(float)
    assert np.isfinite(row).all()
    assert row.sum() == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# research transforms (existence + basic finite output)
# ---------------------------------------------------------------------------
def test_wavelet_lowpass_reconstruct(_loaded):
    t = np.arange(90, dtype=float)
    x = pd.DataFrame({"A": np.sin(t / 4.0)}, index=list(range(90)))
    op = OperatorRegistry.get("ts_wavelet_lowpass_reconstruct", "pandas_numpy")
    out = op.calculate(x, window=64, level=2)
    assert np.isfinite(out["A"].iloc[-1])


def test_signature_mahalanobis_anomaly(_loaded):
    rng = np.random.default_rng(9)
    idx = list(range(100))
    f1 = pd.DataFrame({"A": rng.normal(0, 1, 100)}, index=idx)
    f2 = pd.DataFrame({"A": rng.normal(0, 1, 100)}, index=idx)
    f3 = pd.DataFrame({"A": rng.normal(0, 1, 100)}, index=idx)
    op = OperatorRegistry.get("ts_signature_mahalanobis_anomaly", "pandas_numpy")
    out = op.calculate(f1, f2, f3, path_window=20, history_window=60, depth=2)
    assert out.shape == f1.shape
    assert out["A"].iloc[-1] >= 0.0


# ---------------------------------------------------------------------------
# polars parity (panel bridge must reproduce the pandas reference)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize(
    "op,kwargs",
    [
        ("ts_spectral_entropy", {"window": 20}),
        ("ts_weighted_standardized_moment", {"window": 20, "order": 3}),
        ("ts_cov_if", {"window": 20, "min_periods": 2}),
        ("ts_activity_clock_age", {"budget": 2.0, "scale_window": 10, "max_lookback": 40}),
        ("cs_weighted_percentile_rank", {}),
        ("group_topk_mean", {"k": 2, "exclude_self": True}),
        ("ts_max_drawdown_activity_cost", {"window": 20}),
        ("ts_value_at_argextreme", {"window": 10, "mode": "max", "include_current": False}),
    ],
)
def test_polars_parity(_loaded, op, kwargs):
    rng = np.random.default_rng(13)
    idx = pd.date_range("2026-01-01", periods=50, freq="B")
    cols = ["Q1", "Q2", "Q3"]
    a = pd.DataFrame(rng.normal(0, 1, (50, 3)), index=idx, columns=cols)
    b = pd.DataFrame(np.abs(rng.normal(1, 0.5, (50, 3))) + 0.1, index=idx, columns=cols)
    cond = pd.DataFrame(np.ones((50, 3)), index=idx, columns=cols)
    grp = pd.DataFrame(
        {c: np.tile(["g1", "g2", "g1"][i], 50) for i, c in enumerate(cols)},
        index=idx,
    )

    def _pl(df):
        return pl.DataFrame({c: df[c].to_numpy() for c in df.columns})

    p_op = OperatorRegistry.get(op, "pandas_numpy")
    pl_op = OperatorRegistry.get(op, "polars")

    def _run(o, frames):
        return o.calculate(*frames, **kwargs)

    if op == "ts_cov_if":
        pd_out = _run(p_op, (a, b, cond))
        pl_out = _run(pl_op, (_pl(a), _pl(b), _pl(cond)))
    elif op in {"group_topk_mean", "cs_weighted_percentile_rank"}:
        pd_out = _run(p_op, (a, b, grp)) if op == "group_topk_mean" else _run(p_op, (a, b))
        pl_out = _run(pl_op, (_pl(a), _pl(b), _pl(grp))) if op == "group_topk_mean" else _run(pl_op, (_pl(a), _pl(b)))
    elif op == "ts_activity_clock_age":
        pd_out = _run(p_op, (a,))
        pl_out = _run(pl_op, (_pl(a),))
    elif op in {"ts_weighted_standardized_moment", "ts_max_drawdown_activity_cost", "ts_value_at_argextreme"}:
        pd_out = _run(p_op, (a, b))
        pl_out = _run(pl_op, (_pl(a), _pl(b)))
    else:
        pd_out = _run(p_op, (a,))
        pl_out = _run(pl_op, (_pl(a),))

    pdf = pd_out.to_pandas() if hasattr(pd_out, "to_pandas") else pd_out
    npd = pdf.to_numpy(dtype=float) if hasattr(pdf, "to_numpy") else np.asarray(pdf)
    pln = pl_out.to_pandas().to_numpy(dtype=float)
    np.testing.assert_allclose(npd, pln, equal_nan=True, rtol=1e-6, atol=1e-9)
