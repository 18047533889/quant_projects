# -*- coding: utf-8 -*-
"""2026-08-08 Gemini V2 round: registration, semantics, PIT, golden parity.

Covers the 39 new primitives (18 daily, 13 extended, 8 research): HVG network,
RQA line structure, GLR/Pettitt change points, EDGE / Abdi-Ranaldo /
Pastor-Stambaugh spreads, Qn / Hodges-Lehmann robust scale, Pickands / EVT
stability / Allan factor, composition (CoDa), cross-spectral coherence/phase,
global-state Hartigan dip / Wasserstein barycenter, intraday impact decay,
multifractal asymmetry, and the research DMD / bicoherence / kernel-Granger /
residualised-HSIC / BDS / Gaussian-SR primitives.

The engineering contract under test:
* prefix invariance ``F(X[:T]) == F(X_full)[:T]`` and future randomisation
  invariance (no future function) for every trailing-window operator;
* determinism — 20 identical executions are bitwise equal (no random jitter);
* golden parity — EDGE matches the official ``bidask`` package exactly and the
  Hartigan dip matches the reference ``diptest`` exactly on synthetic data;
* NaN fail-closed and shape-preserving.
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
    "ts_hvg_degree_entropy",
    "ts_hvg_forward_backward_asymmetry",
    "ts_recurrence_determinism",
    "ts_recurrence_laminarity",
    "ts_glr_mean_shift_score",
    "ts_glr_variance_shift_score",
    "ts_edge_effective_spread",
    "ts_abdi_ranaldo_spread",
    "ts_qn_scale",
    "composition_clr_component",
    "composition_aitchison_distance",
    "composition_ilr_balance",
    "cs_hartigan_dip",
    "ts_cross_spectral_coherence",
    "ts_cross_spectral_phase",
    "event_allan_factor",
    "composition_entropy",
    "composition_js_divergence",
]
_EXTENDED_OPS = [
    "ts_hvg_clustering_coefficient",
    "ts_hvg_assortativity",
    "ts_hvg_motif_entropy",
    "ts_recurrence_mean_diagonal_length",
    "ts_recurrence_longest_vertical_length",
    "ts_pettitt_change_score",
    "ts_pickands_tail_index",
    "ts_evt_threshold_stability",
    "ts_hodges_lehmann_location",
    "ts_pastor_stambaugh_liquidity_gamma",
    "group_wasserstein_barycenter_distance",
    "intraday_impact_decay_rate",
    "ts_multifractal_asymmetry",
    "event_allan_scaling_slope",
    "event_allan_log_mean",
]
_RESEARCH_OPS = [
    "ts_dmd_dominant_growth_rate",
    "ts_dmd_dominant_frequency",
    "ts_dmd_mode_concentration",
    "ts_bicoherence_max",
    "ts_kernel_granger_score",
    "ts_residualized_hsic",
    "ts_bds_statistic",
    "ts_sr_gaussian_mean_shift_score",
]
_ALL_OPS = _DAILY_OPS + _EXTENDED_OPS + _RESEARCH_OPS


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all

    load_all()


def _make_panels(n: int = 140, m: int = 8, seed: int = 3) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(m)]
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal((n, m)), axis=0)),
        index=idx, columns=cols,
    )
    logr = np.vstack([np.zeros(m), np.diff(np.log(close.to_numpy()), axis=0)])
    ret = pd.DataFrame(logr, index=idx, columns=cols)
    vol = pd.DataFrame(1.0e4 + rng.integers(0, 3.0e4, (n, m)), index=idx, columns=cols).astype(float)
    amt = (vol.to_numpy() * close.to_numpy())
    # 0/1 event indicator panel (for the Allan-factor family input contract).
    event = pd.DataFrame(
        (rng.uniform(size=(n, m)) < 0.02).astype(float), index=idx, columns=cols
    )
    return {
        "close": close,
        "ret": ret,
        "vol": pd.DataFrame(vol),
        "amt": pd.DataFrame(amt, index=idx, columns=cols),
        "high": close * 1.02,
        "low": close * 0.98,
        "event": event,
        "grp": pd.DataFrame(
            np.tile(np.array(["g0", "g0", "g0", "g1", "g1", "g1", "g2", "g2"]), (n, 1)),
            index=idx, columns=cols,
        ),
    }


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op", _ALL_OPS)
def test_registered_both_backends(_loaded, op):
    backends = OperatorRegistry.backends_for(op)
    assert "pandas_numpy" in backends, op
    assert "polars" in backends, op


def test_surface_classification(_loaded):
    for op in _DAILY_OPS:
        assert classify_canonical(op) == "daily", op
    for op in _EXTENDED_OPS:
        assert classify_canonical(op) == "extended", op
    for op in _RESEARCH_OPS:
        assert classify_canonical(op) == "research", op


# ---------------------------------------------------------------------------
# prefix invariance + future randomisation invariance
# ---------------------------------------------------------------------------
# canonical -> (panel arg names, kwargs).  Windows stay < the prefix cut so
# the trailing windows are fully populated in the prefix.
_TS_SPECS: dict[str, tuple[list[str], dict]] = {
    "ts_hvg_degree_entropy": (["ret"], {"window": 32}),
    "ts_hvg_forward_backward_asymmetry": (["ret"], {"window": 32}),
    "ts_hvg_clustering_coefficient": (["ret"], {"window": 32}),
    "ts_hvg_assortativity": (["ret"], {"window": 32}),
    "ts_hvg_motif_entropy": (["ret"], {"window": 24}),
    "ts_recurrence_determinism": (["ret"], {"window": 32, "dim": 1}),
    "ts_recurrence_laminarity": (["ret"], {"window": 32, "dim": 1}),
    "ts_recurrence_mean_diagonal_length": (["ret"], {"window": 32, "dim": 1}),
    "ts_recurrence_longest_vertical_length": (["ret"], {"window": 32, "dim": 1}),
    "ts_glr_mean_shift_score": (["ret"], {"window": 48, "min_segment": 6}),
    "ts_glr_variance_shift_score": (["ret"], {"window": 48, "min_segment": 6}),
    "ts_pettitt_change_score": (["ret"], {"window": 48, "min_segment": 6}),
    "ts_edge_effective_spread": (["close", "high", "low", "close"], {"window": 20}),
    "ts_abdi_ranaldo_spread": (["close", "high", "low"], {"window": 20}),
    "ts_pastor_stambaugh_liquidity_gamma": (["ret", "ret", "amt"], {"window": 40, "min_periods": 12}),
    "ts_qn_scale": (["ret"], {"window": 32}),
    "ts_hodges_lehmann_location": (["ret"], {"window": 32}),
    "ts_pickands_tail_index": (["ret"], {"window": 60, "k": 4}),
    "ts_evt_threshold_stability": (["ret"], {"window": 60, "k_min": 4, "k_max": 12}),
    "event_allan_factor": (["event"], {"window": 64, "scale": 8}),
    "event_allan_scaling_slope": (["event"], {"window": 64, "max_scale": 8}),
    "event_allan_log_mean": (["event"], {"window": 64, "max_scale": 8}),
    "cs_hartigan_dip": (["ret"], {"min_cross": 3}),
    "group_wasserstein_barycenter_distance": (["ret", "grp"], {"window": 32}),
    "ts_cross_spectral_coherence": (["ret", "vol"], {"window": 40}),
    "ts_cross_spectral_phase": (["ret", "vol"], {"window": 40}),
    "ts_multifractal_asymmetry": (["ret"], {"window": 60}),
    "ts_dmd_dominant_growth_rate": (["ret"], {"window": 60, "rank": 3, "dim": 3}),
    "ts_dmd_dominant_frequency": (["ret"], {"window": 60, "rank": 3, "dim": 3}),
    "ts_dmd_mode_concentration": (["ret"], {"window": 60, "rank": 3, "dim": 3, "top_k": 2}),
    "ts_bicoherence_max": (["ret"], {"window": 48, "n_segments": 2}),
    "ts_kernel_granger_score": (["ret", "vol"], {"window": 48, "lag": 2}),
    "ts_residualized_hsic": (["ret", "vol", "amt"], {"window": 48}),
    "ts_bds_statistic": (["ret"], {"window": 60, "embedding_dim": 2, "distance_multiplier": 1.5}),
    "ts_sr_gaussian_mean_shift_score": (["ret"], {"window": 48, "shift_sigma": 1.0, "baseline_window": 12}),
    "composition_clr_component": (["ret", "vol", "amt"], {}),
    "composition_entropy": (["vol", "amt", "close"], {}),
    "composition_aitchison_distance": (["vol", "amt", "close", "vol", "amt", "close"], {}),
    "composition_ilr_balance": (["vol", "amt", "close", "vol", "amt", "close"], {}),
    "composition_js_divergence": (["vol", "amt", "close", "vol", "amt", "close"], {}),
}


def _future_randomization_invariant(canonical: str, panel_keys: list[str], kwargs: dict) -> None:
    n, cut = 120, 70
    panels = _make_panels(n=n)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    args_full = [panels[k] for k in panel_keys]
    out_full = op.calculate(*args_full, **kwargs)

    # prefix: run on first `cut` rows only
    args_pre = [p.iloc[:cut] for p in args_full]
    out_pre = op.calculate(*args_pre, **kwargs)
    assert out_pre.shape[0] == cut, canonical
    assert np.allclose(
        out_full.iloc[:cut].to_numpy(), out_pre.to_numpy(), equal_nan=True
    ), f"{canonical}: prefix invariance broken"

    # future randomisation: replace rows strictly after `cut` with noise
    rng = np.random.default_rng(123)
    args_fut = [p.copy() for p in args_full]
    for p in args_fut:
        p.iloc[cut + 1 :] = rng.normal(size=(n - cut - 1, p.shape[1]))
    out_fut = op.calculate(*args_fut, **kwargs)
    assert np.allclose(
        out_full.iloc[:cut].to_numpy(), out_fut.iloc[:cut].to_numpy(), equal_nan=True
    ), f"{canonical}: future randomisation invariance broken"


@pytest.mark.parametrize("canonical", sorted(_TS_SPECS.keys()))
def test_prefix_and_future_invariance(_loaded, canonical):
    panel_keys, kwargs = _TS_SPECS[canonical]
    _future_randomization_invariant(canonical, panel_keys, kwargs)


# ---------------------------------------------------------------------------
# determinism (bitwise identical across 20 runs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical", ["ts_hvg_degree_entropy", "ts_qn_scale", "ts_glr_mean_shift_score"])
def test_deterministic_20_runs(_loaded, canonical):
    panels = _make_panels(n=80)
    args = [panels["ret"]]
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    first = op.calculate(*args, window=32).to_numpy()
    for _ in range(19):
        again = op.calculate(*args, window=32).to_numpy()
        assert np.array_equal(first, again, equal_nan=True), canonical


# ---------------------------------------------------------------------------
# golden parity: EDGE vs official bidask, dip vs reference diptest
# ---------------------------------------------------------------------------
def test_edge_golden_vs_bidask(_loaded):
    pytest.importorskip("bidask")
    from bidask.edge import edge as bidask_edge

    rng = np.random.default_rng(7)
    n = 60
    p = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    o = p * np.exp(rng.normal(0, 0.002, n))
    c = p * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    official = float(bidask_edge(o, h, l, c))
    op = OperatorRegistry.get("ts_edge_effective_spread", "pandas_numpy")
    frame = lambda a: pd.DataFrame({"s0": a})
    out = op.calculate(frame(o), frame(h), frame(l), frame(c), window=n).to_numpy()[-1, 0]
    assert np.isfinite(official) and np.isfinite(out)
    assert abs(official - out) <= 1e-12 * max(1.0, abs(official))


def test_hartigan_dip_golden(_loaded):
    pytest.importorskip("diptest")
    from diptest import dipstat

    rng = np.random.default_rng(5)
    samples = {
        "normal": rng.standard_normal(200),
        "bimodal": np.concatenate([rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)]),
        "uniform": rng.uniform(-1, 1, 150),
    }
    op = OperatorRegistry.get("cs_hartigan_dip", "pandas_numpy")
    for label, data in samples.items():
        ref = float(dipstat(np.sort(data)))
        # one row whose cross-section is the whole sample (dip over N values)
        frame = pd.DataFrame(data.reshape(1, -1))
        out_dip = op.calculate(frame, min_cross=2).to_numpy()[0, 0] / np.sqrt(data.size)
        assert abs(ref - out_dip) <= 1e-12 * max(1.0, abs(ref)), label


# ---------------------------------------------------------------------------
# semantic pinned references
# ---------------------------------------------------------------------------
def test_hvg_reversibility_symmetry(_loaded):
    from cleaned_operators.hvg_ext import _hvg_stats

    rng = np.random.default_rng(2)
    wn = rng.standard_normal(200)
    op = OperatorRegistry.get("ts_hvg_forward_backward_asymmetry", "pandas_numpy")
    f1 = op.calculate(pd.DataFrame({"a": wn}), window=100).to_numpy()[-1, 0]
    assert np.isfinite(f1)
    assert f1 < 0.5  # white noise is near-reversible -> asymmetry small
    # the asymmetry of a *fixed window* is invariant to time reversal
    v = wn[:100]
    a_fwd = _hvg_stats(v)["asymmetry"]
    a_rev = _hvg_stats(v[::-1])["asymmetry"]
    assert abs(a_fwd - a_rev) < 1e-12


def test_glr_detects_mean_break(_loaded):
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.0, 0.05, 40), rng.normal(1.0, 0.05, 40)])
    flat = rng.normal(0.0, 0.05, 80)
    op = OperatorRegistry.get("ts_glr_mean_shift_score", "pandas_numpy")
    f = lambda a: pd.DataFrame({"a": a})
    brk = op.calculate(f(x), window=80, min_segment=6).to_numpy()[-1, 0]
    base = op.calculate(f(flat), window=80, min_segment=6).to_numpy()[-1, 0]
    assert np.isfinite(brk) and np.isfinite(base)
    assert brk > base + 2.0  # a real break scores far above the noise floor


def test_rqa_determinism_high_for_periodic(_loaded):
    periodic = np.tile([1.0, 0.5, -0.5, -1.0], 25)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(100)
    op = OperatorRegistry.get("ts_recurrence_determinism", "pandas_numpy")
    f = lambda a: pd.DataFrame({"a": a})
    d_periodic = op.calculate(f(periodic), window=60, dim=1, min_line=4).to_numpy()[-1, 0]
    d_noise = op.calculate(f(noise), window=60, dim=1, min_line=4).to_numpy()[-1, 0]
    assert np.isfinite(d_periodic) and np.isfinite(d_noise)
    assert d_periodic > d_noise


def test_qn_scale_unbiased_on_normal(_loaded):
    rng = np.random.default_rng(6)
    op = OperatorRegistry.get("ts_qn_scale", "pandas_numpy")
    samples = np.stack([rng.standard_normal(120) for _ in range(12)])
    frame = pd.DataFrame({f"s{i}": samples[i] for i in range(12)})
    out = op.calculate(frame, window=120).to_numpy()[-1, :]
    assert np.all(np.isfinite(out))
    assert abs(np.mean(out) - 1.0) < 0.2  # unbiased for sigma under normality


def test_composition_rejects_nonpositive(_loaded):
    panels = _make_panels(n=30)
    op = OperatorRegistry.get("composition_clr_component", "pandas_numpy")
    out = op.calculate(panels["vol"], panels["amt"], panels["close"]).to_numpy()
    assert np.all(np.isfinite(out[5:, :]))  # positive parts -> finite CLR
    # introduce a zero / negative part -> that cell fail-closes
    bad = panels["amt"].copy()
    bad.iloc[10, 0] = -1.0
    out2 = op.calculate(panels["vol"], bad, panels["close"]).to_numpy()
    assert np.isnan(out2[10, 0])


def test_abdi_ranaldo_matches_reference_formula(_loaded):
    # hand-computed monthly Abdi-Ranaldo on a tiny clean series
    close = pd.DataFrame({"a": [10.0, 10.5, 11.0, 10.8, 11.2]})
    high = close * 1.03
    low = close * 0.97
    op = OperatorRegistry.get("ts_abdi_ranaldo_spread", "pandas_numpy")
    out = op.calculate(close, high, low, window=5).to_numpy()[-1, 0]
    c = np.log(close["a"].to_numpy())
    eta = (np.log(high["a"].to_numpy()) + np.log(low["a"].to_numpy())) / 2.0
    terms = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    expected = np.sqrt(max(4.0 * np.mean(terms), 0.0))
    assert abs(out - expected) < 1e-12


def test_cross_spectral_phase_known_shift(_loaded):
    t = np.arange(128)
    freq = 8.0 / 128
    x = np.sin(2 * np.pi * freq * t)
    y = np.sin(2 * np.pi * freq * (t - 3))
    op = OperatorRegistry.get("ts_cross_spectral_phase", "pandas_numpy")
    f = lambda a: pd.DataFrame({"a": a})
    ph = op.calculate(f(x), f(y), window=128).to_numpy()[-1, 0]
    exp_ph = (2 * np.pi * freq * 3 + np.pi) % (2 * np.pi) - np.pi
    assert abs(ph - exp_ph) < 0.05


# ---------------------------------------------------------------------------
# polars parity for representative ops
# ---------------------------------------------------------------------------
_POLARS_PARITY = [
    ("ts_hvg_degree_entropy", ["ret"], {"window": 32}),
    ("ts_glr_mean_shift_score", ["ret"], {"window": 48, "min_segment": 6}),
    ("ts_qn_scale", ["ret"], {"window": 32}),
    ("ts_abdi_ranaldo_spread", ["close", "high", "low"], {"window": 20}),
    ("composition_clr_component", ["ret", "vol", "amt"], {}),
    ("cs_hartigan_dip", ["ret"], {"min_cross": 3}),
    ("ts_cross_spectral_coherence", ["ret", "vol"], {"window": 40}),
]


@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize("canonical,panel_keys,kwargs", _POLARS_PARITY)
def test_polars_parity(_loaded, canonical, panel_keys, kwargs):
    panels = _make_panels(n=90)
    pd_op = OperatorRegistry.get(canonical, "pandas_numpy")
    pl_op = OperatorRegistry.get(canonical, "polars")
    args = [panels[k] for k in panel_keys]
    out_pd = pd_op.calculate(*args, **kwargs)

    # wide polars frames: a date column (skipped by the bridge) plus one column
    # per instrument — exactly what the runtime layer passes to the polars backend.
    date_col = np.asarray(panels["close"].index, dtype="datetime64[ns]").astype("int64")
    pl_wide = []
    for j in range(len(args)):
        data = {f"S{i}": args[j][c].to_numpy() for i, c in enumerate(panels["close"].columns)}
        pl_wide.append(pl.DataFrame(
            {"date": date_col} | {k: pl.Series(k, v) for k, v in data.items()}
        ))
    out_pl = pl_op._calculate_series(*pl_wide, **kwargs)
    arr_pl = np.stack([out_pl[k].to_numpy() for k in out_pl.columns if k != "date"], axis=1)
    assert np.allclose(out_pd.to_numpy(), arr_pl, equal_nan=True), canonical
