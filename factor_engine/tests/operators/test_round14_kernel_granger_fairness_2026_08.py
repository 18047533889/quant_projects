# -*- coding: utf-8 -*-
"""R14 estimator-fairness regression tests for the statistical operators.

* P1-11 / P1-12 — ``ts_kernel_granger_score``: the X RBF bandwidth must come
  from X's OWN training-block geometry (P1-11), and the full-model kernel must
  be trace-normalised (``K_F = (K_Y + eta·K_X)/(1+eta)``) so it shares the same
  regularisation strength as the restricted kernel (P1-12).  Under an
  independent-X null the score is ≈ 0 and NOT systematically positive.
* P1-13 — ``ts_residualized_hsic``: blocked/purged CONTIGUOUS cross-fitting
  replaces the old interleaved even/odd split (test points no longer have
  immediate neighbours in the training set).  ``purge_gap`` is exposed.
* P1-14 — ``ts_cross_spectral_phase``: the default coherence gate is non-zero
  (0.2), so near-zero-coherence pairs emit NaN instead of a random angle.
* P2-34 — ``ts_bicoherence_top_decile_excess``: the top-decile bicoherence is
  reported minus a deterministic phase-surrogate null, so the value no longer
  drifts with window/n_segments/max_freq; under white noise it hovers near 0.

The operator modules are imported directly (each module's ``_register``
populates the registry) instead of ``load_all()`` so this suite stays fast.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import cleaned_operators.cross_spectrum  # noqa: F401  (registers)
import cleaned_operators.research_spectral  # noqa: F401  (registers)

from cleaned_operators.cross_spectrum import _cross_spectrum_window
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.research_spectral import (
    _bicoherence_top_decile_excess,
    _bicoherence_top_decile_mean,
    _kernel_granger_score,
    _rbf,
    _residualized_hsic,
)

_SEG = 64


def _ar1(n: int, rho: float, sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + sd * e[i]
    return out


def _median_bandwidth(tr: np.ndarray) -> float:
    """Median pairwise distance in a TRAINING block (mirrors the operator)."""
    d2 = (
        np.sum(tr[:, None, :] ** 2, axis=2)
        + np.sum(tr[None, :, :] ** 2, axis=2)
        - 2.0 * tr @ tr.T
    )
    s = float(np.median(np.sqrt(np.maximum(d2, 0.0))))
    return s if np.isfinite(s) and s > 1e-12 else 1.0


# ---------------------------------------------------------------------------
# P1-11 / P1-12 — kernel Granger fairness
# ---------------------------------------------------------------------------
def test_kernel_granger_independent_x_y_score_about_zero() -> None:
    # X completely independent of Y (two independent AR(1) series): the score
    # ln(MSE_restricted / MSE_full) must be ≈ 0 (loose tolerance) and NOT
    # systematically positive across several random draws.  The per-draw std of
    # the estimator is ~0.2 in log-units, so the mean over 16 draws is centred
    # near 0 and stays below the positive-bias threshold.
    scores: list[float] = []
    for k in range(16):
        y = _ar1(300, 0.5, 1.0, 1000 + k)
        x = _ar1(300, 0.3, 1.0, 3000 + k)
        s = _kernel_granger_score(y, x, lag=2)
        assert np.isfinite(s)
        scores.append(float(s))
    scores_arr = np.asarray(scores, dtype=float)
    # about zero (loose tolerance)
    assert abs(scores_arr.mean() < 0.35, scores_arr
    # NOT systematically positive: the mean must not be positive and the
    # majority of individual draws must be non-positive.
    assert scores_arr.mean() <= 0.1, scores_arr
    assert (scores_arr > 0).mean() < 0.5, scores_arr


def test_kernel_granger_full_kernel_trace_fair_and_own_x_bandwidth() -> None:
    # X and Y with very different scales/geometries (returns-like vs smooth
    # volume).  The full-model kernel must satisfy trace(K_F) == trace(K_R)
    # (P1-12) and the X kernel must be built with X's OWN training bandwidth
    # (P1-11), which here differs from Y's.
    y = 0.01 * _ar1(300, 0.2, 1.0, 7)
    x = 1e7 + 1e6 * _ar1(300, 0.95, 1.0, 8)
    n, lg = 300, 2
    n_avail = n - lg
    Y = y[lg:]
    Ylags = np.column_stack([y[lg - k : n - k] for k in range(1, lg + 1)])
    Xlags = np.column_stack([x[lg - k : n - k] for k in range(1, lg + 1)])
    tr_n = int(0.7 * n_avail)

    def _std(tr, te):
        mu = tr.mean(axis=0)
        sd = tr.std(axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        return (tr - mu) / sd, (te - mu) / sd

    Yl_tr, _ = _std(Ylags[:tr_n], Ylags[tr_n:])
    Xl_tr, _ = _std(Xlags[:tr_n], Xlags[tr_n:])
    sigma_y = _median_bandwidth(Yl_tr)
    sigma_x = _median_bandwidth(Xl_tr)

    Kr = _rbf(Yl_tr, Yl_tr, sigma_y)
    Kx = _rbf(Xl_tr, Xl_tr, sigma_x)
    ntr = Kr.shape[0]
    # RBF Gram diagonals are all 1 -> each trace == ntr.
    assert abs(np.trace(Kr) - ntr) < 1e-9
    assert abs(np.trace(Kx) - ntr) < 1e-9
    # P1-11: the X kernel's smoothness is driven by X's own geometry — the
    # smooth X block has a different median pairwise distance than the noisy Y
    # block even after column-wise standardisation.
    assert abs(sigma_x - sigma_y) > 0.05, (sigma_x, sigma_y)
    eta = 0.5
    Kf = (Kr + eta * Kx) / (1.0 + eta)
    # P1-12: trace-normalised full kernel has exactly the same trace as the
    # restricted kernel, so the shared ridge regularises both models equally.
    assert abs(np.trace(Kf) - np.trace(Kr) < 1e-9
    # The operator itself still runs and is finite for wildly different scales.
    score = _kernel_granger_score(y, x, lag=2)
    assert np.isfinite(score)


def test_kernel_granger_public_operator_smoke() -> None:
    op = OperatorRegistry.get("ts_kernel_granger_score", "pandas_numpy")
    assert op is not None
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    y = pd.DataFrame({"a": 0.01 * np.random.default_rng(0).standard_normal(n)}, index=idx)
    x = pd.DataFrame({"a": 1e7 + np.random.default_rng(1).normal(0.0, 1e6, n)}, index=idx)
    out = op.calculate(y, x, window=120, lag=2).to_numpy()
    assert np.isfinite(out[-1, 0])


# ---------------------------------------------------------------------------
# P1-13 — residualised HSIC blocked/purged cross-fitting
# ---------------------------------------------------------------------------
def test_residualized_hsic_accepts_purge_gap_comparable_magnitude() -> None:
    # A mild autocorrelated process where x and y both load on z.
    z = _ar1(240, 0.6, 1.0, 50)
    x = 0.5 * z + _ar1(240, 0.2, 0.5, 51)
    y = 0.5 * z + _ar1(240, 0.2, 0.5, 52)
    base = _residualized_hsic(x, y, z, purge_gap=3)
    assert np.isfinite(base) and base >= 0.0
    # Different purge gaps give comparable magnitudes (the blocked path runs
    # and is stable) while still being distinct residualisations.
    for pg in (1, 5, 8):
        v = _residualized_hsic(x, y, z, purge_gap=pg)
        assert np.isfinite(v) and v >= 0.0
        assert v / max(base, 1e-12) < 3.0, (base, v)
    assert abs(_residualized_hsic(x, y, z, purge_gap=1) - _residualized_hsic(x, y, z, purge_gap=8) > 1e-6
    # A genuine residual dependence (y still depends on x beyond z) shows up.
    y_dep = 0.5 * z + 0.8 * x + _ar1(240, 0.2, 0.3, 53)
    assert _residualized_hsic(x, y_dep, z, purge_gap=3) > _residualized_hsic(x, y, z, purge_gap=3)


def test_residualized_hsic_public_op_exposes_purge_gap() -> None:
    op = OperatorRegistry.get("ts_residualized_hsic", "pandas_numpy")
    assert op is not None
    assert "purge_gap" in op.metadata.param_names
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    z = _ar1(n, 0.6, 1.0, 50)
    x = 0.5 * z + _ar1(n, 0.2, 0.5, 51)
    y = 0.5 * z + _ar1(n, 0.2, 0.5, 52)
    mk = lambda a: pd.DataFrame({"a": np.asarray(a, dtype=float)}, index=idx)  # noqa: E731
    out = op.calculate(mk(x), mk(y), mk(z), window=120, purge_gap=3).to_numpy()
    assert np.isnan(out[-1, 0]) or out[-1, 0] >= 0.0


# ---------------------------------------------------------------------------
# P1-14 — cross-spectral phase coherence gate
# ---------------------------------------------------------------------------
def test_cross_spectral_phase_default_gate_is_nonzero() -> None:
    from cleaned_operators.cross_spectrum import _ts_cross_spectral_phase

    sig = inspect.signature(_ts_cross_spectral_phase)
    # was 0.0 (every finite spectrum emitted a random angle); now a meaningful
    # default gate.
    assert sig.parameters["min_coherence"].default == 0.2


def test_cross_spectral_phase_gates_near_zero_coherence() -> None:
    # Anti-phased segment construction -> genuine near-zero band coherence.
    # With the default gate the phase is NaN; with the gate explicitly disabled
    # (min_coherence=0.0) a random angle is still emitted (the gate is what
    # suppresses the noise angle).
    for f in (4, 8, 12):
        tau = np.arange(_SEG, dtype=float)
        tone = np.sin(2.0 * np.pi * f * tau / _SEG)
        x = np.concatenate([tone, tone, tone])
        y = np.concatenate([tone, -tone, tone])
        c, p = _cross_spectrum_window(x, y, "all", 0.2)
        assert np.isfinite(c) and c < 0.2, (f, c)
        assert np.isnan(p)
        _, p0 = _cross_spectrum_window(x, y, "all", 0.0)
        assert np.isfinite(p0)


def test_cross_spectral_phase_finite_with_strong_coherence() -> None:
    t = np.arange(128)
    freq = 8.0 / 128
    x = np.sin(2.0 * np.pi * freq * t)
    y = np.sin(2.0 * np.pi * freq * (t - 3))
    c, p = _cross_spectrum_window(x, y, "all", 0.2)
    assert c > 0.5
    assert np.isfinite(p)


def test_cross_spectral_phase_public_default_gate_nan_for_antiphased() -> None:
    op = OperatorRegistry.get("ts_cross_spectral_phase", "pandas_numpy")
    assert op is not None
    tau = np.arange(_SEG, dtype=float)
    tone = np.sin(2.0 * np.pi * 8 * tau / _SEG)
    x = np.concatenate([tone, tone, tone])
    y = np.concatenate([tone, -tone, tone])
    idx = pd.date_range("2023-01-01", periods=len(x), freq="D")
    out = op.calculate(
        pd.DataFrame({"a": x}, index=idx),
        pd.DataFrame({"a": y}, index=idx),
        window=128,
    ).to_numpy()
    assert np.isnan(out[-1, 0])


# ---------------------------------------------------------------------------
# P2-34 — bicoherence top-decile excess
# ---------------------------------------------------------------------------
def test_bicoherence_top_decile_excess_white_noise_near_zero() -> None:
    # The excess (real minus phase-surrogate null) hovers near 0 for white
    # noise, whereas the raw top-decile mean drifts with n_segments (P2-34):
    # e.g. n=256 raw ≈ 0.94 at n_seg=2 vs ≈ 0.37 at n_seg=8.
    raw_drift = abs(
        _bicoherence_top_decile_mean(np.random.default_rng(5).standard_normal(256), 2)
        - _bicoherence_top_decile_mean(np.random.default_rng(5).standard_normal(256), 8)
    )
    assert raw_drift > 0.2  # the raw statistic is configuration-sensitive
    rng = np.random.default_rng(5)
    w = rng.standard_normal(256)
    for ns in (2, 4):
        exc = _bicoherence_top_decile_excess(w, ns)
        assert np.isfinite(exc) and abs(exc) < 0.15, (ns, exc)
    for seed in range(5):
        exc = _bicoherence_top_decile_excess(np.random.default_rng(seed).standard_normal(256), 4)
        assert np.isfinite(exc) and abs(exc) < 0.15, (seed, exc)


def test_bicoherence_top_decile_excess_detects_qpc() -> None:
    # A signal with genuine quadratic phase coupling (third-harmonic phase
    # locked to phi1+phi2, plus a little noise so segments differ) survives the
    # phase scramble -> excess is clearly positive.
    ns = 8
    f1, f2, phi1, phi2 = 4, 6, 0.3, 1.1
    tau = np.arange(_SEG, dtype=float)
    for seed in (1, 2, 3):
        g = np.random.default_rng(seed)
        s = np.zeros(ns * _SEG)
        for k in range(ns):
            blk = (
                np.sin(2.0 * np.pi * f1 * tau / _SEG + phi1)
                + np.sin(2.0 * np.pi * f2 * tau / _SEG + phi2)
                + np.sin(2.0 * np.pi * (f1 + f2) * tau / _SEG + phi1 + phi2)
            )
            s[k * _SEG : (k + 1) * _SEG] = blk + 0.1 * g.standard_normal(_SEG)
        exc = _bicoherence_top_decile_excess(s, ns)
        assert np.isfinite(exc) and exc > 0.02, (seed, exc)


def test_bicoherence_top_decile_excess_registered() -> None:
    op = OperatorRegistry.get("ts_bicoherence_top_decile_excess", "pandas_numpy")
    assert op is not None
    assert list(op.metadata.param_names) == ["x", "window", "n_segments", "n_surrogates"]
    # white-noise excess through the public rolling operator hovers near 0.
    rng = np.random.default_rng(11)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"a": rng.standard_normal(n)}, index=idx)
    out = op.calculate(df, window=120, n_segments=4).to_numpy()
    fin = out[np.isfinite(out)]
    assert fin.size > 0
    assert abs(float(np.mean(fin)) < 0.15
