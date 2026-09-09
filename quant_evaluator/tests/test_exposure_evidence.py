"""
R61-FI-024: per-style exposure / purity evidence tests.

Pins against hand-constructed synthetic exposure panels:
  - per-style exposure signs and magnitudes (each style a typed field, no
    single scalar "exposure" API);
  - max_absolute_style_exposure picks the largest |mean| style;
  - purity_ratio matches hand computation;
  - neutralized_rank_ic matches hand OLS-residual Spearman;
  - the injected-provider semantic: exposure data always arrives through an
    ExposurePanel (source_ref = DataAccess-authoritative ref, provider
    injectable); tests inject synthetic panels and never send a real COS
    request;
  - the registry wiring: 12 new exposure ids registered with bound compute_fn,
    and no single-scalar exposure id exists;
  - CPU-vs-GPU parity of the neutralized rank IC when CuPy/CUDA is present
    (GPU optional — CPU reference authoritative).

Run (from the repo root):
    PYTHONPATH=/home/sunhaiwei/quant_projects python -m pytest -q \
        quant_evaluator/tests/test_exposure_evidence.py --timeout=300
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.metrics.exposure_evidence import (
    ExposureStyle,
    ExposurePanel,
    FactorLoadingSeries,
    StyleExposureEvidence,
    build_factor_loading_series,
    compute_style_exposure_evidence,
    compute_max_absolute_style_exposure,
    compute_exposure_drift,
    compute_purity_ratio,
    compute_neutralized_rank_ic,
    compute_residual_rank_ic,
    compute_industry_exposure,
    compute_size_exposure,
    compute_beta_exposure,
    compute_liquidity_exposure,
    compute_volatility_exposure,
    compute_momentum_exposure,
    compute_exposure_evidence,
)
from quant_evaluator.registry.metrics import get_metric, list_metrics, MetricStatus

# Registry ids that must exist after R61-FI-024 (all bound compute_fn).
_EXPOSURE_IDS = (
    "industry_exposure",
    "size_exposure",
    "beta_exposure",
    "liquidity_exposure",
    "volatility_exposure",
    "momentum_exposure",
    "max_absolute_style_exposure",
    "exposure_drift",
    "neutralized_rank_ic",
    "residual_rank_ic",
    "purity_ratio",
)

_STYLES = (
    "industry",
    "size",
    "beta",
    "liquidity",
    "volatility",
    "momentum",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _panel(T=60, N=50, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, (T, N, len(_STYLES)))
    return ExposurePanel(
        values=values,
        style_names=_STYLES,
        source_ref="ds://test_exposures",
        provider="synthetic-test",
        date_index=tuple(range(T)),
    )


def _known_panel():
    """Panel with a dominant, positively-loaded industry column and a
    negative size column -> known per-style exposure signs/magnitudes."""
    T, N = 60, 50
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.05, (T, N, len(_STYLES)))
    values[:, :, 0] = 0.30 + rng.normal(0.0, 0.02, (T, N))  # industry +0.30
    values[:, :, 1] = -0.20 + rng.normal(0.0, 0.02, (T, N))  # size -0.20
    values[:, :, 2] = rng.normal(0.0, 0.01, (T, N))          # beta ~0
    return ExposurePanel(
        values=values,
        style_names=_STYLES,
        source_ref="ds://test_exposures",
        provider="synthetic-test",
    )


def _known_loadings():
    """Typed factor evidence with a positive industry and negative size loading."""
    p = _panel(seed=5)
    factor = 2.0 * p.values[:, :, 0] - p.values[:, :, 1] + 0.2 * p.values[:, :, 2]
    return build_factor_loading_series(p, factor, factor_id="known", min_obs=10)


def _loading(values, *, r_squared=None, style_names=None):
    """Small typed fixture for reducers that consume already-estimated loadings."""
    values = np.asarray(values, dtype=float)
    t, k = values.shape
    names = tuple(style_names or (f"s{i}" for i in range(k)))
    r2 = np.asarray(r_squared if r_squared is not None else np.zeros(t), dtype=float)
    return FactorLoadingSeries(
        values=values,
        raw_loadings=values,
        r_squared=r2,
        counts=np.full(t, 20),
        style_names=names,
        factor_id="fixture",
        source_ref="risk:fixture",
        provider="test",
        date_index=tuple(range(t)),
        common_support_ref="support:fixture",
        diagnostics=tuple({} for _ in range(t)),
        factor_value_ref="factor:fixture",
    )


# ---------------------------------------------------------------------------
# 1. ExposurePanel contract (DataAccess-authoritative ref, injectable provider)
# ---------------------------------------------------------------------------
def test_exposure_panel_frozen_and_authoritative_ref():
    p = _panel()
    assert p.source_ref == "ds://test_exposures"
    assert p.provider == "synthetic-test"
    # QE never fabricates exposure data: the panel must carry a source_ref.
    assert p.source_ref
    # frozen values (mutation raises)
    with pytest.raises(ValueError):
        p.values[0, 0, 0] = 99.0


def test_exposure_panel_shape_validation():
    with pytest.raises(ValueError):
        ExposurePanel(values=np.zeros((5, 5)), style_names=_STYLES)
    with pytest.raises(ValueError):
        ExposurePanel(
            values=np.zeros((5, 5, 3)),
            style_names=_STYLES,  # length mismatch
        )


def test_style_exposure_evidence_typed_per_style():
    """The evidence is per-style typed fields — no single scalar API."""
    ev = compute_style_exposure_evidence(_known_loadings())
    assert isinstance(ev, StyleExposureEvidence)
    assert len(ev.values) == len(_STYLES)
    assert tuple(ev.style_names) == _STYLES
    assert ev.source_ref == "ds://test_exposures"
    assert ev.provider == "synthetic-test"


# ---------------------------------------------------------------------------
# 2. Per-style exposure signs and magnitudes
# ---------------------------------------------------------------------------
def test_known_style_signs_and_magnitudes():
    p = _known_loadings()
    assert compute_industry_exposure(p) > 0.5
    assert compute_size_exposure(p) < -0.2
    assert compute_beta_exposure(p) > 0.0
    # the typed per-style accessors match the evidence vector
    ev = compute_style_exposure_evidence(p, absolute=False)
    assert ev.values[0] == pytest.approx(compute_industry_exposure(p), abs=1e-9)
    assert ev.values[1] == pytest.approx(compute_size_exposure(p), abs=1e-9)


def test_missing_style_is_nan_not_zero():
    """A panel without a style dimension -> NaN (explicit missing evidence),
    never a fabricated 0."""
    T, N = 40, 30
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 1.0, (T, N, 2))
    p = _loading(np.zeros((T, 2)), style_names=("industry", "size"))
    assert np.isnan(compute_momentum_exposure(p))
    assert np.isnan(compute_liquidity_exposure(p))


def test_absolute_vs_signed_mean():
    """absolute=True returns mean |x| >= |signed mean| (equal for constant
    sign), and both are per-style fields."""
    p = _known_loadings()
    signed = compute_style_exposure_evidence(p, absolute=False)
    abs_ev = compute_style_exposure_evidence(p, absolute=True)
    assert abs_ev.values[0] >= abs(signed.values[0])
    assert abs_ev.values[1] >= abs(signed.values[1])


# ---------------------------------------------------------------------------
# 3. max_absolute_style_exposure
# ---------------------------------------------------------------------------
def test_max_absolute_style_exposure_picks_largest_abs():
    p = _known_loadings()
    res = compute_max_absolute_style_exposure(p)
    assert res["style"] == "industry"
    assert res["value"] > 0.5
    assert res["absolute_mean"] >= abs(res["value"])


def test_max_absolute_style_exposure_unknown_when_no_data():
    T, N = 40, 30
    p = _loading(np.full((T, 3), np.nan), style_names=("a", "b", "c"))
    res = compute_max_absolute_style_exposure(p)
    assert res["style"] == ExposureStyle.UNKNOWN.value
    assert np.isnan(res["value"])


# ---------------------------------------------------------------------------
# 4. exposure_drift
# ---------------------------------------------------------------------------
def test_exposure_drift_hand_computed():
    T, N, K = 5, 4, 3
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1.0, (T, K))
    p = _loading(values, style_names=("a", "b", "c"))
    drift = compute_exposure_drift(p)
    diffs = []
    for t in range(T - 1):
        a = values[t]
        b = values[t + 1]
        joint = np.isfinite(a) & np.isfinite(b)
        diffs.append(np.mean(np.abs(a[joint] - b[joint])))
    assert drift == pytest.approx(float(np.mean(diffs)), abs=1e-9)


def test_exposure_drift_constant_panel_zero():
    T, N, K = 5, 4, 3
    p = _loading(np.ones((T, K)), style_names=("a", "b", "c"))
    assert compute_exposure_drift(p) == pytest.approx(0.0, abs=1e-12)


def test_exposure_drift_short_nan():
    p = _loading(np.ones((1, 3)), style_names=("a", "b", "c"))
    assert np.isnan(compute_exposure_drift(p))


# ---------------------------------------------------------------------------
# 5. purity_ratio
# ---------------------------------------------------------------------------
def test_purity_ratio_hand_computed():
    r2 = np.array([0.0, 0.25, 0.75, 1.0])
    p = _loading(np.zeros((4, 2)), r_squared=r2, style_names=("a", "b"))
    purity = compute_purity_ratio(p)
    expected = float(np.mean(1.0 - r2))
    assert compute_purity_ratio(p, min_finite=1) == pytest.approx(expected, abs=1e-12)


def test_purity_ratio_no_valid_style_nan():
    T, N, K = 40, 30, 2
    p = _loading(np.full((T, K), np.nan), r_squared=np.full(T, np.nan), style_names=("a", "b"))
    assert np.isnan(compute_purity_ratio(p))


def test_purity_ratio_high_for_clean_factor():
    """A factor with tiny style footprint -> high purity (close to 1)."""
    T, N, K = 40, 30, 3
    rng = np.random.default_rng(13)
    p = _loading(np.zeros((T, K)), r_squared=np.full(T, 1e-6), style_names=("a", "b", "c"))
    assert compute_purity_ratio(p) > 0.99


# ---------------------------------------------------------------------------
# 6. neutralized_rank_ic / residual_rank_ic
# ---------------------------------------------------------------------------
def test_neutralized_rank_ic_hand_computed():
    """Per-date OLS residualization on the panel, then Spearman vs forward
    returns, time-averaged — hand-computed with the same building blocks."""
    from quant_evaluator.metrics.exposure import compute_factor_loadings
    from quant_evaluator.metrics.ic import _spearman_rank_correlation
    from scipy.stats import spearmanr

    T, N, K = 60, 50, 3
    rng = np.random.default_rng(17)
    panel = rng.normal(0.0, 1.0, (T, N, K))
    factor = 0.4 * panel[:, :, 0] + rng.normal(0.0, 1.0, (T, N))
    fwd = 0.3 * factor + rng.normal(0.0, 0.5, (T, N))
    p = ExposurePanel(values=panel, style_names=("a", "b", "c"))

    mine = compute_neutralized_rank_ic(factor, fwd, p, min_obs=10)

    _, _, residuals = compute_factor_loadings(
        factor, panel, intercept=True, min_obs=10
    )
    daily = []
    for t in range(T):
        joint = np.isfinite(fwd[t]) & np.isfinite(residuals[t])
        if np.sum(joint) < 2:
            continue
        rho = spearmanr(
            residuals[t][joint], fwd[t][joint], nan_policy="omit"
        ).statistic
        if np.isfinite(rho):
            daily.append(float(rho))
    assert mine == pytest.approx(float(np.mean(daily)), abs=1e-9)


def test_neutralized_rank_ic_survives_style_removal():
    """A pure-style factor (factor == industry column) has ~0 neutralized IC
    after residualization, while its raw IC may be non-zero."""
    T, N, K = 80, 60, 3
    rng = np.random.default_rng(19)
    panel = rng.normal(0.0, 1.0, (T, N, K))
    factor = panel[:, :, 0].copy()  # exactly the industry style
    fwd = rng.normal(0.0, 1.0, (T, N))
    # inject a tiny independent signal so fwd is not pure noise
    fwd = fwd + 0.05 * panel[:, :, 0]
    p = ExposurePanel(values=panel, style_names=("a", "b", "c"))
    neutralized = compute_neutralized_rank_ic(factor, fwd, p, min_obs=10)
    # Exact style replication leaves a constant residual, so rank IC is
    # undefined rather than fabricated as zero.
    assert np.isnan(neutralized)


def test_residual_rank_ic_alias_matches():
    T, N, K = 60, 50, 3
    rng = np.random.default_rng(23)
    panel = rng.normal(0.0, 1.0, (T, N, K))
    factor = rng.normal(0.0, 1.0, (T, N))
    fwd = rng.normal(0.0, 1.0, (T, N))
    p = ExposurePanel(values=panel, style_names=("a", "b", "c"))
    assert compute_residual_rank_ic(factor, fwd, p) == pytest.approx(
        compute_neutralized_rank_ic(factor, fwd, p), abs=1e-12
    )


def test_neutralized_rank_ic_shape_mismatch_raises():
    p = _panel(T=60, N=50)
    with pytest.raises(ValueError):
        compute_neutralized_rank_ic(np.zeros((5, 5)), np.zeros((5, 5)), p)


# ---------------------------------------------------------------------------
# 7. compute_exposure_evidence bundle
# ---------------------------------------------------------------------------
def test_exposure_evidence_bundle_has_all_fields():
    p = _known_panel()
    T, N = 60, 50
    rng = np.random.default_rng(29)
    factor = rng.normal(0.0, 1.0, (T, N))
    fwd = rng.normal(0.0, 1.0, (T, N))
    bundle = compute_exposure_evidence(p, factor, fwd)
    expected_keys = {
        "industry_exposure",
        "size_exposure",
        "beta_exposure",
        "liquidity_exposure",
        "volatility_exposure",
        "momentum_exposure",
        "max_absolute_style_exposure",
        "exposure_drift",
        "neutralized_rank_ic",
        "residual_rank_ic",
        "purity_ratio",
    }
    assert expected_keys.issubset(bundle)
    assert bundle["estimation_scope"] == "SAME_DATE_DESCRIPTIVE"
    assert bundle["method_version"] == "factor_standardized_wls.v2"
    assert isinstance(bundle["factor_loading_series"], FactorLoadingSeries)


# ---------------------------------------------------------------------------
# 8. Registry wiring
# ---------------------------------------------------------------------------
def test_exposure_ids_registered_with_bound_compute_fn():
    ids = set(list_metrics())
    for metric_id in _EXPOSURE_IDS:
        assert metric_id in ids, metric_id
        spec = get_metric(metric_id)
        assert spec.compute_fn is not None, metric_id
        # Value-based (not `is`) comparison: the shared registry may be
        # reloaded by the seal-lifecycle tests, which recreates the enum
        # classes; `name`/`value` stay stable across the reload.
        assert spec.status.name == MetricStatus.EXPERIMENTAL.name, metric_id


def test_no_single_scalar_exposure_metric():
    """R61-FI-024: there must NOT be a single-scalar "exposure" long-term API;
    exposure is per-style typed fields only."""
    ids = set(list_metrics())
    for bad in ("exposure", "style_exposure", "factor_exposure"):
        assert bad not in ids, bad


def test_exposure_specs_domain_exposure():
    from quant_evaluator.registry.metrics import Domain

    for metric_id in _EXPOSURE_IDS:
        assert get_metric(metric_id).domain is Domain.EXPOSURE, metric_id


# ---------------------------------------------------------------------------
# 9. CPU vs GPU parity (GPU optional — CPU reference authoritative)
# ---------------------------------------------------------------------------
def test_cpu_gpu_neutralized_rank_ic_parity():
    """When CuPy + CUDA are present the GPU path must match the CPU reference
    (delta < 1e-6).  Without CuPy the GPU wrapper fails closed with
    OptionalDependencyMissing and the CPU reference stays authoritative."""
    try:
        import cupy as cp  # noqa: F401
        from quant_evaluator.kernels.gpu.exposure_evidence import (
            gpu_neutralized_rank_ic,
        )
    except Exception:
        pytest.skip("CuPy not available on this host; CPU reference authoritative")

    T, N, K = 60, 50, 3
    rng = np.random.default_rng(31)
    panel = rng.normal(0.0, 1.0, (T, N, K))
    factor = rng.normal(0.0, 1.0, (T, N))
    fwd = rng.normal(0.0, 1.0, (T, N))
    p = ExposurePanel(values=panel, style_names=("a", "b", "c"))
    cpu = compute_neutralized_rank_ic(factor, fwd, p, min_obs=10)
    gpu = gpu_neutralized_rank_ic(factor, fwd, panel, min_obs=10)
    assert cpu == pytest.approx(gpu, abs=1e-6)
