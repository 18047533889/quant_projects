"""Independent V6 exposure contract oracles and semantic refusal tests."""

import numpy as np
import pytest

from quant_evaluator.metrics.exposure_evidence import (
    ExposurePanel,
    FactorLoadingSeries,
    build_factor_loading_series,
    compute_industry_exposure,
    compute_portfolio_exposure,
    compute_purity_ratio,
    compute_style_exposure_evidence,
)


def _panel(risk, *, weights=None):
    t, n, _ = risk.shape
    return ExposurePanel(
        risk,
        style_names=("industry", "size"),
        source_ref="risk:v6-oracle",
        provider="independent-test",
        date_index=tuple(range(t)),
        security_ids=tuple(f"A{i}" for i in range(n)),
        regression_weights=weights,
        weight_ref="cap-weight" if weights is not None else "",
    )


def test_weighted_standardized_loading_and_purity_match_independent_math():
    rng = np.random.default_rng(610)
    t, n = 3, 24
    risk = rng.normal(size=(t, n, 2))
    weights = rng.uniform(0.2, 2.0, size=(t, n))
    noise = rng.normal(scale=0.35, size=(t, n))
    factor = 1.5 * risk[:, :, 0] - 0.7 * risk[:, :, 1] + noise
    loading = build_factor_loading_series(_panel(risk, weights=weights), factor)

    expected_loading = np.empty((t, 2))
    expected_purity = np.empty(t)
    for day in range(t):
        w = weights[day] / weights[day].sum()
        x = np.column_stack((np.ones(n), risk[day]))
        gram = x.T @ (w[:, None] * x)
        beta = np.linalg.solve(gram, x.T @ (w * factor[day]))
        fitted = x @ beta
        y_centered = factor[day] - np.sum(w * factor[day])
        residual = factor[day] - fitted
        total = np.sum(w * y_centered**2)
        expected_purity[day] = np.sum(w * residual**2) / total
        risk_centered = risk[day] - np.sum(w[:, None] * risk[day], axis=0)
        sd_risk = np.sqrt(np.sum(w[:, None] * risk_centered**2, axis=0))
        sd_factor = np.sqrt(total)
        expected_loading[day] = beta[1:] * sd_risk / sd_factor

    np.testing.assert_allclose(loading.values, expected_loading, rtol=1e-12, atol=1e-12)
    assert compute_purity_ratio(loading, min_finite=1) == pytest.approx(
        float(expected_purity.mean()), abs=1e-12
    )
    assert loading.weight_ref == "cap-weight"
    assert loading.estimation_scope == "SAME_DATE_DESCRIPTIVE"


def test_portfolio_exposure_is_weight_transpose_risk_not_factor_loading():
    risk = np.array([
        [[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]],
        [[0.5, -2.0], [1.5, 2.0], [4.0, 1.0]],
    ])
    portfolio_weights = np.array([[0.5, -0.25, 0.75], [1.0, 0.0, -0.5]])
    result = compute_portfolio_exposure(
        _panel(risk), portfolio_weights, portfolio_ref="portfolio:known"
    )
    expected = np.einsum("tn,tnk->tk", portfolio_weights, risk)
    np.testing.assert_allclose(result.values, expected, atol=0.0)
    assert result.portfolio_ref == "portfolio:known"
    assert result.estimation_scope == "PORTFOLIO_WEIGHTED_EXPOSURE"


def test_security_panel_refused_as_factor_evidence_without_factor_values():
    panel = _panel(np.ones((2, 12, 2)))
    for reducer in (compute_style_exposure_evidence, compute_industry_exposure, compute_purity_ratio):
        with pytest.raises(TypeError, match="not factor evidence"):
            reducer(panel)


def test_bound_loading_refuses_second_factor_or_weight_binding():
    rng = np.random.default_rng(611)
    risk = rng.normal(size=(2, 15, 2))
    factor = rng.normal(size=(2, 15))
    loading = build_factor_loading_series(_panel(risk), factor)
    assert isinstance(loading, FactorLoadingSeries)
    with pytest.raises(ValueError, match="already binds"):
        compute_style_exposure_evidence(loading, factor_values=factor)
    with pytest.raises(ValueError, match="already binds"):
        compute_style_exposure_evidence(loading, weights=np.ones_like(factor))


def test_constant_factor_purity_is_undefined():
    rng = np.random.default_rng(612)
    risk = rng.normal(size=(3, 20, 2))
    loading = build_factor_loading_series(_panel(risk), np.ones((3, 20)))
    assert np.isnan(compute_purity_ratio(loading, min_finite=1))
