import numpy as np
import pytest

from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_batch


def test_gpu_adapter_requests_daily_artifact_and_explicit_ten_bins(monkeypatch):
    from types import SimpleNamespace
    import quant_evaluator.runtime.evaluator as runtime
    def fake_evaluate(fb, lb, **kwargs):
        assert kwargs['metrics'] == ('rank_ic_series', 'quantile_returns_daily')
        assert kwargs['metric_parameters']['quantile_returns_daily'] == {'n_quantiles': 10, 'min_assets': 20}
        return SimpleNamespace(series_metrics={'rank_ic_series': np.full((3, 2), .1)},
            artifacts={'quantile_returns_daily': SimpleNamespace(
                values=np.zeros((3, 10, 2)), factor_axis=('a', 'b'))})
    monkeypatch.setattr(runtime, 'evaluate', fake_evaluate)
    result = evaluate_report_batch(np.arange(120.).reshape(3,20,2),
        np.zeros((3,20)), factor_ids=('a','b'), backend='cuda_strict',
        min_assets=20, min_ic_periods=2)
    assert result.backend_used == 'cuda'
    assert result.factors['a'].quantile_returns.shape == (3,10)


def test_gpu_strict_rejects_noncanonical_quantile_count():
    with pytest.raises(ValueError, match='n_quantiles=10'):
        evaluate_report_batch(
            np.arange(120, dtype=float).reshape(3, 20, 2),
            np.arange(60, dtype=float).reshape(3, 20) / 100,
            factor_ids=('a', 'b'), backend='cuda_strict',
            n_quantiles=5, min_assets=20,
        )


def test_gpu_auto_preserves_requested_quantile_count():
    result = evaluate_report_batch(
        np.arange(120, dtype=float).reshape(3, 20, 2),
        np.arange(60, dtype=float).reshape(3, 20) / 100,
        factor_ids=('a', 'b'), backend='auto',
        n_quantiles=5, min_assets=20, min_ic_periods=2,
    )
    assert result.backend_used == 'cpu'
    assert 'n_quantiles=10' in result.backend_fallback_reason
    assert result.factors['a'].quantile_returns.shape == (3, 5)
