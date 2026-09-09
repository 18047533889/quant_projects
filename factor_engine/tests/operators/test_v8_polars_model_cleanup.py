from factor_engine.cleaned_operators.ts_model import polars_regression


def test_obsolete_unregistered_slope_helpers_are_removed():
    assert not hasattr(polars_regression, "_ols_beta")
    assert not hasattr(polars_regression, "_rolling_slope")


def test_callback_metadata_does_not_claim_native_execution():
    metadata = polars_regression._meta("ts_variance_ratio_slope", "test", [])
    assert "native" not in metadata.tags
    assert "python_callback" in metadata.tags
    native = polars_regression._meta("ts_market_liquidity_beta", "test", [])
    assert "native" in native.tags
    assert "python_callback" not in native.tags
