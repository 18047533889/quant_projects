"""Converted pandas-frame kernels are not native Polars expressions."""
import pytest


@pytest.mark.parametrize("canonical", [
    "ts_cross_spectral_coherence", "ts_cross_spectral_phase",
    "ts_bicoherence_top_decile_mean", "ts_bicoherence_top_decile_excess",
])
def test_actual_dual_bridge_is_not_research_native(canonical):
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.polars_backend_kind import (
        polars_backend_kind, PolarsImplementationKind,
    )
    ensure_cleaned_loaded()
    op = OperatorRegistry.get(canonical, "polars", mode="research")
    assert op is not None
    assert type(op).__module__ == "factor_engine.cleaned_operators.gemini_v2_common"
    assert polars_backend_kind(op, production_mode=False) == PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE
    # Recognizing an actual delegate does not grant it a production certificate.
    assert polars_backend_kind(op, production_mode=True) == PolarsImplementationKind.UNSUPPORTED
