"""Unspecified dual-kernel panel roles must not become scalar declarations."""
import pandas as pd
import pytest

def test_real_spectral_kernel_has_panel_contract_and_compiles():
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    parser,engine=build_runtime()
    name="ts_rolling_sr_gaussian_mean_shift_score"
    for backend in ("pandas_numpy","polars"):
        meta=OperatorRegistry.get(name,backend,mode="any").metadata
        assert meta.panel_params==("x",)
        assert "x" not in meta.scalar_params
        assert "window" in meta.scalar_params
    for formula in (f"{name}(ret, window=120)",f"{name}(x=ret, window=120)"):
        engine.compile(Factor(name="test_roles",expr=parser.parse(formula),
                              source_expr=formula,surface="compat_research"))
    with pytest.raises((TypeError,ValueError,NotImplementedError)):
        formula=f"{name}(ret, window=close)"
        engine.compile(Factor(name="test_bad",expr=parser.parse(formula),
                              source_expr=formula,surface="compat_research"))

def test_public_signature_inference_preserves_required_scalars():
    from types import SimpleNamespace
    from factor_engine.cleaned_operators.operator_spec import _infer_panel_params
    from factor_engine.cleaned_operators.base import ParamSpec
    def kernel(x: pd.DataFrame, window: int, y: pd.DataFrame=None):
        return x
    meta=SimpleNamespace(panel_params=(),param_names=("x","window","y"),
                         input_fields=(),param_specs={"window":ParamSpec(dtype=int)})
    assert _infer_panel_params(SimpleNamespace(_contract_callable=kernel),meta,{})==("x","y")
