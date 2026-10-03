from types import SimpleNamespace
import pandas as pd
import polars as pl
import pytest
from factor_engine.scripts.native_polars_parameter_points import certify_selected_point
from factor_engine.runtime.parameter_domain_store import ParameterDomainCertificationStore

def _fixture():
    from factor_engine.cleaned_operators import load_all
    load_all()
    ref = pd.DataFrame({"A": [1., 2., 4., 8., 16.]})
    return pl.DataFrame({"A": ref.A.to_list()}), ref

def test_fresh_native_point_binds_actual_variant_dtype_and_defaults():
    native, ref = _fixture()
    store = ParameterDomainCertificationStore()
    key = certify_selected_point("ts_mean", [native], [ref], {"window": 2},
                                 context=SimpleNamespace(data_source=None, grain="daily"),
                                 store=store)
    assert key.backend == "polars"
    assert key.execution_variant != "reference"
    assert key.source_context == "memory"
    assert dict(key.parameter_point) == {"window": 2, "min_periods": 1}
    assert store.exact_call_is_certified(
        "ts_mean", dict(key.parameter_point), backend="polars",
        semantic_version=key.semantic_version, execution_variant=key.execution_variant,
        source_context=key.source_context, dtype=key.dtype, grain=key.grain)
    assert not store.operator_has_any_certified_region_by_backend("ts_mean", "pandas_numpy")

def test_mismatched_reference_inputs_cannot_publish_native_point():
    native, ref = _fixture()
    store = ParameterDomainCertificationStore()
    ref.iloc[0, 0] = 99.
    with pytest.raises(ValueError, match="input values differ"):
        certify_selected_point("ts_mean", [native], [ref], {"window": 2},
                               context=SimpleNamespace(data_source=None), store=store)
    assert not store.operator_has_any_certified_region_by_backend("ts_mean", "polars")


@pytest.mark.parametrize("rtol", [float("nan"), float("inf"), -1., 1.])
def test_relaxed_or_nonfinite_tolerance_cannot_grant_evidence(rtol):
    native, ref = _fixture()
    store = ParameterDomainCertificationStore()
    with pytest.raises(ValueError, match="tolerances"):
        certify_selected_point("ts_mean", [native], [ref], {"window": 2},
                               context=SimpleNamespace(data_source=None),
                               store=store, rtol=rtol)
    assert not store.operator_has_any_certified_region_by_backend("ts_mean", "polars")


def test_certificate_sampling_is_bounded_before_materialization(monkeypatch):
    native, ref = _fixture()
    from factor_engine.scripts import native_polars_parameter_points as module
    monkeypatch.setattr(module, "MAX_CERTIFICATION_CELLS", 4)
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized sample materialized")
    monkeypatch.setattr(pl.DataFrame, "to_numpy", forbidden)
    store = ParameterDomainCertificationStore()
    with pytest.raises(ValueError, match="bounded cell limit"):
        certify_selected_point("ts_mean", [native], [ref], {"window": 2},
                               context=SimpleNamespace(data_source=None), store=store)
    assert not store.operator_has_any_certified_region_by_backend("ts_mean", "polars")


@pytest.mark.parametrize("representation", ["panel", "series", "polars"])
def test_runtime_point_uses_bridge_input_signature_not_prepared_dtype(representation):
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.backend.cleaned_bridge import _input_dtype, _prepare_call_args
    from factor_engine.backend.panel_polars import panel_to_polars
    from factor_engine.scripts.native_polars_parameter_points import certify_runtime_point
    _fixture()
    panel = pd.DataFrame({"A": [1., 2., 4., 8., 16.]},
                         index=pd.date_range("2024-01-01", periods=5, name="timestamp"))
    panel.columns.name = "instrument"
    if representation == "panel":
        evaluated = [panel]
    elif representation == "series":
        evaluated = [panel.stack().rename("close")]
    else:
        evaluated = [panel_to_polars(panel)]
    context = ExecutionContext(data_source=None)
    store = ParameterDomainCertificationStore()
    key = certify_runtime_point("ts_mean", evaluated, {"window": 2},
                                context=context, store=store)
    assert key.dtype == _input_dtype(evaluated).to_key()
    if representation != "polars":
        prepared, _, _ = _prepare_call_args(evaluated, context, backend="polars")
        assert key.dtype != _input_dtype(prepared).to_key()
    assert key.source_context == "memory"
    assert dict(key.parameter_point) == {"window": 2, "min_periods": 1}


def test_shifted_reference_time_axis_cannot_publish_point():
    from factor_engine.backend.panel_polars import panel_to_polars
    _, ref = _fixture()
    ref.index = pd.date_range("2024-01-01", periods=5)
    native = panel_to_polars(ref)
    ref.index = ref.index + pd.Timedelta(days=1)
    store = ParameterDomainCertificationStore()
    with pytest.raises(ValueError, match="input coordinates differ"):
        certify_selected_point("ts_mean", [native], [ref], {"window": 2},
                               context=SimpleNamespace(data_source=None), store=store)
    assert not store.operator_has_any_certified_region_by_backend("ts_mean", "polars")
