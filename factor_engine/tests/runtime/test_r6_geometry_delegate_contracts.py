"""Numerical and labeled-axis parity for actual final geometry delegates."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.contracts import ExecutionKind

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

CASES=[
    ("ts_interval_union_coverage",2,{"window":20}),
    ("ts_binned_response_monotonicity",2,{"window":60,"bins":5}),
    ("ts_binned_response_curvature",2,{"window":60,"bins":5}),
    ("ts_vector_path_efficiency",2,{"window":20}),
    ("ts_spectral_centroid",1,{"window":32}),
    ("event_fano_factor",1,{"window":60,"block":10}),
]
EXPECTED_EXECUTION_KIND = {
    # Exact two-panel NumPy geometry running inside the Polars backend; this is
    # no longer a pandas delegate and must remain classified truthfully.
    "ts_vector_path_efficiency": ExecutionKind.POLARS_NUMPY_KERNEL,
}
@pytest.mark.parametrize("name,n_panels,params",CASES)
def test_final_delegate_preserves_numerics_scalar_binding_and_dates(name,n_panels,params):
    rng=np.random.default_rng(453)
    x=pd.DataFrame(rng.normal(size=(100,3)).cumsum(axis=0)+30,
                   index=pd.date_range("2025-01-01",periods=100),columns=["A","B","C"])
    y=x+2 if "interval" in name else x*.3+pd.DataFrame(rng.normal(size=(100,3)),index=x.index,columns=x.columns)
    if name=="event_fano_factor":
        x=(x>x.rolling(5,min_periods=1).mean()).astype(float)
    pandas=get=lambda backend:OperatorRegistry.get(name,backend,mode="research")
    reference=get("pandas_numpy")
    op=get("polars")
    expected_kind = EXPECTED_EXECUTION_KIND.get(name, ExecutionKind.POLARS_PANDAS_DELEGATE)
    assert op._physical_spec.execution_kind == expected_kind
    panel_names=tuple(reference.metadata.param_names[:n_panels])
    frames=dict(zip(panel_names,[x,y][:n_panels]))
    convert=lambda frame:pl.from_pandas(frame.rename_axis("date").reset_index())
    inputs={key:convert(frame) for key,frame in frames.items()}
    expected=reference.calculate(*frames.values(),**params)
    for call in (lambda:op.calculate(**inputs,**params),
                 lambda:op.calculate(*inputs.values(),**params),
                 lambda:op.calculate(*inputs.values(),*params.values())):
        actual=call().to_pandas().set_index("date").rename_axis(None)
        pd.testing.assert_frame_equal(actual,expected,check_freq=False,atol=1e-10,rtol=1e-10)
    assert np.isfinite(expected.to_numpy()).any()
    short=op.calculate(**{k:convert(v.iloc[:80]) for k,v in frames.items()},**params)
    pd.testing.assert_frame_equal(short.to_pandas().set_index("date").rename_axis(None),expected.iloc[:80],
                                  check_freq=False,atol=1e-10,rtol=1e-10)
