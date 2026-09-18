"""Independent R29 oracles for digital_count and ts_max_buildup repairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _input(backend: str, values: list[float]):
    dates=pd.date_range("2026-01-01",periods=len(values))
    values=[float(value) for value in values]
    if backend=="polars": return pl.DataFrame({"timestamp":dates,"A":values})
    return pd.DataFrame({"A":values},index=dates)


def _values(frame) -> np.ndarray:
    return frame["A"].to_numpy()


@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_digital_count_default_and_capped_oracle(backend: str) -> None:
    values=(100*np.power(1.005,np.arange(32))).tolist()
    op=OperatorRegistry.get("digital_count",backend,mode="any")
    omitted=_values(op.calculate(_input(backend,values)))
    explicit=_values(op.calculate(_input(backend,values),d=20,threshold=.01,run=3))
    expected=np.minimum(np.arange(32),20).astype(float); expected[expected<3]=0
    np.testing.assert_allclose(omitted,expected); np.testing.assert_allclose(explicit,expected)
    assert np.all(omitted[20:]==20), "streak must cap at d, never reset after d"


@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_digital_count_prefix_and_reset(backend: str) -> None:
    values=[100,100.5,101,101.5,102,120,120.5,121,121.5,122]
    op=OperatorRegistry.get("digital_count",backend,mode="any")
    full=_values(op.calculate(_input(backend,values),d=4,threshold=.01,run=2))
    prefix=_values(op.calculate(_input(backend,values[:7]),d=4,threshold=.01,run=2))
    np.testing.assert_allclose(full[:7],prefix); np.testing.assert_allclose(full,[0,0,2,3,4,0,0,2,3,4])


@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_ts_max_buildup_default_window_oracle_and_prefix(backend: str) -> None:
    values=[1,2,1,3,2,4,1,5,2,6,3,7,4,8,5,9,6,10,7,11,8,12,9]
    op=OperatorRegistry.get("ts_max_buildup",backend,mode="any")
    source=_input(backend,values)
    omitted_frame=op.calculate(source)
    omitted=_values(omitted_frame)
    explicit=_values(op.calculate(_input(backend,values),d=20))
    expected=[]
    for end in range(len(values)):
        segment=values[max(0,end-19):end+1]; high=-np.inf; count=0
        for value in segment:
            if value>=high: high=value; count+=1
        expected.append(float(count))
    np.testing.assert_allclose(omitted,expected); np.testing.assert_allclose(explicit,expected)
    prefix=_values(op.calculate(_input(backend,values[:17])))
    np.testing.assert_allclose(omitted[:17],prefix)
    if backend=="polars":
        assert omitted_frame["timestamp"].to_list()==source["timestamp"].to_list()


@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name,kwargs",[
    ("digital_count",{"d":True}),
    ("digital_count",{"d":3.5}),
    ("ts_max_buildup",{"d":True}),
    ("ts_max_buildup",{"d":3.5}),
])
def test_integer_parameters_reject_bool_and_fractional(backend: str,name: str,kwargs: dict[str,object]) -> None:
    op=OperatorRegistry.get(name,backend,mode="any")
    with pytest.raises((TypeError,ValueError)):
        op.calculate(_input(backend,[1.,1.001,1.002,1.003]),**kwargs)
