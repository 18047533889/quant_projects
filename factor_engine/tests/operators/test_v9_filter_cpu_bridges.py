"""Winner and alternate-startup tests for Bessel/Butterworth CPU bridges."""
from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import (
    PolarsImplementationKind,
    polars_backend_kind,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


CASES = (
    ("ts_bessel_lowpass_causal", {"order": 4, "cutoff": 0.05}),
    ("ts_butterworth_lowpass_causal", {"cutoff_period": 20, "order": 2}),
)


def _frames():
    rng = np.random.default_rng(2727)
    values = rng.normal(size=(90, 2)) + 100.0
    values[20:25, 0] = np.nan
    pandas_frame = pd.DataFrame(values, columns=["A", "B"])
    polars_frame = pl.from_pandas(pandas_frame)
    return pandas_frame, polars_frame


def _assert_reference_parity(name, polars_op, params):
    pandas_frame, polars_frame = _frames()
    reference = OperatorRegistry.get(name, "pandas_numpy", mode="research")
    expected = reference.calculate(pandas_frame, **params)
    actual = polars_op.calculate(x=polars_frame, **params)
    assert actual.columns == polars_frame.columns
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=0, atol=0)


def test_01_explicit_batch1_classes_preserve_contract_and_reference_parity():
    code = r'''\
import hashlib, inspect, numpy as np, pandas as pd, polars as pl
import factor_engine.cleaned_operators.polars_native.ts_advanced_batch1 as batch1
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
classes={
 "ts_bessel_lowpass_causal":batch1.TSBesselLowpassCausalPolarsNative,
 "ts_butterworth_lowpass_causal":batch1.TSButterworthLowpassCausalPolarsNative,
}
cases=(("ts_bessel_lowpass_causal",{"order":4,"cutoff":.05}),("ts_butterworth_lowpass_causal",{"cutoff_period":20,"order":2}))
expected_names={"ts_bessel_lowpass_causal":["x","order","cutoff"],"ts_butterworth_lowpass_causal":["x","cutoff_period","order"]}
helper_source=inspect.getsource(_call_pandas_delegate)
for name,params in cases:
 op=classes[name]()
 assert op.metadata.param_names == expected_names[name]
 assert op._physical_spec.execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE
 assert op._physical_spec.validation_errors() == ()
 assert op._physical_spec.is_production_eligible() is False
 digest=hashlib.sha256((helper_source+inspect.getsource(type(op)._calculate_series)).encode()).hexdigest()
 assert op._physical_spec.implementation_source_hash == digest
ensure_cleaned_loaded()
rng=np.random.default_rng(2727); values=rng.normal(size=(90,2))+100; values[20:25,0]=np.nan
pdf=pd.DataFrame(values,columns=["A","B"]); plf=pl.from_pandas(pdf)
for name,params in cases:
 op=classes[name](); ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
 np.testing.assert_allclose(op.calculate(x=plf,**params).to_numpy(),ref.calculate(pdf,**params).to_numpy(),equal_nan=True,rtol=0,atol=0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ), text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr


def test_02_default_startup_real_winners_are_non_native_and_match_reference():
    code = r'''\
import numpy as np, pandas as pd, polars as pl
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.polars_backend_kind import PolarsImplementationKind, polars_backend_kind
from factor_engine.cleaned_operators.registry import OperatorRegistry
ensure_cleaned_loaded()
rng=np.random.default_rng(2727); values=rng.normal(size=(90,2))+100; values[20:25,0]=np.nan
pdf=pd.DataFrame(values,columns=["A","B"]); plf=pl.from_pandas(pdf)
cases=(("ts_bessel_lowpass_causal",{"order":4,"cutoff":.05}),("ts_butterworth_lowpass_causal",{"cutoff_period":20,"order":2}))
for name,params in cases:
 op=OperatorRegistry.get(name,"polars",mode="research"); ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
 assert polars_backend_kind(op,production_mode=False) != PolarsImplementationKind.POLARS_NATIVE
 np.testing.assert_allclose(op.calculate(x=plf,**params).to_numpy(),ref.calculate(pdf,**params).to_numpy(),equal_nan=True,rtol=0,atol=0)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ), text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr


def test_bridge_defaults_and_parameter_domain_match_references():
    import factor_engine.cleaned_operators.polars_native.ts_advanced_batch1 as batch1

    ensure_cleaned_loaded()
    pandas_frame, polars_frame = _frames()
    for name, cls in (
        ("ts_bessel_lowpass_causal", batch1.TSBesselLowpassCausalPolarsNative),
        ("ts_butterworth_lowpass_causal", batch1.TSButterworthLowpassCausalPolarsNative),
    ):
        reference = OperatorRegistry.get(name, "pandas_numpy", mode="research")
        actual = cls().calculate(x=polars_frame)
        expected = reference.calculate(pandas_frame)
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=0, atol=0)

    with pytest.raises(Exception):
        batch1.TSButterworthLowpassCausalPolarsNative().calculate(x=polars_frame, cutoff_period=2)
    with pytest.raises(Exception):
        batch1.TSButterworthLowpassCausalPolarsNative().calculate(x=polars_frame, order=11)
    with pytest.raises(Exception):
        batch1.TSBesselLowpassCausalPolarsNative().calculate(x=polars_frame, cutoff=0.5)
    with pytest.raises(Exception):
        batch1.TSBesselLowpassCausalPolarsNative().calculate(x=polars_frame, order=9)
