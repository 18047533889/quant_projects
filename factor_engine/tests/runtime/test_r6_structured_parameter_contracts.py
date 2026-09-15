"""Opt-in structured parameters: shared planning/runtime/hash validation and JSON Schema."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.base import ParamSpec,ParamRole,OperatorMetadata,SeriesOperator
from factor_engine.cleaned_operators.common.strict_params import normalize_and_validate_scalar_param as validate
from factor_engine.runtime.operator_snapshot import _parameter_contract
PAIR=ParamSpec(dtype=tuple,items=ParamSpec(dtype=float,min=0.,max=1.),min_items=2,max_items=2,
               default=(.25,.75),param_role=ParamRole.ECONOMIC)
UNION=ParamSpec(alternatives=(ParamSpec(dtype=str),ParamSpec(dtype=float)),
               default="quantile",param_role=ParamRole.STATE_THRESHOLD)
class Example(SeriesOperator):
    metadata=OperatorMetadata(name="structured_contract_example",category="test",
        param_names=["x","interval","threshold"],panel_params=("x",),scalar_params=("interval","threshold"),
        param_specs={"interval":PAIR,"threshold":UNION})
    def _calculate_series(self,x,interval=(.25,.75),threshold="quantile"):
        assert isinstance(interval,tuple)
        if isinstance(threshold,str):
            if threshold.lower()!="quantile":
                raise ValueError("unknown threshold mode")
            scale=1.
        else:
            scale=threshold
        return x*(interval[1]-interval[0])*scale

@pytest.mark.parametrize("phase",["planning","runtime","hash"])
@pytest.mark.parametrize("container",[list,tuple,np.array])
def test_typed_pair_same_validation_all_phases(phase,container):
    assert validate("example","interval",container([.1,.9]),phase=phase,spec=PAIR)==(.1,.9)
    for bad in ([.1],[.1,.2,.3],[True,.2],[np.inf,.2],[-.1,.2],[[.1],.2],{"a":1},np.ones((2,2))):
        with pytest.raises((TypeError,ValueError)):
            validate("example","interval",bad,phase=phase,spec=PAIR)

@pytest.mark.parametrize("phase",["planning","runtime","hash"])
def test_union_retains_string_number_without_bool_or_nonfinite(phase):
    for value in ("quantile","QUANTILE",3.,-2,np.float64(.5)):
        assert validate("example","threshold",value,phase=phase,spec=UNION)==value
    for value in (True,False,np.nan,np.inf,[],{},np.array([1.])):
        with pytest.raises((TypeError,ValueError)):
            validate("example","threshold",value,phase=phase,spec=UNION)

def test_legacy_untyped_vectors_are_not_reinterpreted():
    value=["legacy",{"parameter":1}]
    assert validate("legacy","vector",value,spec=ParamSpec(dtype=tuple)) is value

def test_snapshot_exposes_real_array_and_union_shape():
    schema,_,verified=_parameter_contract(Example(),("x",))
    assert verified
    pair=schema["properties"]["interval"]
    assert pair["type"]=="array" and pair["minItems"]==pair["maxItems"]==2
    assert pair["items"]["type"]=="number" and pair["items"]["minimum"]==0.
    union=schema["properties"]["threshold"]
    assert {p["type"] for p in union["anyOf"]}=={"string","number"}
    bad=Example()
    bad.metadata=OperatorMetadata(name="unknown_sequence",category="test",
        param_names=["x","interval"],panel_params=("x",),scalar_params=("interval",),
        param_specs={"interval":ParamSpec(dtype=tuple,items=ParamSpec())})
    assert not _parameter_contract(bad,("x",))[2]

def test_real_operator_binds_structured_kwargs_and_positionals():
    x=pd.DataFrame({"A":[1.,2.]},index=pd.date_range("2025-01-01",periods=2))
    op=Example()
    pd.testing.assert_frame_equal(op.calculate(x,[.1,.9],2.),x*1.6)
    pd.testing.assert_frame_equal(op.calculate(x=x,interval=(.1,.9),threshold="QUANTILE"),x*.8)
    for pair in ([.1], [.1,.2,.3], [True,.9]):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(x,interval=pair)

def test_invalid_structured_declarations_fail_at_construction():
    for kwargs in (
        {"alternatives":()}, {"alternatives":(ParamSpec(dtype=float),),"dtype":float},
        {"dtype":float,"items":ParamSpec(dtype=float)},
        {"dtype":tuple,"min_items":2},
        {"dtype":tuple,"items":ParamSpec(dtype=float),"min_items":3,"max_items":2},
    ):
        with pytest.raises(ValueError):
            ParamSpec(**kwargs)
