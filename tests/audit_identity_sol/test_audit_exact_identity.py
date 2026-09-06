import math

import pytest

from factor_engine.planner.canonicalize_params import ParameterCanonicalizer, _round_sig
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.plan_hash import _jsonable, structural_key


def test_exact_cse_key_keeps_nearby_float_parameters_distinct():
    a=PlanNode(op="audit_probe",inputs=[],attrs={"threshold":0.12345678901231})
    b=PlanNode(op="audit_probe",inputs=[],attrs={"threshold":0.12345678901234})
    assert structural_key(a)!=structural_key(b)


def test_exact_cse_key_keeps_large_integer_parameters_distinct():
    a=PlanNode(op="audit_probe",inputs=[],attrs={"seed":1000000000001})
    b=PlanNode(op="audit_probe",inputs=[],attrs={"seed":1000000000002})
    assert structural_key(a)!=structural_key(b)


def test_nonfinite_typed_encoding_cannot_collide_with_strings():
    assert _jsonable(float("nan"))!=_jsonable("__nan__")
    assert _jsonable(float("inf"))!=_jsonable("__inf__")
    assert _jsonable(-float("inf"))!=_jsonable("__ninf__")
    assert _jsonable(float("nan"))!=_jsonable({"type":"float","value":{"__float_special__":"NaN"}})
    assert structural_key(PlanNode(op="literal",attrs={"value":float("nan")}))!=structural_key(PlanNode(op="literal",attrs={"value":{"type":"float","value":{"__float_special__":"NaN"}}}))


def test_research_rounding_handles_subnormal_finite_value():
    out=_round_sig(1e-310,12)
    assert math.isfinite(out) and out==1e-310


def test_research_rounding_preserves_legacy_half_up_for_normal_values():
    value=1.234567890125
    shift=12-int(math.floor(math.log10(abs(value))))-1
    expected=math.floor(value*(10.0**shift)+0.5)/(10.0**shift)
    assert _round_sig(value,12)==expected


def test_unknown_mapping_key_types_fail_closed_instead_of_stringifying():
    with pytest.raises(TypeError,match="string keys"):
        structural_key(PlanNode(op="audit_probe",attrs={"config":{1:"x"}}))


def test_research_approximation_has_a_separate_key_namespace():
    key=ParameterCanonicalizer("audit_probe").hash_key({"threshold":0.12345678901231})
    assert key[0]=="search-equivalence-v1"
