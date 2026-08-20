
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "factor_preprocess" / "build" / "lib"))
import warnings
from factor_preprocess.contracts.policy import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
from factor_preprocess.registry.policies import PolicyPreset, PolicyLevel, TransformStep, get_default_policy_registry


def test_duplicate_transform_names_allowed():
    spec1 = TransformSpec(name="cs_rank", kind=TransformKind.CROSS_SECTIONAL, mode=TransformMode.STATELESS, version="1.0.0")
    spec2 = TransformSpec(name="cs_rank", kind=TransformKind.CROSS_SECTIONAL, mode=TransformMode.STATELESS, version="1.0.0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        policy = PreprocessingPolicy(policy_id="p1", transforms=[spec1, spec2], version="1.0.0")
    assert len(policy.transforms) == 2


def test_duplicate_step_ids_rejected():
    p = PolicyPreset(name="dup", description="dup", level=PolicyLevel.RESEARCH, steps=[TransformStep(name="cs_rank", parameters={"pct": True}), TransformStep(name="cs_rank", parameters={"pct": True})], causal_safe=True)
    valid, errors = p.validate()
    assert not valid
    assert any("Duplicate step_id" in e for e in errors)


def test_step_id_uniqueness_enforced():
    from dataclasses import replace
    step_a = TransformStep(name="cs_rank", parameters={"pct": True}, step_id="rank_a")
    step_b = replace(step_a, step_id="rank_b")
    p = PolicyPreset(name="dup", description="dup", level=PolicyLevel.RESEARCH, steps=[step_a, step_b], causal_safe=True)
    valid, errors = p.validate()
    assert valid, errors


def test_production_full_neutralization_skip_if_missing_is_false():
    registry = get_default_policy_registry()
    prod = registry.get("production_full")
    neutralize_step = next(step for step in prod.steps if step.name == "ols_neutralize")
    assert neutralize_step.skip_if_missing is False

