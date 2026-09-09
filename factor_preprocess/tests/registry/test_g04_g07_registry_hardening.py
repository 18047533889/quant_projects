import math

import pytest

from factor_preprocess.errors import GovernanceError
from factor_preprocess.registry.transforms import (
    TransformCategory,
    TransformRegistry,
    create_default_registry,
)


def _bounded(values, alpha=0.5):
    return alpha


def test_default_registry_does_not_extend_builtin_admission_to_plugins():
    registry = create_default_registry()
    registry.register("external", _bounded, TransformCategory.TEMPORAL)
    assert registry.get("external").admission == "RESEARCH_ONLY"
    with pytest.raises(ValueError, match="RESEARCH_ONLY"):
        registry.validate_production("external")
    registry.validate_production("cs_rank")


@pytest.mark.parametrize("args, kwargs", [
    ((None, 2.0), {}),
    ((None,), {"alpha": math.nan}),
    ((None,), {"alpha": math.inf}),
])
def test_runtime_domain_validates_positional_and_nonfinite(args, kwargs):
    registry = TransformRegistry()
    registry.register(
        "bounded", _bounded, TransformCategory.TEMPORAL,
        parameter_domain={"alpha": (0.0, 1.0)},
    )
    with pytest.raises(ValueError):
        registry.get_execution("bounded")(*args, **kwargs)
    assert registry.get_execution("bounded")(None, 0.25) == 0.25


def test_implementation_identity_is_independent_of_compile_path():
    one = {}
    two = {}
    exec(compile("def f(x):\n    return x + 1\n", "/main/pkg/mod.py", "exec"), one)
    exec(compile("def f(x):\n    return x + 1\n", "/wheel/pkg/mod.py", "exec"), two)
    r1 = TransformRegistry()
    r2 = TransformRegistry()
    r1.register("f", one["f"], TransformCategory.TEMPORAL)
    r2.register("f", two["f"], TransformCategory.TEMPORAL)
    assert r1.get("f").implementation_hash == r2.get("f").implementation_hash


def test_nested_mutable_closure_and_mutable_default_are_rejected():
    state = ([1.0],)

    def closure(values):
        return state[0][0]

    def default(values, cache=[]):
        return cache

    registry = TransformRegistry()
    with pytest.raises(ValueError, match="mutable closure"):
        registry.register("closure", closure, TransformCategory.TEMPORAL)
    with pytest.raises(ValueError, match="default state"):
        registry.register("default", default, TransformCategory.TEMPORAL)


def test_helper_drift_is_rejected_before_execution():
    namespace = {}
    exec("def helper(x): return x + 1\ndef outer(x): return helper(x)\n", namespace)
    registry = TransformRegistry()
    registry.register("outer", namespace["outer"], TransformCategory.TEMPORAL)
    executor = registry.get_execution("outer")
    assert executor(1) == 2
    exec("def helper(x): return x + 2\n", namespace)
    with pytest.raises(GovernanceError, match="changed after registration"):
        executor(1)
