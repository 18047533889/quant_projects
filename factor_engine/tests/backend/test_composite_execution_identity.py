import pytest
from types import FunctionType

from factor_engine.backend.composite_execution_identity import (
    CompositeExecutionIdentityError,
    OLS_EFFECTIVE_RANK_RECIPE,
    build_ols_effective_rank_identity,
)


def test_ols_composite_identity_is_deterministic_and_scoped():
    first = build_ols_effective_rank_identity()
    second = build_ols_effective_rank_identity()

    assert first == second
    assert first.digest and len(first.digest) == 64
    assert first.recipe_identity == OLS_EFFECTIVE_RANK_RECIPE
    assert tuple(name.rsplit(".", 1)[-1] for name, _ in first.callable_digests) == (
        "ols_effective_rank", "_private_name"
    )
    assert {name for name, _ in first.runtime_versions} == {"numpy", "pandas", "python"}
    assert "not a full transitive runtime closure" in first.coverage_scope


def test_live_ols_entrypoint_replacement_changes_identity(monkeypatch):
    from factor_engine.backend import long_neutralization

    before = build_ols_effective_rank_identity()

    def replacement(values, exposures, *, marker=17):
        return marker

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", replacement)
    after = build_ols_effective_rank_identity()

    assert before.digest != after.digest


def test_live_ols_helper_replacement_changes_identity(monkeypatch):
    from factor_engine.backend import long_neutralization

    before = build_ols_effective_rank_identity()

    def replacement(base, used):
        return f"changed_{base}"

    monkeypatch.setattr(long_neutralization, "_private_name", replacement)
    after = build_ols_effective_rank_identity()

    assert before.digest != after.digest


def test_source_relocation_metadata_does_not_change_identity(monkeypatch):
    from factor_engine.backend import long_neutralization

    original = long_neutralization.ols_effective_rank
    relocated_code = original.__code__.replace(
        co_filename="/different/checkout/long_neutralization.py",
        co_firstlineno=original.__code__.co_firstlineno + 500,
        **({"co_linetable": b""} if hasattr(original.__code__, "co_linetable") else {}),
    )
    relocated = FunctionType(
        relocated_code, original.__globals__, original.__name__,
        original.__defaults__, original.__closure__,
    )
    relocated.__module__ = original.__module__
    relocated.__qualname__ = original.__qualname__
    relocated.__annotations__ = original.__annotations__
    relocated.__kwdefaults__ = original.__kwdefaults__
    before = build_ols_effective_rank_identity()

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", relocated)

    assert before.digest == build_ols_effective_rank_identity().digest


def test_immutable_default_and_closure_changes_participate_in_identity(monkeypatch):
    from factor_engine.backend import long_neutralization

    def make_replacement(default_value, closure_value):
        def replacement(values, exposures, marker=default_value):
            return marker, closure_value
        return replacement

    before = build_ols_effective_rank_identity()
    monkeypatch.setattr(
        long_neutralization, "ols_effective_rank", make_replacement(1, "a")
    )
    first = build_ols_effective_rank_identity()
    monkeypatch.setattr(
        long_neutralization, "ols_effective_rank", make_replacement(2, "a")
    )
    changed_default = build_ols_effective_rank_identity()
    monkeypatch.setattr(
        long_neutralization, "ols_effective_rank", make_replacement(2, "b")
    )
    changed_closure = build_ols_effective_rank_identity()

    assert before.digest != first.digest
    assert first.digest != changed_default.digest
    assert changed_default.digest != changed_closure.digest


def test_unrelated_module_callable_does_not_change_scoped_identity(monkeypatch):
    import math

    before = build_ols_effective_rank_identity()
    monkeypatch.setattr(math, "sqrt", lambda value: value)

    assert before.digest == build_ols_effective_rank_identity().digest


def test_numpy_and_pandas_versions_participate_in_identity(monkeypatch):
    from factor_engine.backend import long_neutralization

    before = build_ols_effective_rank_identity()
    monkeypatch.setattr(long_neutralization.np, "__version__", "test-numpy-version")
    after_numpy = build_ols_effective_rank_identity()
    monkeypatch.setattr(long_neutralization.pd, "__version__", "test-pandas-version")
    after_pandas = build_ols_effective_rank_identity()

    assert before.digest != after_numpy.digest
    assert after_numpy.digest != after_pandas.digest


def test_unsupported_callable_state_fails_closed(monkeypatch):
    from factor_engine.backend import long_neutralization

    mutable_default = []

    def unsupported(values, exposures, state=mutable_default):
        return state

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", unsupported)
    with pytest.raises(CompositeExecutionIdentityError, match="unsupported dynamic state"):
        build_ols_effective_rank_identity()


def test_class_default_fails_closed_as_dynamic_state(monkeypatch):
    from factor_engine.backend import long_neutralization

    class MutableConfiguration:
        pass

    def unsupported(values, exposures, state=MutableConfiguration):
        return state

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", unsupported)
    with pytest.raises(CompositeExecutionIdentityError, match="unsupported dynamic state"):
        build_ols_effective_rank_identity()


def test_class_keyword_default_fails_closed_as_dynamic_state(monkeypatch):
    from factor_engine.backend import long_neutralization

    class MutableConfiguration:
        pass

    def unsupported(values, exposures, *, state=MutableConfiguration):
        return state

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", unsupported)
    with pytest.raises(CompositeExecutionIdentityError, match="unsupported dynamic state"):
        build_ols_effective_rank_identity()


def test_non_python_helper_fails_closed(monkeypatch):
    from factor_engine.backend import long_neutralization

    monkeypatch.setattr(long_neutralization, "_private_name", len)
    with pytest.raises(CompositeExecutionIdentityError, match="Python function"):
        build_ols_effective_rank_identity()
