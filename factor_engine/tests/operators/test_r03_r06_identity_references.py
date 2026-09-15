from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import numpy as np
import pytest

from factor_engine.cleaned_operators.math_certificate import (
    _rank_pct_rowwise_np,
    _rolling_cov_ref,
)
from factor_engine.cleaned_operators.registry import (
    _fn_payload,
    _freeze_value,
    _impl_source_hash,
)


def _twice(x):
    return 2 * x


class _WrappedPlusOne:
    _fn = staticmethod(_twice)

    def calculate(self, x):
        return self._fn(x) + 1


class _WrappedPlusTwo:
    _fn = staticmethod(_twice)

    def calculate(self, x):
        return self._fn(x) + 2


def test_wrapper_and_wrapped_kernel_are_both_identity_dependencies():
    assert _impl_source_hash(_WrappedPlusOne()) != _impl_source_hash(_WrappedPlusTwo())


def test_referenced_global_value_changes_payload_without_scanning_unreferenced_globals():
    namespace = {"GAIN": 2, "UNUSED": list(range(100_000))}
    exec("def kernel(x):\n    return x * GAIN\n", namespace)
    fn = namespace["kernel"]
    first = _fn_payload(fn, object)
    namespace["GAIN"] = 3
    second = _fn_payload(fn, object)
    assert first != second
    assert len(first) < 4096
    assert "UNUSED" not in first


def test_arbitrary_module_name_cannot_bypass_user_helper_closure_identity():
    module = types.ModuleType("user_ops")
    exec(
        "def make(gain):\n"
        "    def helper(x): return x * gain\n"
        "    return helper\n"
        "def outer(x): return helper(x)\n",
        module.__dict__,
    )
    sys.modules[module.__name__] = module
    try:
        module.helper = module.make(2)
        first = _fn_payload(module.outer, object)
        module.helper = module.make(3)
        second = _fn_payload(module.outer, object)
    finally:
        sys.modules.pop(module.__name__, None)
    assert first != second


def test_arbitrary_module_name_cannot_hide_callable_instance_state():
    class UserCallable:
        __module__ = "user_ops"

        def __init__(self, gain):
            self.gain = gain

        def __call__(self, x):
            return x * self.gain

    namespace = {"helper": UserCallable(2)}
    exec("def outer(x): return helper(x)\n", namespace)
    assert _fn_payload(namespace["outer"], object) is None


def test_unversioned_sourceless_module_is_explicitly_uncertifiable():
    from factor_engine.cleaned_operators.registry import (
        UncertifiableImplementationIdentity,
        _module_dependency_payload,
    )

    module = types.ModuleType("runtime_generated_user_module")
    with pytest.raises(UncertifiableImplementationIdentity, match="version.*source"):
        _module_dependency_payload(module)


def test_uncertifiable_candidate_can_be_recorded_but_not_certified():
    from factor_engine.cleaned_operators.registry import (
        UncertifiableImplementationIdentity,
        _registration_impl_hash,
    )

    class UserCallable:
        def __call__(self, value):
            return value

    helper = UserCallable()

    class Candidate:
        def calculate(self, value):
            return helper(value)

    with pytest.raises(UncertifiableImplementationIdentity):
        _impl_source_hash(Candidate())
    assert _registration_impl_hash(Candidate()) is None


class _MutableCallable:
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, x):
        return x * self.gain


def test_python_callable_instance_without_declared_identity_fails_closed():
    with pytest.raises(TypeError, match="semantic_identity"):
        _freeze_value(_MutableCallable(2))


@dataclass(frozen=True)
class _Multiplier:
    gain: int

    def apply(self, x):
        return x * self.gain


def test_bound_method_binds_auditable_self_state():
    assert _fn_payload(_Multiplier(2).apply, object) != _fn_payload(_Multiplier(3).apply, object)


def test_rank_pct_reference_matches_pandas_inf_contract():
    actual = _rank_pct_rowwise_np(np.array([[1.0, 2.0, np.inf, np.nan]]))
    np.testing.assert_allclose(actual[0, :3], [1 / 3, 2 / 3, 1.0])
    assert np.isnan(actual[0, 3])


def test_rolling_cov_honours_min_periods_and_ddof():
    x = np.arange(1.0, 7.0)
    actual = _rolling_cov_ref(x, 2 * x, window=5, min_periods=5, ddof=1)
    assert np.isnan(actual[:4]).all()
    np.testing.assert_allclose(actual[4:], [5.0, 5.0])


def test_rolling_cov_rejects_ambiguous_shape_and_parameters():
    with pytest.raises(ValueError, match="equal-length"):
        _rolling_cov_ref(np.ones(3), np.ones(2), window=2)
    with pytest.raises(ValueError, match="1-D"):
        _rolling_cov_ref(np.ones((3, 1)), np.ones((3, 1)), window=2)
    with pytest.raises(ValueError, match="min_periods"):
        _rolling_cov_ref(np.ones(3), np.ones(3), window=2, min_periods=3)
    with pytest.raises(ValueError, match="ddof"):
        _rolling_cov_ref(np.ones(3), np.ones(3), window=2, ddof=-1)
