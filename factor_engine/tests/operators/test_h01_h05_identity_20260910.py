from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ExecutionVariantIdentity
from factor_engine.cleaned_operators.registry import (
    _code_payload,
    _deepfreeze_catalog,
    _defrost_catalog,
    _freeze_value,
    _impl_source_hash,
)


def test_h01_resolves_encoded_name_operands() -> None:
    def left(x):
        return np.sin(x)

    def right(x):
        return np.cos(x)

    assert _code_payload(left.__code__, include_names=True) != _code_payload(
        right.__code__, include_names=True
    )
    assert any(i.opname == "LOAD_GLOBAL" and "NULL +" in i.argrepr for i in __import__("dis").get_instructions(left))


def test_h01_exception_table_changes_code_identity() -> None:
    def guarded(x):
        try:
            return 1 / x
        except ZeroDivisionError:
            return 0

    def unguarded(x):
        return 1 / x

    assert guarded.__code__.co_exceptiontable
    assert _code_payload(guarded.__code__, include_names=True) != _code_payload(
        unguarded.__code__, include_names=True
    )


def test_h02_defaults_kwdefaults_and_named_closures_are_identity() -> None:
    def factory(value, *, swap=False):
        first, second = (0, value) if swap else (value, 0)

        def kernel(x=3, *, window=5):
            return x + first - second + window

        return kernel

    base = factory(2)
    changed_default = factory(2)
    changed_default.__defaults__ = (4,)
    changed_kwdefault = factory(2)
    changed_kwdefault.__kwdefaults__ = {"window": 6}
    assert _freeze_value(base) != _freeze_value(changed_default)
    assert _freeze_value(base) != _freeze_value(changed_kwdefault)
    assert _freeze_value(base) != _freeze_value(factory(2, swap=True))


def test_h02_cyclic_capture_fails_closed() -> None:
    captured = []
    captured.append(captured)
    with pytest.raises(ValueError, match="cyclic semantic identity"):
        _freeze_value(captured)


def test_h02_cyclic_closure_error_is_not_swallowed_as_empty_cell() -> None:
    captured = []
    captured.append(captured)

    def kernel():
        return captured

    with pytest.raises(ValueError, match="cyclic semantic identity"):
        _freeze_value(kernel)


def test_h02_referenced_global_helper_implementation_is_identity() -> None:
    def helper_a(x):
        return x + 1

    def helper_b(x):
        return x + 2

    def kernel(x):
        return _h_identity_helper(x)

    namespace = kernel.__globals__
    previous = namespace.get("_h_identity_helper")
    try:
        namespace["_h_identity_helper"] = helper_a
        first = _freeze_value(kernel)
        namespace["_h_identity_helper"] = helper_b
        second = _freeze_value(kernel)
    finally:
        if previous is None:
            namespace.pop("_h_identity_helper", None)
        else:
            namespace["_h_identity_helper"] = previous
    assert first != second


def test_h03_frame_identity_is_lossless_for_integer_boundaries_and_labels() -> None:
    # Both values collapse to the same float64; the declared int64 payload must not.
    a = pd.DataFrame([[2**63 - 1]], columns=[("a,b", "")], dtype="int64")
    b = pd.DataFrame([[2**63 - 2]], columns=[("a,b", "")], dtype="int64")
    assert float(a.iloc[0, 0]) == float(b.iloc[0, 0])
    assert _freeze_value(a) != _freeze_value(b)
    c = a.copy()
    c.columns = [("a", "b,")]
    assert _freeze_value(a) != _freeze_value(c)


def test_h03_nullable_mask_and_signed_zero_are_identity() -> None:
    nullable = pd.DataFrame({"x": pd.array([1, None], dtype="Int64")})
    filled = pd.DataFrame({"x": pd.array([1, 0], dtype="Int64")})
    assert _freeze_value(nullable) != _freeze_value(filled)
    assert _freeze_value(pd.DataFrame({"x": [0.0]})) != _freeze_value(
        pd.DataFrame({"x": [-0.0]})
    )


def test_h03_series_axis_and_unused_category_dictionary_are_identity() -> None:
    left = pd.Series(["a"], dtype=pd.CategoricalDtype(["a", "unused"], ordered=False))
    right = pd.Series(["a"], dtype=pd.CategoricalDtype(["a"], ordered=False))
    assert _freeze_value(left) != _freeze_value(right)
    renamed = left.copy()
    renamed.index.name = "different-axis"
    assert _freeze_value(left) != _freeze_value(renamed)
    assert _freeze_value(pd.NA) == "pd.NA"


def test_h04_catalog_is_recursive_immutable_and_defrost_detached() -> None:
    source = {"op": {"items": [{"array": np.array([1, 2])}], "tags": {"a"}}}
    frozen = _deepfreeze_catalog(source)
    source["op"]["items"][0]["array"][0] = 99
    assert frozen["op"]["items"][0]["array"][0] == 1
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["op"]["items"], tuple)
    assert not hasattr(frozen["op"]["items"][0]["array"], "setflags")
    thawed = _defrost_catalog(frozen)
    thawed["op"]["items"][0]["array"][0] = 77
    assert frozen["op"]["items"][0]["array"][0] == 1


def test_h04_nested_dataclass_param_spec_is_detached() -> None:
    from dataclasses import dataclass

    @dataclass
    class ParamSpecProbe:
        choices: list[int]
        defaults: np.ndarray

    original = ParamSpecProbe([1, 2], np.array([3, 4]))
    frozen = _deepfreeze_catalog({"param_specs": {"window": original}})
    original.choices.append(9)
    original.defaults[0] = 99
    assert frozen["param_specs"]["window"].choices == (1, 2)
    assert frozen["param_specs"]["window"].defaults[0] == 3
    thawed = _defrost_catalog(frozen)
    assert isinstance(thawed["param_specs"]["window"], ParamSpecProbe)
    assert isinstance(thawed["param_specs"]["window"].choices, list)
    thawed["param_specs"]["window"].defaults[0] = 77
    assert frozen["param_specs"]["window"].defaults[0] == 3


def test_h05_variant_binds_executed_code_not_only_class_name() -> None:
    def calc_a(self, value):
        return value + 1

    def calc_b(self, value):
        return value + 2

    cls_a = type("SameOperator", (), {"__module__": "identity_probe", "calculate": calc_a})
    cls_b = type("SameOperator", (), {"__module__": "identity_probe", "calculate": calc_b})
    a = ExecutionVariantIdentity.of(cls_a(), "pandas_numpy", "probe")
    b = ExecutionVariantIdentity.of(cls_b(), "pandas_numpy", "probe")
    assert a.implementation_id == b.implementation_id
    assert a.code_hash != b.code_hash
    assert a.kernel_variant == "pandas_reference"


def test_h05_polars_delegate_is_not_called_native() -> None:
    class Delegate:
        __module__ = "factor_engine.cleaned_operators.rolling_pack"

        def calculate(self, value):
            return value

    variant = ExecutionVariantIdentity.of(Delegate(), "polars", "probe_delegate")
    assert variant.kernel_variant == "polars_udf_pandas_delegate"
    assert variant.kernel_variant != "native_polars"


def test_h05_actual_overhaul_winners_bind_distinct_wrapped_kernels() -> None:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all(include_research=True)
    winners = [OperatorRegistry.get(name, "pandas_numpy") for name in ("ADX", "ATR_WILDER", "MACD")]
    assert all(winner is not None for winner in winners)
    hashes = [_impl_source_hash(winner) for winner in winners]
    assert len(set(hashes)) == len(hashes), hashes
