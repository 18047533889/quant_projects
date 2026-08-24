# -*- coding: utf-8 -*-
"""R10 registry-identity regression tests.

Covers three review findings on ``cleaned_operators/registry.py``:

* R10 #13 (P0) — implementation-hash collision in ``_code_payload``: bytecode
  operands reference ORIGINAL ``co_consts``/``co_names`` tuple indices, so the
  old ``sorted(consts)`` digest let ``2*x+3`` collide with ``3*x+2``.  The
  identity is now a normalized disassembly ``[(opcode, resolved_operand), ...]``
  in bytecode order (never sorted).
* R10 #14 (P1) — ``_freeze_value`` no longer produces repr/pointer-dependent
  digests for object ndarrays / MultiIndex / custom elements; unfreezeable
  objects raise a clear ``TypeError``.
* R10 #15 (P0) — ``register_catalog_only(merge_existing=True)`` and
  ``rename_canonical`` preserve the full canonical contract and FAIL when two
  canonicals merging under the same name declare different rich contracts.

These tests exercise the registry's pure machinery on DIRECTLY-CONSTRUCTED
operator instances.  Registry mutation uses the bootstrap token to thaw and
always restores the FROZEN lifecycle (the autouse conftest fixture runs
``load_all()`` once per session).
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from factor_engine.cleaned_operators.base import (
    MISSING,
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
)
from factor_engine.cleaned_operators.registry import (
    OperatorRegistry,
    _BOOTSTRAP_TOKEN,
    _code_payload,
    _freeze_value,
    _impl_source_hash,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_kernel(
    name: str,
    *,
    param_specs=None,
    input_units=None,
    output_unit=None,
    compatible_units=None,
    panel_params=(),
    scalar_params=(),
    input_grain=None,
    output_grain=None,
    available_at=None,
    same_session_usable=None,
    input_fields=(),
    window_semantics=None,
    role=None,
    param_aliases=None,
    tags=(),
):
    meta = OperatorMetadata(
        name=name,
        category="test",
        description="r10 test kernel",
        param_names=["x"],
        param_specs=param_specs or {},
        input_units=input_units or {},
        output_unit=output_unit,
        compatible_units=compatible_units or {},
        panel_params=tuple(panel_params or ()),
        scalar_params=tuple(scalar_params or ()),
        input_grain=input_grain,
        output_grain=output_grain,
        available_at=available_at,
        same_session_usable=same_session_usable,
        input_fields=list(input_fields or ()),
        window_semantics=window_semantics,
        role=role,
        param_aliases=param_aliases or {},
        tags=list(tags),
    )

    class _Kernel(SeriesOperator):
        metadata = meta

        def _calculate_series(self, x, **kwargs):
            return x

    _Kernel.__name__ = "R10Kernel_" + name.replace("-", "_")
    return _Kernel()


@pytest.fixture
def mutable_registry():
    """Thaw the (frozen) registry, yield it, then remove every test canonical
    and restore the FROZEN lifecycle even if the test body fails."""
    OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
    first_registered_before = copy.deepcopy(OperatorRegistry._first_registered)
    override_chain_before = copy.deepcopy(OperatorRegistry._override_chain)
    declared_manifest_before = copy.deepcopy(
        OperatorRegistry._DECLARED_OVERRIDE_MANIFEST
    )
    overwrite_log_before = copy.deepcopy(OperatorRegistry._overwrite_log)
    try:
        yield OperatorRegistry
    finally:
        for name in list(OperatorRegistry._operators):
            if name.startswith("__r10_"):
                OperatorRegistry.unregister(name)
        for name in list(OperatorRegistry._catalog):
            if name.startswith("__r10_"):
                OperatorRegistry._catalog.pop(name, None)
        for alias in list(OperatorRegistry._aliases):
            if alias.startswith("__r10_") or alias.startswith("r10_"):
                OperatorRegistry._aliases.pop(alias, None)
        OperatorRegistry._first_registered.clear()
        OperatorRegistry._first_registered.update(
            copy.deepcopy(first_registered_before)
        )
        OperatorRegistry._override_chain.clear()
        OperatorRegistry._override_chain.update(copy.deepcopy(override_chain_before))
        OperatorRegistry._DECLARED_OVERRIDE_MANIFEST.clear()
        OperatorRegistry._DECLARED_OVERRIDE_MANIFEST.update(
            copy.deepcopy(declared_manifest_before)
        )
        OperatorRegistry._overwrite_log[:] = copy.deepcopy(overwrite_log_before)
        if OperatorRegistry.lifecycle() == "building":
            OperatorRegistry.finalize()
        OperatorRegistry.freeze()


# ---------------------------------------------------------------------------
# R10 #13 — implementation identity from disassembled bytecode
# ---------------------------------------------------------------------------

def test_code_payload_distinguishes_constant_swap():
    """``2*x + 3`` and ``3*x + 2`` share the constant set {2, 3} and structurally
    identical bytecode with the same resolved indices — the old sorted-consts
    digest collided.  The disassembly identity must NOT collide."""
    def f1(x):
        return 2 * x + 3

    def f2(x):
        return 3 * x + 2

    h1 = _code_payload(f1.__code__, include_names=True)
    h2 = _code_payload(f2.__code__, include_names=True)
    assert h1 != h2


def test_code_payload_distinguishes_global_call_order():
    """Two functions that only swap the order of two distinct global-function
    calls must hash differently (the resolved name operands are part of the
    identity, and bytecode order is preserved)."""
    def g1(x):
        return abs(x) + round(x)

    def g2(x):
        return round(x) + abs(x)

    h1 = _code_payload(g1.__code__, include_names=True)
    h2 = _code_payload(g2.__code__, include_names=True)
    assert h1 != h2


def test_code_payload_is_deterministic_and_reproducible():
    """Same function -> identical hash across separate calls; the digest is a
    stable, reproducible hex string."""
    def f(x):
        return 2 * x + 3

    h1 = _code_payload(f.__code__, include_names=True)
    h2 = _code_payload(f.__code__, include_names=True)
    assert h1 == h2
    assert len(h1) == 16
    # Reproducible: the digest is a stable hex string of the normalized payload.
    assert all(c in "0123456789abcdef" for c in h1)
    # And through the public implementation-hash path on real operator instances.
    op1 = _make_kernel("__r10_det")
    op2 = _make_kernel("__r10_det")
    assert _impl_source_hash(op1) == _impl_source_hash(op2)


def test_code_payload_recurses_nested_code_objects_in_order():
    """Nested code objects recurse with their ORDER preserved."""
    def outer_a():
        def inner_1():
            return 1
        def inner_2():
            return 2
        return inner_1(), inner_2()

    def outer_b():
        def inner_1():
            return 2
        def inner_2():
            return 1
        return inner_1(), inner_2()

    assert _code_payload(outer_a.__code__, include_names=True) != _code_payload(
        outer_b.__code__, include_names=True
    )


def test_impl_hash_distinguishes_kernels_after_identity_change():
    """Two class-defined kernels whose bodies differ ONLY by a swapped constant
    (``2*x+3`` vs ``3*x+2``) must now hash differently through the operator-level
    ``_impl_source_hash`` — the R10 #13 collision reproduced at the public path."""

    class _SwapA(SeriesOperator):
        metadata = OperatorMetadata(name="__r10_swap_a", category="test", param_names=["x"])

        def _calculate_series(self, x, **kwargs):
            return 2 * x + 3

    class _SwapB(SeriesOperator):
        metadata = OperatorMetadata(name="__r10_swap_b", category="test", param_names=["x"])

        def _calculate_series(self, x, **kwargs):
            return 3 * x + 2

    assert _impl_source_hash(_SwapA()) != _impl_source_hash(_SwapB())


# ---------------------------------------------------------------------------
# R10 #14 — canonical freezing (no repr / pointer-dependent digests)
# ---------------------------------------------------------------------------

def test_freeze_value_missing_sentinel_is_canonical():
    assert _freeze_value(MISSING) == "MISSING"


def test_freeze_value_param_spec_with_missing_default():
    spec = ParamSpec(
        dtype=int, min=1, max=10, choices=("a", "b"),
        history_semantics="exact_rows", default=MISSING,
    )
    frozen = _freeze_value(spec)
    assert isinstance(frozen, str)
    assert "MISSING" in frozen
    assert "exact_rows" in frozen


def test_freeze_value_object_ndarray_is_canonical_not_pointer():
    arr = np.array(["x", "y", "z"], dtype=object)
    h1 = _freeze_value(arr)
    h2 = _freeze_value(np.array(["x", "y", "z"], dtype=object))
    assert h1 == h2  # deterministic, not address-derived
    assert _freeze_value(np.array(["x", "z", "y"], dtype=object)) != h1


def test_freeze_value_object_ndarray_unfreezeable_raises_typeerror():
    class _Opaque:
        pass

    arr = np.array([_Opaque(), _Opaque()], dtype=object)
    with pytest.raises(TypeError):
        _freeze_value(arr)


def test_freeze_value_multiindex_roundtrips_canonically():
    import pandas as pd

    mi1 = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    mi2 = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    df1 = pd.DataFrame({"x": [1.0, 2.0]}, index=mi1)
    df2 = pd.DataFrame({"x": [1.0, 2.0]}, index=mi2)
    assert _freeze_value(df1) == _freeze_value(df2)
    df3 = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 3)]))
    assert _freeze_value(df1) != _freeze_value(df3)


# ---------------------------------------------------------------------------
# R10 #15 — full canonical-contract preservation on merge / rename
# ---------------------------------------------------------------------------

_RICH_SPEC = ParamSpec(
    dtype=int,
    min=1,
    max=10,
    default=3,
    history_semantics="exact_rows",
)


def _assert_full_contract(cat: dict) -> None:
    """Assert every rich contract field the helpers below declare survives."""
    assert cat["param_names"] == ["x"]
    assert cat["param_specs"] == {"x": _RICH_SPEC}
    assert cat["param_aliases"] == {"d": "x"}
    assert cat["panel_params"] == ("x",)
    assert cat["scalar_params"] == ("window",)
    assert cat["input_units"] == {"x": "price"}
    assert cat["output_unit"] == "ret"
    assert cat["compatible_units"] == {"x": ("price", "log_price")}
    assert cat["input_grain"] == "intraday"
    assert cat["output_grain"] == "intraday"
    assert cat["available_at"] == "session_close"
    assert cat["same_session_usable"] is False
    assert cat["input_fields"] == ["x"]
    assert cat["window_semantics"] == "rolling"
    assert cat["semantic_version"] == "3.0"


def _register_rich(registry, name: str, aliases=()):
    op = _make_kernel(
        name,
        param_specs={"x": _RICH_SPEC},
        param_aliases={"d": "x"},
        panel_params=("x",),
        scalar_params=("window",),
        input_units={"x": "price"},
        output_unit="ret",
        compatible_units={"x": ("price", "log_price")},
        input_grain="intraday",
        output_grain="intraday",
        available_at="session_close",
        same_session_usable=False,
        input_fields=["x"],
        window_semantics="rolling",
    )
    registry.register(op, canonical=name, aliases=list(aliases), source=f"src_{name}", semantic_version="3.0")


def test_register_catalog_only_merge_preserves_full_contract(mutable_registry):
    """A catalog-only overlay (merge_existing=True) must NOT rebuild a minimal
    dict that drops the runtime-backed rich contract."""
    _register_rich(mutable_registry, "__r10_overlay", aliases=["r10_ov_alias"])
    mutable_registry.register_catalog_only(
        "__r10_overlay",
        merge_existing=True,
        status="implemented",
        business_category="demo",
    )
    cat = mutable_registry._catalog["__r10_overlay"]
    _assert_full_contract(cat)
    # aliases survive AND the overlay can add its own.
    assert "r10_ov_alias" in cat["aliases"]
    assert cat["status"] == "implemented"
    assert cat["business_category"] == "demo"


def test_register_catalog_only_merge_rejects_different_param_specs(mutable_registry):
    """Merging two canonicals under the same name with DIFFERENT param_specs
    must fail loudly (R10 #15) instead of silently overwriting one contract."""
    _register_rich(mutable_registry, "__r10_conflict")
    with pytest.raises(ValueError, match="rich canonical contracts differ"):
        mutable_registry.register_catalog_only(
            "__r10_conflict",
            merge_existing=True,
            param_specs={"x": ParamSpec(dtype=int, min=100, max=200)},
        )
    # Original contract untouched.
    assert mutable_registry._catalog["__r10_conflict"]["param_specs"] == {"x": _RICH_SPEC}


def test_rename_canonical_merge_preserves_full_contract(mutable_registry):
    """rename_canonical(old, new) when ``new`` already exists must carry every
    contract field over — including the renamed-away canonical's aliases and any
    backend provenance it uniquely contributes."""
    _register_rich(mutable_registry, "__r10_target", aliases=["r10_t_alias"])
    _register_rich(mutable_registry, "__r10_source", aliases=["r10_s_alias"])
    mutable_registry.rename_canonical("__r10_source", "__r10_target")
    cat = mutable_registry._catalog["__r10_target"]
    _assert_full_contract(cat)
    assert "r10_t_alias" in cat["aliases"]
    assert "r10_s_alias" in cat["aliases"]
    # The renamed-away canonical becomes an alias of the target.
    assert "__r10_source" in cat["aliases"]
    assert mutable_registry.resolve_canonical("__r10_source") == "__r10_target"
    assert "__r10_source" not in mutable_registry._operators
    # backend provenance survived (the target's own backend entry).
    assert "pandas_numpy" in (cat.get("backend_meta") or {})


def test_rename_canonical_pure_rename_carries_full_contract(mutable_registry):
    """A pure rename (new canonical did not exist) must carry the ENTIRE catalog
    dict — nothing dropped."""
    _register_rich(mutable_registry, "__r10_pure_old", aliases=["r10_p_alias"])
    mutable_registry.rename_canonical("__r10_pure_old", "__r10_pure_new")
    cat = mutable_registry._catalog["__r10_pure_new"]
    _assert_full_contract(cat)
    assert "r10_p_alias" in cat["aliases"]
    assert "__r10_pure_old" in cat["aliases"]
    # operator metadata renamed to the new canonical.
    op = mutable_registry.get("__r10_pure_new", "pandas_numpy", mode="any")
    assert op is not None and op.metadata.name == "__r10_pure_new"


def test_rename_canonical_merge_fills_none_backend_placeholder(mutable_registry):
    """A concrete source implementation must replace a None target slot."""
    source = _make_kernel("__r10_collision_source")
    mutable_registry.register(
        source,
        canonical="__r10_collision_source",
        source="src_concrete",
    )
    mutable_registry._operators["__r10_collision_target"] = {"pandas_numpy": None}
    mutable_registry._catalog["__r10_collision_target"] = {
        "canonical": "__r10_collision_target",
        "aliases": [],
        "backends": ["pandas_numpy"],
        "backend_meta": {
            "pandas_numpy": {"explicit": True, "source": "target_placeholder"}
        },
    }

    mutable_registry.rename_canonical(
        "__r10_collision_source", "__r10_collision_target"
    )

    assert (
        mutable_registry.get(
            "__r10_collision_target", "pandas_numpy", mode="any"
        )
        is source
    )
    assert (
        mutable_registry._catalog["__r10_collision_target"]["backend_meta"]
        ["pandas_numpy"]["source"]
        == "src_concrete"
    )
    assert source.metadata.name == "__r10_collision_target"


def test_rename_canonical_merge_preserves_concrete_target_backend(mutable_registry):
    """A concrete target backend remains authoritative during a merge."""
    target = _make_kernel("__r10_concrete_target")
    source = _make_kernel("__r10_concrete_source")
    mutable_registry.register(
        target,
        canonical="__r10_concrete_target",
        source="target_concrete",
    )
    mutable_registry.register(
        source,
        canonical="__r10_concrete_source",
        source="source_concrete",
    )

    mutable_registry.rename_canonical(
        "__r10_concrete_source", "__r10_concrete_target"
    )

    assert (
        mutable_registry.get(
            "__r10_concrete_target", "pandas_numpy", mode="any"
        )
        is target
    )
    assert (
        mutable_registry._catalog["__r10_concrete_target"]["backend_meta"]
        ["pandas_numpy"]["source"]
        == "target_concrete"
    )


def test_rename_canonical_merge_rejects_different_param_specs(mutable_registry):
    """Merging two canonicals under the same name with DIFFERENT rich contracts
    must fail (R10 #15) — the merge must not silently pick one contract."""
    _register_rich(mutable_registry, "__r10_m_target")
    other = _make_kernel(
        "__r10_m_source",
        param_specs={"x": ParamSpec(dtype=float, min=0.0, max=1.0)},
        output_unit="pct",
    )
    mutable_registry.register(other, canonical="__r10_m_source", source="src_other", semantic_version="3.0")
    with pytest.raises(ValueError, match="rich canonical contracts differ"):
        mutable_registry.rename_canonical("__r10_m_source", "__r10_m_target")
    # Neither side was mutated by the failed merge.
    assert "__r10_m_source" in mutable_registry._operators
    assert "__r10_m_target" in mutable_registry._operators
    assert mutable_registry._catalog["__r10_m_target"]["param_specs"] == {"x": _RICH_SPEC}
