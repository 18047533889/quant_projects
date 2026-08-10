# -*- coding: utf-8 -*-
"""R19-001..008: parameter-system SINGLE authority + R19-139/140 targeted tests.

* R19-001 / R19-140 — ``ParameterCanonicalizer.hash_key`` is a stable tuple
  schema (was an illegal ``str + tuple``): same params -> same key, different
  semantic params -> different key, order-invariant and cross-process
  deterministic.
* R19-002 — ``normalize_and_validate_scalar_param`` is the ONE entry shared by
  planning / runtime / hash; np scalars, Decimal, enum and explicit ``None`` are
  validated on the same declared ParamSpec domain (no more planning gap that
  skipped anything not ``isinstance(value, (int, float, str, bool))``).
* R19-003 — an explicit ``None`` positional literal is a real bound value that
  enters ParamSpec validation / active_when / relational / hash identity.
* R19-004 — ``cleaned_operators.parameter_validation`` is a PURE re-export of
  ``common.strict_params`` (no independent logic).
* R19-005 / R19-139 — numeric-string conversion happens ONLY at the declaration
  layer (``bind_numeric_string_if_declared``); kernel gates
  (``strict_int`` / ``strict_int_param`` / ``strict_int_runtime``) REJECT
  ``"20"``.
* R19-006 — parameter alias SINGLE authority = ``metadata.param_aliases``;
  ``derive_alias_map_from_metadata`` + ``GLOBAL_ALIAS_MAP ==
  DERIVED_METADATA_ALIAS_MAP`` (compat-only fallback explicitly marked).
* R19-007/008 — ``bind_operator_call`` builds ONE normalized bound that
  active_when / relational specs / history / hash / kernel all read.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.base import (
    MISSING,
    OperatorMetadata,
    ParamSpec,
    _normalise_call,
    bind_operator_call,
    validate_operator_call,
)
from cleaned_operators.common.strict_params import (
    normalize_and_validate_scalar_param,
    strict_int,
)
from backend.operator_errors import OperatorParameterError


def _panel():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3))


# ---------------------------------------------------------------------------
# R19-001 / R19-140: ParameterCanonicalizer.hash_key golden tests
# ---------------------------------------------------------------------------

def test_r19_140_hash_key_callable_and_deterministic():
    from planner.canonicalize_params import ParameterCanonicalizer

    pc = ParameterCanonicalizer("custom_op")
    key = pc.hash_key({"window": 20})
    assert isinstance(key, tuple)
    assert key[0] == "custom_op"
    assert isinstance(key[1], tuple)
    # same params -> same key
    assert pc.hash_key({"window": 20}) == key
    # different semantic params -> different key
    assert pc.hash_key({"window": 21}) != key
    assert pc.hash_key({"window": 20, "lag": 1}) != key
    # order-invariant -> cross-process deterministic (sorted element tuples)
    assert pc.hash_key({"b": 2, "a": 1}) == pc.hash_key({"a": 1, "b": 2})
    # float-noise canonicalization still applies (12 significant digits)
    assert pc.hash_key({"window": 0.123456789012345}) == pc.hash_key({"window": 0.123456789012})
    # explicit None survives the hash identity
    assert pc.hash_key({"center": None}) == pc.hash_key({"center": None})
    assert pc.hash_key({"center": None}) != pc.hash_key({})


def test_r19_001_hash_key_no_str_plus_tuple_type_error():
    from planner.canonicalize_params import ParameterCanonicalizer

    # The old ``(canonical or "") + tuple(...)`` raised TypeError at call time.
    pc = ParameterCanonicalizer("ts_mean")
    key = pc.hash_key({"window": 20})
    assert isinstance(key[0], str) and isinstance(key[1], tuple)
    assert pc.canonicalize({"window": 20})  # canonicalize still works


# ---------------------------------------------------------------------------
# R19-002: unified scalar validation — planning/runtime/hash same domain
# ---------------------------------------------------------------------------

def test_r19_002_np_scalar_validated_at_planning():
    spec = ParamSpec(dtype=int, min=2)
    # np.int64 is a real scalar the old planning filter (int/float/str/bool)
    # silently skipped; the unified entry validates it.
    assert normalize_and_validate_scalar_param(
        "ts_mean", "window", np.int64(20), phase="planning", spec=spec
    ) == 20
    # np.float64 fractional rejected identically at every phase.
    for phase in ("planning", "runtime", "hash"):
        with pytest.raises(OperatorParameterError):
            normalize_and_validate_scalar_param(
                "ts_mean", "window", np.float64(20.5), phase=phase, spec=spec
            )


def test_r19_002_decimal_and_enum_same_domain_as_runtime():
    spec = ParamSpec(dtype=int, min=2)
    # Decimal is not in the accepted numeric family -> rejected by the SAME rule
    # runtime applies (a value runtime would reject is rejected at planning).
    with pytest.raises(OperatorParameterError):
        normalize_and_validate_scalar_param(
            "op", "window", Decimal("20"), phase="planning", spec=spec
        )
    # Enum scalar accepted when choices declare it (same as runtime choices gate).
    class Mode:
        A = "A"
        B = "B"

    spec = ParamSpec(dtype=str, choices=(Mode.A, Mode.B))
    assert normalize_and_validate_scalar_param(
        "op", "mode", Mode.B, phase="planning", spec=spec
    ) == Mode.B


def test_r19_002_structured_values_pass_through():
    # Vector/structured params are NOT scalar params — the vector path owns them.
    assert normalize_and_validate_scalar_param(
        "op", "weights", [1, 2, 3], phase="planning", spec=None
    ) == [1, 2, 3]
    assert normalize_and_validate_scalar_param(
        "op", "weights", (0.5, 0.5), phase="runtime", spec=None
    ) == (0.5, 0.5)
    # unknown phase rejected
    with pytest.raises(ValueError):
        normalize_and_validate_scalar_param("op", "x", 1, phase="bogus")


def test_r19_002_explicit_none_with_declared_none_default_accepted():
    assert (
        normalize_and_validate_scalar_param(
            "op", "center", None, phase="planning",
            spec=ParamSpec(dtype=float, default=None),
        )
        is None
    )


# ---------------------------------------------------------------------------
# R19-003: explicit None positional literal is NOT conflated with a missing one
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _registry():
    from cleaned_operators import load_all

    load_all()
    return None


def _literal_plan(op: str, panel_count: int, *values):
    from planner.logical_plan import PlanNode

    def _col(name: str) -> PlanNode:
        return PlanNode(op="column", inputs=[], attrs={"name": name})

    def _lit(value) -> PlanNode:
        return PlanNode(op="literal", inputs=[], attrs={"value": value})

    inputs = [_col(f"p{i}") for i in range(panel_count)]
    inputs.extend(_lit(v) for v in values)
    return PlanNode(op=op, inputs=inputs, attrs={})


def test_r19_003_explicit_none_positional_literal_enters_validation(_registry):
    from planner.canonicalize_params import validate_plan_params

    # ts_transfer_entropy declares window as ParamSpec(dtype=int, min=2) with no
    # None default -> an explicit positional None for window is a contract
    # violation and must be REJECTED at planning (previously the None literal was
    # skipped by ``attrs.get("value") is not None``, so this invalid call
    # silently passed).
    with pytest.raises(OperatorParameterError, match="planning-time"):
        validate_plan_params(_literal_plan("ts_transfer_entropy", 2, None))
    # A valid integer window passes.
    validate_plan_params(_literal_plan("ts_transfer_entropy", 2, 20))
    # ts_argmax ALSO spec's an int window -> an explicit None window is rejected,
    # proving the None literal enters validation for a second operator.
    with pytest.raises(OperatorParameterError, match="planning-time"):
        validate_plan_params(_literal_plan("ts_argmax", 1, None))


def test_r19_003_explicit_none_tolerated_when_declared_none_default(_registry):
    from planner.canonicalize_params import validate_plan_params

    # ts_topk_sum declares k: ParamSpec(dtype=int, min=1, default=None) -> an
    # explicit positional None for k is a LEGAL bound value (the declared None
    # default), which proves the None literal is BOUND and judged — not dropped.
    validate_plan_params(_literal_plan("ts_topk_sum", 1, 20, None))
    validate_plan_params(_literal_plan("ts_topk_sum", 1, 20, 5))


# ---------------------------------------------------------------------------
# R19-004: parameter_validation.py is a PURE re-export
# ---------------------------------------------------------------------------

def test_r19_004_parameter_validation_is_pure_reexport():
    import cleaned_operators.parameter_validation as pv
    from cleaned_operators.common.strict_params import strict_float, strict_int

    assert pv.strict_integer is strict_int
    assert pv.strict_finite_scalar is strict_float
    # authoritative gate accepts np scalars (the old independent impl rejected
    # np.integer / np.floating) -> no more domain drift.
    assert pv.strict_integer(np.int64(5), "x", minimum=1) == 5
    assert pv.strict_finite_scalar(np.float64(0.5), "x", minimum=0.0) == 0.5
    # strings rejected by the kernel gate (R19-005).
    with pytest.raises(OperatorParameterError):
        pv.strict_integer("20", "x")
    with pytest.raises(OperatorParameterError):
        pv.strict_finite_scalar("0.5", "x")
    # the module source must contain no independent validation logic.
    import inspect

    src = inspect.getsource(pv)
    assert "def strict_integer(" not in src
    assert "def strict_finite_scalar(" not in src


# ---------------------------------------------------------------------------
# R19-005 / R19-139: numeric-string conversion only at the declaration layer
# ---------------------------------------------------------------------------

def test_r19_139_numeric_string_binder_kernel_separation():
    from cleaned_operators.base import (
        bind_numeric_string_if_declared,
        strict_int_param,
        strict_int_runtime,
    )

    # declaration layer: numeric string bound ONLY when the contract is numeric.
    assert bind_numeric_string_if_declared("20", "window", spec=ParamSpec(dtype=int)) == 20
    assert bind_numeric_string_if_declared("20", "window", declared_type=int) == 20
    # string param / undeclared keeps the exact string.
    assert bind_numeric_string_if_declared("000001", "code", None, ParamSpec(dtype=str)) == "000001"
    assert bind_numeric_string_if_declared("20", "x", None, None) == "20"
    # kernel gates REJECT numeric strings — strict_int("20") must fail.
    for gate in (strict_int, strict_int_param, strict_int_runtime):
        with pytest.raises(OperatorParameterError):
            gate("20", "window")
    # numeric values still accepted (incl. np scalars).
    assert strict_int(20.0, "window") == 20
    assert strict_int(np.int64(20), "window") == 20


def test_r19_005_runtime_call_gate_converts_then_kernel_rejects_strings():
    # _normalise_call applies the declaration-layer binder ONCE for a declared
    # numeric param, so the normalized value reaching the kernel is numeric.
    meta = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["x", "window"],
        param_specs={"window": ParamSpec(dtype=int, min=2)},
    )
    args, kwargs = _normalise_call(meta, (_panel(),), {"window": "20"})
    assert kwargs["window"] == 20
    # A string on an UNDECLARED numeric param is NOT converted by the binder; a
    # kernel that then calls strict_int rejects it (no bypass).
    with pytest.raises(OperatorParameterError):
        strict_int("20", "window")


# ---------------------------------------------------------------------------
# R19-006: parameter alias SINGLE authority = metadata.param_aliases
# ---------------------------------------------------------------------------

def test_r19_006_alias_single_authority_derived_from_metadata(_registry):
    from backend.parameter_aliases import (
        DERIVED_METADATA_ALIAS_MAP,
        GLOBAL_ALIAS_MAP,
        PARAMETER_ALIASES,
        _COMPAT_ONLY_ALIAS_CANONICALS,
        assert_global_alias_map_consistent,
        derive_alias_map_from_metadata,
        global_alias_map,
        normalize_parameter_aliases,
    )

    derived = DERIVED_METADATA_ALIAS_MAP()
    # metadata is the authority: operators that DECLARE param_aliases appear.
    assert "ts_argmax" in derived
    assert derived["ts_argmax"] == {"d": "window"}
    # compat fallback still applies for operators whose metadata declares none.
    assert normalize_parameter_aliases("ts_mean", {"d": 5}) == {"window": 5}
    # every hand-written compat entry is explicitly marked compat-only when
    # metadata does not back it.
    for canon, aliases in PARAMETER_ALIASES.items():
        if not derived.get(canon):
            assert canon in _COMPAT_ONLY_ALIAS_CANONICALS, (
                f"R19-006: {canon}: compat alias {aliases} not backed by metadata "
                "and not marked compat-only"
            )
    # the CI gate passes after a full load_all.
    assert_global_alias_map_consistent()
    # GLOBAL_ALIAS_MAP exposes every metadata-derived alias unchanged.
    global_map = GLOBAL_ALIAS_MAP()
    for canon, aliases in derived.items():
        assert global_map.get(canon) == aliases, (
            f"R19-006: {canon}: global alias map {global_map.get(canon)} "
            f"!= metadata-derived {aliases}"
        )
    # metadata wins over the compat map when BOTH declare an alias for the same
    # canonical (no conflict tolerated).
    assert global_alias_map()["ts_argmax"] == derived["ts_argmax"]


# ---------------------------------------------------------------------------
# R19-007/008: active_when / relations / hash / kernel read ONE normalized bound
# ---------------------------------------------------------------------------

class _TestOp:
    def __init__(self, metadata):
        self.metadata = metadata

    def _calculate_series(self, x=None, mode="A", center=None, window=20, **kwargs):
        # Kernel-signature defaults mirror what a real operator declares, so
        # ``_kernel_param_defaults`` resolves them into the normalized bound.
        return None

    def validate_params(self, *args, **kwargs):
        return True


def _active_when_meta():
    return OperatorMetadata(
        name="op",
        category="ts",
        param_names=["x", "mode", "center"],
        panel_params=("x",),
        param_aliases={"m": "mode"},
        param_specs={
            "mode": ParamSpec(dtype=str, choices=("A", "B"), default="A"),
            "center": ParamSpec(dtype=float, active_when=("mode", ("B",)), default=None),
        },
    )


def test_r19_007_active_when_reads_alias_resolved_normalized_bound():
    # m="B" is an ALIAS for the active_when controller "mode".  The OLD code fed
    # raw kwargs to _enforce_active_when (bound had 'm' but no 'mode' -> default
    # 'A' -> center judged INACTIVE -> rejected).  R19-007/008 resolves the alias
    # onto the canonical bound first, so center is ACTIVE and accepted.
    op = _TestOp(_active_when_meta())
    args, kwargs = validate_operator_call(op, (_panel(),), {"m": "B", "center": 1.0})
    assert kwargs["center"] == 1.0
    # center under an inactive controller is still rejected (same dead-knob rule).
    with pytest.raises(OperatorParameterError):
        validate_operator_call(op, (_panel(),), {"m": "A", "center": 1.0})


def test_r19_008_bind_operator_call_single_bound_shared():
    meta = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["x", "mode", "center", "window"],
        panel_params=("x",),
        param_aliases={"m": "mode"},
        param_specs={
            "mode": ParamSpec(dtype=str, choices=("A", "B"), default="A"),
            "center": ParamSpec(dtype=float, active_when=("mode", ("B",)), default=None),
            "window": ParamSpec(dtype=int, min=2, default=20),
        },
    )
    op = _TestOp(meta)
    call = bind_operator_call(op, (_panel(),), {"m": "B", "center": 1.0})
    # the SAME bound serves active_when / relations / history / hash / kernel:
    bound = call.bound
    assert bound.normalized_values["mode"] == "B"  # alias resolved
    assert bound.normalized_values["center"] == 1.0
    assert bound.normalized_values["window"] == 20  # default applied
    assert bound.canonical_aliases_resolved == {"m": "mode"}
    assert bound.scalar_params["mode"] == "B"  # alias resolved onto canonical
    assert bound.scalar_params["center"] == 1.0
    assert "center" not in bound.inactive_params  # active
    assert call.kwargs["m"] == "B"  # kernel receives the processed kwarg
    # inactive partition is exposed for the dead-knob case.
    call2 = bind_operator_call(op, (_panel(),), {"m": "A"})
    assert "center" in call2.bound.inactive_params
    # relational specs / common relations are evaluated on the same bound.
    from cleaned_operators.base import RelationalParamSpec

    meta_rel = OperatorMetadata(
        name="op",
        category="ts",
        param_names=["window"],
        param_specs={"window": ParamSpec(dtype=int, min=2, default=20)},
        relational_specs=[RelationalParamSpec("window >= 10")],
    )
    validate_operator_call(_TestOp(meta_rel), (), {"window": 20})
    with pytest.raises(ValueError, match="parameter relation violated"):
        validate_operator_call(_TestOp(meta_rel), (), {"window": 5})
