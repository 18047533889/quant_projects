# -*- coding: utf-8 -*-
"""Focused hard gates for R42-001..006 and R42-009..011."""
from __future__ import annotations

import pytest

from cleaned_operators.base import OperatorMetadata, SeriesOperator
from cleaned_operators.operator_policy import OperatorPolicy
from cleaned_operators.operator_spec import (
    DeterminismContract,
    NumericStabilityContract,
    OutputShapeContract,
    OperatorSpec,
    _compute_allow_in_production,
    _infer_panel_params,
    resolve_output_shape_contract,
    spec_to_manifest_entry,
    validate_output_shape_contract,
)
from ir.types import SemanticIdentityDigest, SemanticLattice, SemanticTypeBundle
from runtime.factor_identity import IdentityHashTypeError, _stable_hash


class _Op(SeriesOperator):
    metadata = OperatorMetadata(name="r42_op", category="test")

    def _calculate_series(self, x):
        return x


def _spec(**overrides):
    base = dict(
        canonical="r42_op", status="research", allow_in_production=False,
        deterministic=True, pit_safe=True, backends=("pandas_numpy",),
        policy=OperatorPolicy(pit_safe=True),
    )
    base.update(overrides)
    return OperatorSpec(**base)


def test_output_shape_resolves_only_explicit_typed_contract():
    contract = OutputShapeContract(
        "minute", "daily", aggregation_keys=("trade_date", "instrument"),
    )
    op = _Op()
    op.metadata.output_shape_contract = contract
    assert resolve_output_shape_contract("r42_op", op, {}, OperatorPolicy()) == contract

    op.metadata.output_shape_contract = None
    assert resolve_output_shape_contract(
        "r42_op", op, {"input_grain": "minute", "output_grain": "daily"}, OperatorPolicy()
    ) is None


def test_output_shape_serializes_to_spec_and_manifest():
    contract = OutputShapeContract(
        "minute", "daily", aggregation_keys=("trade_date", "instrument"),
    )
    spec = _spec(output_shape=contract)
    expected = contract.to_dict()
    assert spec.to_dict()["output_shape"] == expected
    assert spec_to_manifest_entry(spec)["output_shape"] == expected


def test_shape_changing_contract_requires_typed_keys():
    assert not validate_output_shape_contract(None, shape_preserving=False)
    assert not validate_output_shape_contract(
        OutputShapeContract("minute", "daily"), shape_preserving=False,
    )
    assert validate_output_shape_contract(
        OutputShapeContract("minute", "daily", aggregation_keys=("trade_date",)),
        shape_preserving=False,
    )


def test_loose_grain_strings_cannot_pass_production_shape_gate(monkeypatch):
    from cleaned_operators.registry import OperatorRegistry
    monkeypatch.setitem(OperatorRegistry._catalog, "r42_shape_gate", {
        "input_grain": "minute", "output_grain": "daily",
    })
    assert _compute_allow_in_production(
        "r42_shape_gate", status="production", pit_safe=True,
        shape_preserving=False, output_shape=None,
    ) is False


def test_determinism_and_numeric_contracts_are_machine_serialized():
    det = DeterminismContract(level="seeded", seed_policy="required")
    num = NumericStabilityContract(
        conditioning_class="ill_conditioned", cancellation_risk="high",
        tolerance_class="strict", preferred_dtype="float64",
    )
    payload = _spec(
        deterministic=True, determinism_contract=det,
        numeric_stability_contract=num,
    ).to_dict()
    assert payload["determinism_contract"]["level"] == "seeded"
    assert payload["numeric_stability_contract"]["cancellation_risk"] == "high"


def test_explicit_parameter_kinds_override_signature_inference():
    op = _Op()
    op.metadata.param_names = ["x", "threshold", "ctx"]
    op.metadata.panel_params = ("x",)
    op.metadata.scalar_params = ("threshold",)
    op.metadata.context_params = ("ctx",)
    assert _infer_panel_params(op, op.metadata, {}) == ("x",)


def test_semantic_type_bundle_joins_all_dimensions():
    left = SemanticTypeBundle(
        market_type="ashare", unit_expr="CNY", price_basis_type="RAW",
        knowledge_type="session_close", universe_type="csi300",
    )
    right = SemanticTypeBundle(
        market_type="ashare", unit_expr="CNY", price_basis_type="CONTINUOUS",
        knowledge_type="session_close", universe_type="csi300",
    )
    joined = SemanticLattice(bundle=left).join(SemanticLattice(bundle=right))
    assert joined.bundle.market_type == "ashare"
    assert joined.bundle.price_basis_type == "MIXED"
    assert joined.bundle.knowledge_type == "session_close"


def test_semantic_serializer_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="string keys"):
        SemanticIdentityDigest.from_attrs({"source_vintage": {1: "a", "1": "b"}})
    with pytest.raises(IdentityHashTypeError, match="string keys"):
        _stable_hash({"payload": {1: "a", "1": "b"}})


def test_semantic_identity_retains_full_sha256_and_short_display():
    digest = SemanticIdentityDigest.from_attrs({"market": "ashare"})
    assert len(digest.value) == 64
    assert digest.display_value == digest.value[:16]
    assert str(digest) == f"SemanticIdentityDigest({digest.value[:16]})"
