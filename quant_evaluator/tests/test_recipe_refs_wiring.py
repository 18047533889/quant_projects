"""Tests locking the FP TreatmentRecipe <-> QE EvaluationRequest ref wiring.

Covers the minimal bidirectional adapter introduced for QRP-P5-INT-1
(``quant_evaluator/adapters/recipe_refs.py``):

- gap a: a recipe's ordered RecipeStep sequence maps to a structured
  :class:`FactorValueRef` (the missing recipe-step -> factor-value-ref path),
  and step ORDER is part of that identity.
- gap b: an ``EvaluationRequest``'s ``factor_value_ref`` /
  ``label_bundle_ref`` CAN be built from a FP recipe + the underlying raw
  factor value (asserted id) + label definition, and round-trip through
  ``to_dict`` / ``from_dict``.  No second value-artifact registry is
  introduced; a recipe on the SAME asserted raw value yields a ref whose
  provenance records the recipe while RAW and TREATED refs remain distinct.
- gap c: ``assert_computed_value`` fails closed on ``None`` / ``valid=False``
  / non-finite payload so a not-computed (e.g. label-not-mature) QE result is
  never treated as success by a downward FP-style consumer.

The rest of this test module deliberately uses only QE contracts — it does
NOT import factor_preprocess (core QE must stay decoupled).  A duck-typed
recipe fixture stands in for the FP authority; the FactorValueRef contract is
the QE authority that both sides serialize.
"""

from __future__ import annotations

import pytest

from quant_evaluator.adapters.recipe_refs import (
    recipe_to_factor_value_ref,
    ref_for_factor_values,
    label_bundle_to_ref,
    request_for_recipe,
    assert_computed_value,
)
from quant_evaluator.api.requests import EvaluationRequest, MetricValue
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef


# ---------------------------------------------------------------------------
# FP-shaped fixture (duck-typed, no factor_preprocess import in QE tests).
# ---------------------------------------------------------------------------

def _recipe(*, source_factor_value_ref="raw_fv_1", steps=None, recipe_id="recipe_1"):
    class _RecipeStep:
        def __init__(self, semantic_transform_id, stage, parameters=None,
                     requires_fit=False, implementation_ref=None):
            self.semantic_transform_id = semantic_transform_id
            self.stage = stage
            self.parameters = dict(parameters or {})
            self.requires_fit = requires_fit
            self.implementation_ref = implementation_ref or semantic_transform_id

    class _Recipe:
        def __init__(self):
            self.recipe_id = recipe_id
            self.source_factor_value_ref = source_factor_value_ref
            self.ordered_steps = tuple(
                steps
                or (
                    _RecipeStep("WINSOR:cs", "outlier"),
                    _RecipeStep("NEUTRAL:ols", "neutralization"),
                    _RecipeStep("CS_RANK:pct", "representation", {"pct": True}),
                )
            )
            self.content_hash = "contenthash-recipe"
            self.treatment_identity = "treatment-identity"

    return _Recipe()


def _label_bundle(target_id="fwd_vwap_10d", horizon=10, source_ref="labels:snap1"):
    class _LB:
        pass

    lb = _LB()
    lb.target_id = target_id
    lb.horizon = horizon
    lb.source_ref = source_ref
    return lb


# ---------------------------------------------------------------------------
# gap a: recipe steps -> FactorValueRef
# ---------------------------------------------------------------------------

def test_recipe_to_factor_value_ref_maps_steps_to_ref():
    recipe = _recipe()
    ref = recipe_to_factor_value_ref(recipe, ref_for="treated")
    assert isinstance(ref, FactorValueRef)
    assert ref.factor_value_id.startswith("factor_value:treated:")
    # Raw source id is bound into the treated ref identity.
    assert "raw_fv_1" in ref.factor_value_id
    assert ref.source_ref == "raw_fv_1"
    # Provenance carries the recipe's treatment identity (not a new catalog).
    meta = dict(ref.metadata)
    assert meta["recipe_id"] == "recipe_1"
    assert meta["treatment_identity"] == "treatment-identity"
    assert "WINSOR:cs" in meta["transform_sequence"]


def test_recipe_to_factor_value_ref_order_is_part_of_identity():
    """gap a: reordering steps changes the treated ref (order is identity)."""
    def _steps(reversed_):
        def _mk():
            return (
                type("S", (), {
                    "semantic_transform_id": "A",
                    "stage": "pre",
                    "parameters": {},
                    "requires_fit": False,
                    "implementation_ref": "a",
                })(),
                type("S", (), {
                    "semantic_transform_id": "B",
                    "stage": "post",
                    "parameters": {},
                    "requires_fit": False,
                    "implementation_ref": "b",
                })(),
            )
        base = list(_mk())
        return tuple(reversed(base)) if reversed_ else tuple(base)

    r1 = _recipe(source_factor_value_ref="fv", steps=_steps(False))
    r2 = _recipe(source_factor_value_ref="fv", steps=_steps(True))
    assert recipe_to_factor_value_ref(r1).factor_value_id != (
        recipe_to_factor_value_ref(r2).factor_value_id
    )
    # RAW ref for same source is identical regardless of steps (raw is raw).
    assert recipe_to_factor_value_ref(r1, ref_for="raw").factor_value_id == (
        recipe_to_factor_value_ref(r2, ref_for="raw").factor_value_id
    )


def test_recipe_to_factor_value_ref_raw_ref_distinct_from_treated():
    recipe = _recipe()
    raw = recipe_to_factor_value_ref(recipe, ref_for="raw")
    treated = recipe_to_factor_value_ref(recipe, ref_for="treated")
    assert raw.factor_value_id != treated.factor_value_id
    assert raw.factor_value_id.startswith("factor_value:raw:")
    assert treated.factor_value_id.startswith("factor_value:treated:")


# ---------------------------------------------------------------------------
# gap b: EvaluationRequest refs constructible from recipe + label, round-trip
# ---------------------------------------------------------------------------

def test_request_for_recipe_builds_refs_and_roundtrips():
    recipe = _recipe()
    lb = _label_bundle()
    req = request_for_recipe(
        recipe,
        label_bundle=lb,
        factor_value_id="fv_asserted_7",
        metric_ids=("rank_ic", "coverage"),
        tier="core",
    )
    assert isinstance(req, EvaluationRequest)
    assert isinstance(req.factor_value_ref, FactorValueRef)
    assert isinstance(req.label_bundle_ref, LabelBundleRef)
    assert req.factor_value_ref.factor_value_id.startswith("factor_value:treated:")
    assert req.label_bundle_ref.label_bundle_id.startswith("label_bundle:")
    assert req.label_bundle_ref.target_id == "fwd_vwap_10d"
    assert req.label_bundle_ref.horizon == 10
    assert req.metric_ids == ("rank_ic", "coverage")

    # Shallow request has the runtime payloads attached but to_dict omits them.
    payload = req.to_dict()
    assert "factor_value_ref" in payload
    assert "label_bundle_ref" in payload
    assert "batch_or_factor_ids" not in payload
    assert "label_bundle" not in payload

    restored = EvaluationRequest.from_dict(req.to_dict())
    assert isinstance(restored.factor_value_ref, FactorValueRef)
    assert isinstance(restored.label_bundle_ref, LabelBundleRef)
    assert restored.factor_value_ref == req.factor_value_ref
    assert restored.label_bundle_ref == req.label_bundle_ref
    assert restored.tier == "core"
    assert restored.metric_ids == ("rank_ic", "coverage")


def test_ref_for_factor_values_pure_asserted_identity():
    ref = ref_for_factor_values("fv_src", ())
    assert isinstance(ref, FactorValueRef)
    assert ref.factor_value_id == "factor_value:raw:fv_src"
    with pytest.raises(ValueError):
        ref_for_factor_values("", ())


def test_label_bundle_to_ref_derives_and_carries():
    lb = _label_bundle(target_id="fwd_vwap_5d", horizon=5)
    ref = label_bundle_to_ref(lb)
    assert isinstance(ref, LabelBundleRef)
    assert ref.target_id == "fwd_vwap_5d"
    assert ref.horizon == 5
    assert ref.source_ref == "labels:snap1"
    # explicit asserted id wins.
    ref2 = label_bundle_to_ref(lb, asserted_label_bundle_id="lb_explicit")
    assert ref2.label_bundle_id == "lb_explicit"


def test_recipe_source_ref_string_accepted_as_asserted_id():
    # The recipe's source_factor_value_ref is a raw *string*; the adapter
    # accepts it as the asserted id without needing separate QE authority.
    recipe = _recipe(source_factor_value_ref="fv_zzz")
    req = request_for_recipe(recipe, label_bundle=_label_bundle())
    assert req.factor_value_ref is not None
    assert "fv_zzz" in req.factor_value_ref.factor_value_id
    assert "fv_zzz" in req.factor_value_ref.source_ref


# ---------------------------------------------------------------------------
# gap c: evidence computed/None/valid=False must NOT be treated as success
# ---------------------------------------------------------------------------

def test_assert_computed_value_accepts_real_metric_value():
    result = assert_computed_value(0.031, metric_id="rank_ic")
    assert result == pytest.approx(0.031)


def test_assert_computed_value_fails_closed_on_none():
    with pytest.raises(RuntimeError, match="not computed"):
        assert_computed_value(None, metric_id="rank_ic")


def test_assert_computed_value_fails_closed_on_valid_false():
    mv = MetricValue(
        metric_id="rank_ic", value=None, valid=False,
        observation_count=0, warnings=("label-not-mature",),
    )
    with pytest.raises(RuntimeError, match="valid=False"):
        assert_computed_value(mv, metric_id="rank_ic")


def test_assert_computed_value_fails_closed_on_non_finite_payload():
    mv = MetricValue(
        metric_id="rank_ic", value=float("nan"), valid=True,
        observation_count=3,
    )
    with pytest.raises(RuntimeError, match="not finite"):
        assert_computed_value(mv, metric_id="rank_ic", factor_id="f_A")


def test_recipe_ref_metadata_not_a_second_catalog():
    """The ref carries provenance only — no invented registry identity."""
    recipe = _recipe()
    ref = recipe_to_factor_value_ref(recipe, ref_for="treated")
    assert "registry" not in " ".join(ref.metadata.keys())
    assert "catalog" not in " ".join(ref.metadata.keys())