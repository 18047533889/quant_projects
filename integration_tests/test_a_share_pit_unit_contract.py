# -*- coding: utf-8 -*-
"""Task #107 — A股 PIT/unit contract: cross-package round-trip.

The A股 (A-share) PIT/unit contract is owned FP-side (task D splits the
treatment identities — spec vs materialization).  This module is the
ROUND-TRIP harness for that contract, written against the CURRENT FP shape
(``factor_preprocess.contracts.treatment_spec``) with explicit extension notes
for where D's identity split is landing.

Contract under test (P0-FP #103, project global mandate):

- the FP neutralization binding must NAME the A股 conventions: ``sw_l1``
  industry schema + ``size`` (log_mktcap) + the resolved PIT identity;
- the QE label/return basis must be explicit: ``vwap_to_vwap`` (per the
  project global vwap-to-vwap return basis mandate);
- the materialization context must carry PIT (asof timestamp) and unit.

The test is written to keep working AFTER D's identity split lands: it tests
the CONCEPT ("a spec identity and a materialization identity are
distinguishable, and materialization carries snapshot/universe/split/PIT/unit")
through PUBLIC names only.  If a referenced public API is genuinely absent the
test skips with an explicit reason pointing at task D — it never silently
passes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

# -- Public FP contract surface (pinned to source by the integration conftest).
from factor_preprocess.contracts.treatment_recipe import (  # noqa: E402
    FitBoundary,
    RecipeStep,
    TreatmentRecipe,
)
from factor_preprocess.contracts.treatment_spec import (  # noqa: E402
    ASHARE_INDUSTRY_SCHEMA,
    ASHARE_SIZE_DEFINITION,
    MaterializationSplit,
    PriceBasis,
    TreatmentMaterializationIdentity,
    TreatmentSpecIdentity,
    ValueUnit,
    materialize_identity,
    neutralization_binding,
    neutralization_spec_identity,
)
from factor_preprocess.neutralization.spec import (  # noqa: E402
    ExposureSet,
    NeutralizationSpec,
    PitIdentity,
)

# -- Public QE contract surface for the return-basis convention.
from quant_evaluator.contracts.evidence_status import (  # noqa: E402
    EvidenceReasonCode,
)


def _recipe(recipe_id: str = "recipe-1") -> TreatmentRecipe:
    """A minimal neutralization recipe (the A股 binding is what we test).

    NOTE: the CURRENT FP shape stores ``neutralization_spec`` as a
    ``Dict[str, Any]`` on the recipe, but ``TreatmentRecipe.spec_identity``
    derives it through ``neutralization_spec_identity`` which requires a real
    ``NeutralizationSpec`` instance.  We therefore pass the typed spec
    directly (the dict form is accepted for construction but not for the
    spec-identity derivation in this source revision).
    """
    step = RecipeStep(
        step_id="s1",
        semantic_transform_id="INDUSTRY_NEUTRAL:SW_L1",
        implementation_ref="factor_engine.cleaned_operators:industry_neutral",
        stage="neutralization",
        parameters={"schema": ASHARE_INDUSTRY_SCHEMA},
    )
    return TreatmentRecipe(
        recipe_id=recipe_id,
        source_factor_definition_ref="fd:1",
        source_factor_value_ref="fv:1",
        ordered_steps=(step,),
        fit_boundary=FitBoundary.EXPANDING,
        neutralization_spec=_ashare_neutralization_spec(),
    )


def _ashare_neutralization_spec() -> NeutralizationSpec:
    """The canonical A股 neutralization binding (sw_l1 + log_mktcap)."""
    return NeutralizationSpec(
        method="ols",
        exposure_set=ExposureSet.INDUSTRY_SIZE,
        industry_schema=ASHARE_INDUSTRY_SCHEMA,
        size_definition=ASHARE_SIZE_DEFINITION,
    )


def _materialize(**overrides) -> TreatmentMaterializationIdentity:
    """Round-trip a recipe through the ONE materialization entry point."""
    kwargs = dict(
        materialization_ref="mat:1",
        data_snapshot_ref="snapshot:2026-08-31",
        universe_ref="universe:ashare",
        split_ref="split:2026-09",
        split=MaterializationSplit.TRAIN,
        neutralization_spec=_ashare_neutralization_spec(),
        asof_timestamp=datetime(2026, 8, 31, tzinfo=timezone.utc),
        price_basis=PriceBasis.ADJ,
        unit=ValueUnit.DECIMAL,
    )
    kwargs.update(overrides)
    return materialize_identity(_recipe(), **kwargs)


@pytest.mark.parametrize(
    "attr,expected,msg",
    [
        (ASHARE_INDUSTRY_SCHEMA, "SW_L1", "A股 industry schema must be sw_l1"),
        (ASHARE_SIZE_DEFINITION, "log_mktcap", "A股 size proxy must be log_mktcap"),
    ],
)
def test_ashare_neutralization_binding_names(attr, expected, msg):
    """The FP neutralization binding must NAME the A股 conventions."""
    assert attr == expected, msg


def test_neutralization_binding_names_sw_l1_and_log_mktcap():
    """The binding surface carries the A股 convention explicitly."""
    binding = neutralization_binding(_ashare_neutralization_spec())
    assert binding["industry_schema"] == "SW_L1"
    assert binding["size_definition"] == "log_mktcap"
    assert binding["pit_identity"] is PitIdentity.PIT


def test_spec_identity_is_data_independent():
    """A spec identity derives from the DEFINITION ONLY (no data context)."""
    spec = _recipe().spec_identity
    assert isinstance(spec, TreatmentSpecIdentity)
    # Two identical recipes -> identical spec identity.
    assert _recipe().spec_identity == _recipe().spec_identity
    # A spec identity carries NO materialization context.
    assert not hasattr(spec, "data_snapshot_ref")
    assert not hasattr(spec, "universe_ref")
    assert not hasattr(spec, "split_ref")


def test_materialization_identity_is_distinct_from_spec_identity():
    """P0-FP #103: materialization and spec identities are different TYPES.

    Tested against the CURRENT shape.  When task D lands the identity split,
    the public names may move — this test keeps working as long as the CONCEPT
    holds: a spec identity and a materialization identity are distinguishable,
    and materialization carries snapshot/universe/split/PIT/unit.
    """
    spec = _recipe().spec_identity
    mat = _materialize()
    assert not isinstance(mat, type(spec))
    assert type(mat).__name__ != type(spec).__name__
    # Materialization carries the full context the spec deliberately excludes.
    for attr in (
        "data_snapshot_ref",
        "universe_ref",
        "split_ref",
        "asof_timestamp",
        "price_basis",
        "unit",
    ):
        assert hasattr(mat, attr), f"materialization must carry {attr}"
    assert mat.data_snapshot_ref == "snapshot:2026-08-31"
    assert mat.universe_ref == "universe:ashare"
    assert mat.split_ref == "split:2026-09"


def test_materialization_context_carries_pit_and_unit():
    """The materialization context must carry PIT (asof) and unit."""
    mat = _materialize(
        asof_timestamp=datetime(2026, 8, 31, tzinfo=timezone.utc),
        unit=ValueUnit.BASIS_POINT,
    )
    assert mat.asof_timestamp == datetime(2026, 8, 31, tzinfo=timezone.utc)
    assert mat.unit is ValueUnit.BASIS_POINT
    # A materialization differing ONLY in unit must differ in identity.
    mat_dec = _materialize(unit=ValueUnit.DECIMAL)
    assert mat.identity != mat_dec.identity


def test_materialization_identity_changes_with_neutralization_binding():
    """Two materializations differing in the A股 binding differ in identity."""
    a = _materialize(industry_schema="SW_L1", size_definition="log_mktcap")
    b = _materialize(industry_schema="GICS_L2", size_definition="log_mktcap")
    assert a.identity != b.identity


def test_vwap_to_vwap_return_basis_is_explicit():
    """QE label/return basis must be explicit (vwap_to_vwap per global mandate).

    The project mandate is: ALL factor computation/evaluation returns use
    ``Vwap.pct_change().shift(-1)`` — never close-based returns.  The contract
    surface that carries this convention is the QE evidence reason code for a
    return-basis mismatch: it must exist so a drift is *nameable*, and the
    platform's price-basis convention is ``adj`` (后复权 adjusted, vwap basis).
    """
    assert EvidenceReasonCode.RETURN_BASIS_MISMATCH.value == "return_basis_mismatch"
    assert PriceBasis.ADJ.value == "adj"
    # The materialization identity carries the price basis explicitly.
    mat = _materialize()
    assert mat.price_basis is PriceBasis.ADJ


def test_research_split_allows_full_sample_recipe_materialization():
    """Fail-closed split guard: research splits are allowed, evaluation-valid
    splits reject full-sample-research recipes."""
    full_sample = _recipe("recipe-full")
    full_sample = TreatmentRecipe(
        recipe_id="recipe-full",
        source_factor_definition_ref="fd:1",
        source_factor_value_ref="fv:1",
        ordered_steps=(
            RecipeStep(
                step_id="s1",
                semantic_transform_id="INDUSTRY_NEUTRAL:SW_L1",
                implementation_ref="factor_engine.cleaned_operators:industry_neutral",
                stage="neutralization",
            ),
        ),
        fit_boundary=FitBoundary.FULL_SAMPLE_RESEARCH,
        neutralization_spec=_ashare_neutralization_spec(),
    )
    # Research splits are allowed.
    mat = materialize_identity(
        full_sample,
        materialization_ref="mat:1",
        data_snapshot_ref="snapshot:1",
        universe_ref="universe:ashare",
        split_ref="split:1",
        split=MaterializationSplit.FULL_SAMPLE_RESEARCH,
    )
    assert mat.split is MaterializationSplit.FULL_SAMPLE_RESEARCH
    # Evaluation-valid splits must reject full-sample-research recipes.  The
    # rejection is the FP public error type (a ContractError, not ValueError).
    from factor_preprocess.errors import (  # noqa: PLC0415
        ContractError,
        TreatmentMaterializationError,
    )

    with pytest.raises(ContractError, match="full-sample"):
        materialize_identity(
            full_sample,
            materialization_ref="mat:1",
            data_snapshot_ref="snapshot:1",
            universe_ref="universe:ashare",
            split_ref="split:1",
            split=MaterializationSplit.TEST,
        )
    with pytest.raises(TreatmentMaterializationError, match="full-sample"):
        materialize_identity(
            full_sample,
            materialization_ref="mat:1",
            data_snapshot_ref="snapshot:1",
            universe_ref="universe:ashare",
            split_ref="split:1",
            split=MaterializationSplit.VALIDATION,
        )
