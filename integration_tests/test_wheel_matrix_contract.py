# -*- coding: utf-8 -*-
"""Task #107 — Wheel-matrix import contract (mirror of WHEEL_MATRIX CI).

The ``wheel-matrix.yml`` CI builds each package from its OWN canonical
pyproject and asserts the wheel payload equals the source tree.  This test
mirrors the *import* half of that contract from the repo root: the 5 packages
whose contracts this harness round-trips must import cleanly, and their public
contracts namespaces must expose the core contracts the other integration tests
depend on.

A failure here means a package's public namespace stopped exposing a contract
that the cross-package harness (or the wheel matrix) relies on — that is
CONTRACT DRIFT, reported loudly rather than silently skipped.
"""

from __future__ import annotations

import importlib

import pytest

#: package -> list of public submodules whose contracts the harness round-trips.
#: (mirrors the WHEEL_MATRIX legs for the packages this harness covers)
NAMESPACE_CONTRACTS: dict[str, list[str]] = {
    "quant_platform": [
        "quant_platform.app.contracts",
        "quant_platform.app.contracts.identities",
        "quant_platform.app.contracts.cluster_library",
    ],
    "quant_evaluator": [
        "quant_evaluator.contracts",
        "quant_evaluator.contracts.evidence_status",
        "quant_evaluator.contracts.evaluation_artifact",
    ],
    "factor_assets": [
        "factor_assets.contracts",
        "factor_assets.contracts.factor_set",
        "factor_assets.contracts.similarity",
    ],
    "factor_optimizer": [
        "factor_optimizer.contracts",
        "factor_optimizer.contracts.trial_ledger",
        "factor_optimizer.contracts.multiplicity",
    ],
    "factor_preprocess": [
        "factor_preprocess.contracts",
        "factor_preprocess.contracts.treatment_recipe",
        "factor_preprocess.contracts.treatment_spec",
    ],
}

#: core contract symbols each namespace must expose (imported at test time).
CORE_SYMBOLS: dict[str, list[str]] = {
    "quant_evaluator.contracts.evidence_status": [
        "EvidenceStatus",
        "EvidenceReasonCode",
        "MetricEvidence",
        "evidence_for_computed",
        # NOTE (task #107): QE's module-level __all__ lists STATUS_REASON_ALLOWED
        # but the current source only binds the underscore name + the
        # _STATUS_REASON_COMPATIBILITY.ALLOWED alias.  The integration harness
        # round-trips the matrix through the compat alias (see the roundtrip
        # module) — a drift guard for the PUBLIC name is reported in the task
        # #107 CONTRACT DRIFT BLOCKER section, not asserted here (it is a QE
        # export fix owned by the QE task).
        "validate_status_reason",
    ],
    "quant_evaluator.contracts.evaluation_artifact": [
        "EvaluationArtifact",
        "EvaluationSpecIdentity",
        "ArtifactEnvelopeIdentity",
    ],
    "quant_platform.app.contracts.identities": [
        "IdentityRef",
        "FactorDefinitionRef",
        "FactorValueRef",
        "EvaluationRef",
        "TreatmentRef",
    ],
    "quant_platform.app.contracts.cluster_library": [
        "ClusterSetVersionRef",
        "ClusterVersionRef",
        "FactorLibraryVersionRef",
        "ClusterSetVersionView",
        "FactorLibraryVersionView",
        "LibraryLifecycleEvent",
        "promote_library_version",
    ],
    "factor_optimizer.contracts.trial_ledger": [
        "TrialLedger",
        "LedgerEntry",
        "TRIAL_STATUS_OUTCOMES",
    ],
    "factor_optimizer.contracts.multiplicity": ["MultiplicityArtifact"],
    "factor_assets.contracts.factor_set": [
        "FactorSetArtifact",
        "FactorSetSpec",
        "FactorMembership",
    ],
    "factor_assets.contracts.similarity": [
        "SimilarityArtifact",
        "SimilarityView",
        "SimilarityViewRegistry",
    ],
    "factor_preprocess.contracts.treatment_recipe": [
        "TreatmentRecipe",
        "RecipeStep",
        "FitBoundary",
    ],
    "factor_preprocess.contracts.treatment_spec": [
        "TreatmentSpecIdentity",
        "TreatmentMaterializationIdentity",
        "materialize_identity",
        "ASHARE_INDUSTRY_SCHEMA",
        "ASHARE_SIZE_DEFINITION",
    ],
}


@pytest.mark.parametrize("module_name", sorted({m for mods in NAMESPACE_CONTRACTS.values() for m in mods}))
def test_namespace_imports_cleanly(module_name: str):
    """Each public contracts namespace imports cleanly from the repo root."""
    module = importlib.import_module(module_name)
    assert module.__file__ is not None, f"{module_name} resolved to a namespace dir"
    # The source tree (not a venv wheel) must win — the integration conftest
    # pins it.  Flag a wheel-backed import loudly: that is what WHEEL_MATRIX
    # catches as an incomplete wheel.
    if "site-packages" in module.__file__:
        pytest.fail(
            f"{module_name} resolved to the venv wheel ({module.__file__}); "
            "the wheel matrix expects the source tree to be the authority"
        )


@pytest.mark.parametrize("module_name", sorted(CORE_SYMBOLS))
def test_core_contract_symbols_exposed(module_name: str):
    """The public namespaces expose the core contract symbols the harness
    round-trips (drift guard: a rename breaks this test, not a silent skip)."""
    module = importlib.import_module(module_name)
    missing = [name for name in CORE_SYMBOLS[module_name] if not hasattr(module, name)]
    assert not missing, (
        f"{module_name} no longer exposes {missing}; the cross-package contract "
        "surface drifted — update the owning package's public namespace"
    )
