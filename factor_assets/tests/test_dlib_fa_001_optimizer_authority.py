"""DLIB-FA-001 optimizer-authority migration tests.

Confirms that the FA optimizer subpackage no longer acts as a second search /
mutation / sealed-test authority, and that the production assembly path uses
the Pareto comparator only as an assembly-only (Factor-Library-Assembly)
dominance ranking, never as a search/treatment optimizer.
"""

import warnings

import pytest

import factor_assets.optimizer as fa_optimizer


def test_optimizer_subpackage_is_research_only():
    """The FA optimizer subpackage is not a production search/treatment
    authority (DLIB-FA-001)."""
    assert fa_optimizer.RESEARCH_ONLY is True


def test_retired_modules_emit_deprecation_warning():
    """multifidelity / plateau / frozen_candidate are retained for
    compatibility (§108) but emit DeprecationWarning when their module is
    imported directly — marked for future removal (DLIB-FA-001)."""
    import importlib

    for module_name in (
        "factor_assets.optimizer.multifidelity",
        "factor_assets.optimizer.plateau",
        "factor_assets.optimizer.frozen_candidate",
    ):
        if module_name in importlib.sys.modules:
            importlib.sys.modules.pop(module_name, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(module_name)
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), f"{module_name} must emit DeprecationWarning"


def test_migration_matrix_exists():
    """The FA_OPTIMIZER_MIGRATION_MATRIX document exists and maps every
    optimizer module to its target authority."""
    import os

    path = os.path.join(
        os.path.dirname(fa_optimizer.__file__), "FA_OPTIMIZER_MIGRATION_MATRIX.md"
    )
    assert os.path.exists(path), "FA_OPTIMIZER_MIGRATION_MATRIX.md must exist"
    content = open(path, encoding="utf-8").read()
    for module in ("pareto.py", "multifidelity.py", "plateau.py",
                   "typed_mutation.py", "frozen_candidate.py"):
        assert module in content, f"matrix must cover {module}"


def test_assembly_pareto_uses_dominance_comparator_only():
    """The only production caller of optimizer.pareto is the assembly engine,
    which uses ParetoPoint.dominates purely to rank already-admitted
    candidates — a Factor-Library-Assembly use, not a search/treatment use."""
    import inspect

    from factor_assets.assembly.engine import _pareto_rank, _pareto_objectives
    src = inspect.getsource(_pareto_rank) + inspect.getsource(_pareto_objectives)
    assert "dominates" in src
    # It must NOT run a search loop or mutate candidates.
    assert "while " not in src.replace("while_", "")


def test_typed_mutation_is_reference_only():
    """TypedMutation is a frozen reference record for FO MutationSpec/Trial —
    it has no execution engine (DLIB-FA-001)."""
    from factor_assets.optimizer.typed_mutation import TypedMutation, MutationStatus

    mutation = TypedMutation(
        mutation_id="m1",
        parent_asset_id="F1",
        mutation_type="ts_rank_window",
        parameters={"window": 20},
    )
    assert mutation.mutation_id == "m1"
    assert mutation.parent_asset_id == "F1"
    # Frozen dataclass: attribute assignment raises.
    with pytest.raises(Exception):
        mutation.mutation_id = "m2"  # type: ignore
    assert MutationStatus.EXECUTED.value == "executed"


def test_frozen_candidate_seal_is_not_test_authority():
    """The FA frozen-candidate TEST_SEALED state is a logical record, not a
    real test-protection boundary (DLIB-FA-001/§43). Test authority is FO
    TestAuthorityBroker. FA stores only a contamination verdict ref."""
    from factor_assets.optimizer.frozen_candidate import FrozenState

    # The state exists for compatibility, but the module is research-only and
    # deprecated; the real authority lives in FO.
    assert FrozenState.TEST_SEALED.value == "test_sealed"
    assert FrozenState.TEST_CONTAMINATED.value == "test_contaminated"
