# -*- coding: utf-8 -*-
"""R55 #97 — quant-modeling packaging independence smoke tests.

``modeling`` must be an independently packaged distribution (quant-modeling,
own ``pyproject.toml``) whose namespace re-exports NOTHING from the sibling
evaluation / preprocessing packages (``quant_evaluator``, ``factor_preprocess``).
Every shared capability is modeling's own implementation or a ported contract —
never a vendored copy and never a re-export of another package's symbol.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import modeling
import modeling.learners

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELING_SRC = _REPO_ROOT / "modeling"

FOREIGN_PACKAGES = ("quant_evaluator", "factor_preprocess")


def test_top_level_api_imports_and_exposes_core_surface():
    assert modeling.__doc__ and "SINGLE SOURCE OF TRUTH" in modeling.__doc__
    for name in (
        "DecisionClock",
        "LabelContract",
        "TimingKind",
        "ModelExecutionClass",
        "SampleAdequacyContract",
        "ModelSemanticRegistry",
        "MODEL_SEMANTIC_REGISTRY",
        "PRODUCTION_LANES",
    ):
        assert hasattr(modeling, name), name
    # the learners subpackage is part of the published surface
    for learner in ("BaseLearner", "FrozenModel", "LearnerSpec", "PCRLearner", "PLSLearner"):
        assert hasattr(modeling.learners, learner), learner


def test_no_foreign_module_leaks_into_modeling_namespace():
    """No QE/FP module object may be reachable as ``modeling.<name>``."""
    leaked = [
        name
        for name, value in vars(modeling).items()
        if inspect.ismodule(value)
        and (name.startswith(FOREIGN_PACKAGES) or getattr(value, "__name__", "").startswith(FOREIGN_PACKAGES))
    ]
    assert leaked == [], f"foreign modules leaked into modeling namespace: {leaked}"


def test_modeling_source_never_imports_quant_evaluator_or_factor_preprocess():
    """grep-level guarantee: the shipped source never imports QE or FP."""
    offenders: list[str] = []
    for py in sorted(_MODELING_SRC.rglob("*.py")):
        rel = py.relative_to(_MODELING_SRC).as_posix()
        if "__pycache__" in rel or rel.startswith("build"):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from ", "import ")) and any(
                pkg in stripped for pkg in FOREIGN_PACKAGES
            ):
                offenders.append(f"{rel}: {stripped}")
    assert offenders == [], f"modeling imports a foreign package: {offenders}"


def test_homonym_symbols_are_modeled_own_not_reexported():
    """The three same-named symbols are modeling's OWN definitions (homonyms
    with different semantics), not re-exports of QE/FP."""
    import modeling.evaluation as evaluation
    import modeling.ledger as ledger

    assert ledger.FeatureBundle.__module__ == "modeling.ledger"
    assert ledger.stable_hash.__module__ == "modeling.ledger"
    assert evaluation.MetricValue.__module__ == "modeling.evaluation"


def test_package_is_self_contained_for_import():
    """``import modeling`` must not require factor_engine / QE / FP on sys.path.

    The modules that *optionally* use factor_engine resolve it lazily; this
    test proves no factor_engine import sits at MODULE top level (indent 0),
    which is what breaks a bare ``import modeling`` in a clean wheel env.
    """
    offenders: list[str] = []
    for py in sorted(_MODELING_SRC.rglob("*.py")):
        rel = py.relative_to(_MODELING_SRC).as_posix()
        if "__pycache__" in rel or rel.startswith("build"):
            continue
        in_type_checking = False
        for raw in py.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if stripped.startswith("if TYPE_CHECKING"):
                in_type_checking = True
            elif stripped and not raw.startswith((" ", "\t", "#")):
                in_type_checking = False
            if in_type_checking or raw.startswith((" ", "\t", "#")) or not stripped:
                continue  # function-level / TYPE_CHECKING / comment
            if stripped.startswith(("from factor_engine", "import factor_engine")):
                offenders.append(f"{rel}: {stripped}")
    assert offenders == [], (
        "module-level factor_engine import breaks wheel self-containment: "
        f"{offenders}"
    )