"""
Optional adapters for external platform integration.

These adapters are NOT imported by core quant_evaluator modules.
They provide Protocol-based boundaries and fail gracefully when
dependencies are missing.

Usage:
    from quant_evaluator.adapters.data_access import DataAccessAdapter
    from quant_evaluator.adapters.factor_engine import FactorEngineAdapter
    from quant_evaluator.adapters.pandas import PandasAdapter

``recipe_refs`` is the FP<->QE treatment-evaluation wiring: it converts a
``factor_preprocess`` :class:`TreatmentRecipe`'s ordered steps into QE
:class:`FactorValueRef` / :class:`LabelBundleRef` and composes the
:class:`EvaluationRequest` refs, plus the fail-closed
:func:`~.recipe_refs.assert_computed_value` guard that prevents a
``valid=False`` / label-not-mature result from being treated as success.

All adapters raise OptionalDependencyMissing if their dependency is unavailable.
"""

__all__ = ["recipe_refs"]  # recipe_refs is lazy-imported by consumers
