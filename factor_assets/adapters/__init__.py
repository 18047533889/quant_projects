"""
Optional adapters for integrating FA with FactorEngine, QuantEvaluator, and DataAccess.

These are NOT required dependencies — they enable optional integration when
the corresponding packages are available. Core FA functionality works without them.

Available adapters:
    - quant_evaluator: EvidenceProvider for QE integration
    - residual_novelty: Residual-IC conditional-novelty producer (QE wiring
      behind SelectionPolicy.make_decision's novelty_score parameter)
    - factor_engine: FactorIdentityProvider for FE integration
    - data_access: Optional DA read integration (future)

Import patterns:
    try:
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
    except OptionalDependencyMissing:
        # Handle gracefully — QE integration not available
        pass
"""

from typing import Optional


class OptionalDependencyMissing(ImportError):
    """Raised when an optional adapter dependency is not available."""

    def __init__(self, package_name: str, adapter_name: str):
        self.package_name = package_name
        self.adapter_name = adapter_name
        # Extras are intentionally adapter-specific; ``adapters`` remains the
        # aggregate install for callers that want every integration.
        extra_name = {
            "quant_evaluator": "quant_evaluator",
            "factor_engine": "factor_engine",
            "data_access": "data_access",
        }.get(package_name, "adapters")
        super().__init__(
            f"Adapter '{adapter_name}' requires optional package '{package_name}'. "
            f"Install with: pip install factor_assets[{extra_name}]"
        )


__all__ = [
    "OptionalDependencyMissing",
]
