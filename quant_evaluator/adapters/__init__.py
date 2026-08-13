"""
Optional adapters for external platform integration.

These adapters are NOT imported by core quant_evaluator modules.
They provide Protocol-based boundaries and fail gracefully when
dependencies are missing.

Usage:
    from quant_evaluator.adapters.data_access import DataAccessAdapter
    from quant_evaluator.adapters.factor_engine import FactorEngineAdapter
    from quant_evaluator.adapters.pandas import PandasAdapter

All adapters raise OptionalDependencyMissing if their dependency is unavailable.
"""

__all__ = []  # Nothing exported by default; explicit imports required
