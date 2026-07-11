"""AutoFactorEvaluation 与 quant_projects 平台组件的统一适配层。"""

from .quant_platform import (
    FactorExecution,
    InMemoryFrameSource,
    MarketDataSpec,
    bootstrap_quant_platform,
    build_factor_engine_config,
    execute_factor_formula,
    execute_factor_on_frame,
    get_market_data_spec,
    load_market_frame,
    materialize_factor_to_staging,
    validate_factor_formula,
)

__all__ = [
    "FactorExecution",
    "InMemoryFrameSource",
    "MarketDataSpec",
    "bootstrap_quant_platform",
    "build_factor_engine_config",
    "execute_factor_formula",
    "execute_factor_on_frame",
    "get_market_data_spec",
    "load_market_frame",
    "materialize_factor_to_staging",
    "validate_factor_formula",
]
