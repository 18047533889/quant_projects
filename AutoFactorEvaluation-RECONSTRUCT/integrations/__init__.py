"""AutoFactorEvaluation 与内嵌量化平台组件的统一适配层。"""

from __future__ import annotations

from . import quant_platform as _quant_platform


def bootstrap_quant_platform():
    """激活并返回当前 AutoFactorEvaluation 绑定的平台运行时。

    该兼容入口由 Gateway、Assetization、Evaluation 等历史模块调用。
    实际解析仍由 ``platform_bootstrap.activate_platform`` 负责，默认只使用
    本目录内嵌的 FactorEngine 与 DataAccess。
    """

    return _quant_platform.activate_platform()


# Direct imports such as
# ``from integrations.quant_platform import bootstrap_quant_platform`` first
# execute this package initializer. Publish the compatibility function on the
# submodule before downstream modules request it.
_quant_platform.bootstrap_quant_platform = bootstrap_quant_platform

from .quant_platform import (  # noqa: E402
    FactorExecution,
    InMemoryFrameSource,
    MarketDataSpec,
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
