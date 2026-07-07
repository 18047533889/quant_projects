from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .Database import DataBase as OfficialDataBase
from .Exposures import PortfolioExposures as OfficialPortfolioExposures
from .Exposures import PureExposures as OfficialPureExposures
from .FactorAnalyzer import FactorAnalyzer as OfficialFactorAnalyzer


HISTORY_MODULE_PACKAGE = "factor_layer.factor_evaluation_alphapurify.history_module"


@dataclass(frozen=True)
class SelectedRuntimeModules:
    database_cls: type[Any]
    portfolio_exposures_cls: type[Any]
    pure_exposures_cls: type[Any]
    factor_analyzer_cls: type[Any]
    factor_analyzer_module: ModuleType
    selections: dict[str, str]


def _load_history_module(module_name: str) -> ModuleType:
    return importlib.import_module(f"{HISTORY_MODULE_PACKAGE}.{module_name}")


def _select_database(module_name: str | None) -> tuple[type[Any], str]:
    if not module_name or module_name == "official":
        return OfficialDataBase, "official"
    module = _load_history_module(module_name)
    return getattr(module, "DataBase"), module_name


def _select_exposures(module_name: str | None) -> tuple[type[Any], type[Any], str]:
    if not module_name or module_name == "official":
        return OfficialPortfolioExposures, OfficialPureExposures, "official"
    module = _load_history_module(module_name)
    return getattr(module, "PortfolioExposures"), getattr(module, "PureExposures"), module_name


def _select_factor_analyzer(module_name: str | None) -> tuple[type[Any], ModuleType, str]:
    if not module_name or module_name == "official":
        return OfficialFactorAnalyzer, importlib.import_module(OfficialFactorAnalyzer.__module__), "official"
    module = _load_history_module(module_name)
    return getattr(module, "FactorAnalyzer"), module, module_name


def select_runtime_modules(config: Any) -> SelectedRuntimeModules:
    database_cls, database_selection = _select_database(getattr(config, "database_module", None))
    portfolio_exposures_cls, pure_exposures_cls, exposures_selection = _select_exposures(
        getattr(config, "exposures_module", None)
    )
    factor_analyzer_cls, factor_analyzer_module, factor_analyzer_selection = _select_factor_analyzer(
        getattr(config, "factor_analyzer_module", None)
    )
    return SelectedRuntimeModules(
        database_cls=database_cls,
        portfolio_exposures_cls=portfolio_exposures_cls,
        pure_exposures_cls=pure_exposures_cls,
        factor_analyzer_cls=factor_analyzer_cls,
        factor_analyzer_module=factor_analyzer_module,
        selections={
            "database": database_selection,
            "exposures": exposures_selection,
            "factor_analyzer": factor_analyzer_selection,
        },
    )
