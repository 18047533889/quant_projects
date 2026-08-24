"""Runtime public surface.

Source-window contracts are installed idempotently before any lazy engine import.
"""

from importlib import import_module

from factor_engine.runtime.source_window_contract_v2 import install_source_window_contract

install_source_window_contract()

__all__ = ["FactorEngine", "FactorEngineConfig", "load_config"]


def __getattr__(name: str):
    if name == "FactorEngine":
        return import_module("factor_engine.runtime.engine").FactorEngine
    if name in {"FactorEngineConfig", "load_config"}:
        module = import_module("factor_engine.runtime.config")
        return getattr(module, name)
    raise AttributeError(name)
