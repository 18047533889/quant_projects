from importlib import import_module

__all__ = ["FactorEngine", "FactorEngineConfig", "load_config"]


def __getattr__(name: str):
	if name == "FactorEngine":
		return import_module("runtime.engine").FactorEngine
	if name in {"FactorEngineConfig", "load_config"}:
		module = import_module("runtime.config")
		return getattr(module, name)
	raise AttributeError(name)
