"""兼容 shim：storage.sources.datasource"""
import sys
import factor_engine.storage.sources.datasource as _mod
sys.modules[__name__] = _mod
