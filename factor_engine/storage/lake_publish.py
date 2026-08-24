"""兼容 shim：storage.materialize.lake_publish"""
import sys
import factor_engine.storage.materialize.lake_publish as _mod
sys.modules[__name__] = _mod
