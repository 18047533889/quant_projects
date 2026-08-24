"""兼容 shim：storage.materialize.materializer"""
import sys
import factor_engine.storage.materialize.materializer as _mod
sys.modules[__name__] = _mod
