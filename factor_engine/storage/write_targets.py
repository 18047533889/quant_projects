"""兼容 shim：storage.materialize.write_targets"""
import sys
import factor_engine.storage.materialize.write_targets as _mod
sys.modules[__name__] = _mod
