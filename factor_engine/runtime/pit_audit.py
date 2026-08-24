"""兼容 shim：runtime.quality.pit_audit"""
import sys
import factor_engine.runtime.quality.pit_audit as _mod
sys.modules[__name__] = _mod
