"""兼容 shim：runtime.quality.pit_audit"""
import sys
import runtime.quality.pit_audit as _mod
sys.modules[__name__] = _mod
