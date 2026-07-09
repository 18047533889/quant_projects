"""兼容 shim：runtime.quality.dq_profiles"""
import sys
import runtime.quality.dq_profiles as _mod
sys.modules[__name__] = _mod
