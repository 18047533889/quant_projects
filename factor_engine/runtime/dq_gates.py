"""兼容 shim：runtime.quality.dq_gates"""
import sys
import runtime.quality.dq_gates as _mod
sys.modules[__name__] = _mod
