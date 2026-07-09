"""兼容 shim：runtime.quality.input_dq"""
import sys
import runtime.quality.input_dq as _mod
sys.modules[__name__] = _mod
