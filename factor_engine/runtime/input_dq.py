"""兼容 shim：runtime.quality.input_dq"""
import sys
import factor_engine.runtime.quality.input_dq as _mod
sys.modules[__name__] = _mod
