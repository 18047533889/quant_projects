"""兼容 shim：storage.sources.composite_source"""
import sys
import storage.sources.composite_source as _mod
sys.modules[__name__] = _mod
