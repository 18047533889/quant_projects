"""兼容 shim：storage.sources.read_session"""
import sys
import storage.sources.read_session as _mod
sys.modules[__name__] = _mod
