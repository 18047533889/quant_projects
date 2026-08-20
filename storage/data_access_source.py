"""兼容 shim：storage.sources.data_access_source"""
import sys
import storage.sources.data_access_source as _mod
sys.modules[__name__] = _mod
