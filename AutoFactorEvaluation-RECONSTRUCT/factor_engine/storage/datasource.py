"""兼容 shim：storage.sources.datasource"""
import sys
import storage.sources.datasource as _mod
sys.modules[__name__] = _mod
