"""兼容 shim：storage.sources.long_table_source"""
import sys
import storage.sources.long_table_source as _mod
sys.modules[__name__] = _mod
