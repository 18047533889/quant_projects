"""兼容 shim：storage.sources.parquet_source"""
import sys
import storage.sources.parquet_source as _mod
sys.modules[__name__] = _mod
