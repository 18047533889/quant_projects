"""兼容 shim：storage.sources.parquet_source"""
import sys
import factor_engine.storage.sources.parquet_source as _mod
sys.modules[__name__] = _mod
