"""兼容 shim：storage.sources.kline_parquet_source"""
import sys
import storage.sources.kline_parquet_source as _mod
sys.modules[__name__] = _mod
