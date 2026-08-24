"""兼容 shim：storage.sources.cleaned_parquet_source"""
import sys
import factor_engine.storage.sources.cleaned_parquet_source as _mod
sys.modules[__name__] = _mod
