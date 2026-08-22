"""兼容 shim：storage.sources.clickhouse_source"""
import sys
import storage.sources.clickhouse_source as _mod
sys.modules[__name__] = _mod
