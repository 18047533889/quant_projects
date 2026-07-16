"""兼容 shim：storage.materialize.clickhouse_materializer"""
import sys
import storage.materialize.clickhouse_materializer as _mod
sys.modules[__name__] = _mod
