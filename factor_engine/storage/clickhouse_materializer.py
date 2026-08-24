"""兼容 shim：storage.materialize.clickhouse_materializer"""
import sys
import factor_engine.storage.materialize.clickhouse_materializer as _mod
sys.modules[__name__] = _mod
