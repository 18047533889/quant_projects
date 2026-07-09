"""兼容 shim：storage.materialize.lake_publish"""
import sys
import storage.materialize.lake_publish as _mod
sys.modules[__name__] = _mod
