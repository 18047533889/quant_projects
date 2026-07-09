"""兼容 shim：storage.materialize.factor_matrix_materializer"""
import sys
import storage.materialize.factor_matrix_materializer as _mod
sys.modules[__name__] = _mod
