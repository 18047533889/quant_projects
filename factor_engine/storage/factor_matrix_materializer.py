"""兼容 shim：storage.materialize.factor_matrix_materializer"""
import sys

import storage.materialize.factor_matrix_materializer as _mod

# 显式 re-export，便于静态检查与 ``from storage.factor_matrix_materializer import ...``
from storage.materialize.factor_matrix_materializer import (  # noqa: F401
    DuplicateMatrixKeyError,
    FactorMatrixConcurrentWriteError,
    FactorMatrixCorruptionError,
    FactorMatrixLayout,
    FactorMatrixMaterializer,
    FactorMatrixReadError,
    FactorMatrixVersionMismatchError,
)

sys.modules[__name__] = _mod
