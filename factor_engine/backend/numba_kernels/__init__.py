# -*- coding: utf-8 -*-
"""R35 Phase F: Numba kernel implementations.

Importing this package registers every compiled kernel with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.  Each kernel has a
canonical NumPy reference AND (when numba is installed) a compiled
``fastmath=False`` / ``nogil=True`` implementation.  Parity and benchmark
evidence are produced by the tests / evidence scripts.
"""
from __future__ import annotations

from factor_engine.backend.numba_kernels import ar_stateful  # noqa: F401  (registers kernels)
from factor_engine.backend.numba_kernels import ewm_pairwise  # noqa: F401  (registers kernels)
from factor_engine.backend.numba_kernels import kalman  # noqa: F401  (registers kernels)
from factor_engine.backend.numba_kernels.rolling import (
    rolling_corr_panel,
    rolling_mean_panel,
    rolling_rank_pct_panel,
    rolling_std_panel,
)

__all__ = [
    "ar_stateful", "ewm_pairwise", "kalman",
    "rolling_corr_panel", "rolling_mean_panel",
    "rolling_rank_pct_panel", "rolling_std_panel",
]
