# -*- coding: utf-8 -*-
"""R35 Phase F: Numba kernel implementations.

Importing this package registers every compiled kernel with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.  Each kernel has a
canonical NumPy reference AND (when numba is installed) a compiled
``fastmath=False`` / ``nogil=True`` implementation.  Parity and benchmark
evidence are produced by the tests / evidence scripts.
"""
from __future__ import annotations

from backend.numba_kernels import ar_stateful  # noqa: F401  (registers kernels)
from backend.numba_kernels import kalman  # noqa: F401  (registers kernels)

__all__ = ["kalman", "ar_stateful"]
