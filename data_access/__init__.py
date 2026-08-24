"""Repository-local `data_access` namespace shim.

The canonical dataaccess package source root is the flat `dataaccess/`
directory (top-level modules like ``cos_contract.py`` and subpackages like
``core/``, ``read/``, ``cos/`` sit directly inside it), and every internal
import uses the ``data_access`` dotted name (e.g. ``from data_access.core.engine
import ...``). To make the repo importable as the canonical ``data_access``
package without editing hundreds of internal import statements, this shim
forwards the ``data_access`` namespace to the ``dataaccess`` source directory —
the same ``"data_access" = "."`` mapping that the wheel matrix uses at build
time.

This mirrors the deployed wheel layout: the package source root is treated as
the ``data_access`` package, so ``data_access.cos_contract`` resolves to
``dataaccess/cos_contract.py`` and ``data_access.core`` to ``dataaccess/core/``.
"""
from __future__ import annotations

import sys as _sys
from importlib import util as _util

# Redirect the ``data_access`` package's import path to the flat source root.
_dataaccess = _util.find_spec("dataaccess")
if _dataaccess is not None and _dataaccess.origin is not None:
    import os as _os

    _root = _os.path.dirname(_dataaccess.origin)
    # Point every submodule resolver at the real source directory so that
    # ``data_access.cos_contract`` -> ``dataaccess/cos_contract.py`` etc.
    __path__ = [_root]  # type: ignore[name-defined]

# Make ``from data_access import X`` resolve through the source package.
from dataaccess import *  # noqa: F401,F403
from dataaccess import (  # noqa: F401
    __build_id__,
    __build_sha__,
    __build_time__,
    __version__,
)

# Mirror the subpackage namespace aliases so relative imports and re-exports
# (``data_access.core``) resolve to the same module objects.
_LOADED = frozenset(getattr(_sys, "modules", ()))
for _mod in list(_LOADED):
    if _mod.startswith("dataaccess."):
        _sys.modules.setdefault("data_access" + _mod[len("dataaccess"):], _sys.modules[_mod])
