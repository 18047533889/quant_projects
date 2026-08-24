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
# Not a bare ``from dataaccess import *``: when this shim is first imported it
# is triggered from *inside* ``dataaccess/__init__.py`` (line 4), so the source
# package is only partially initialized and a star-import would capture an
# incomplete attribute set (e.g. ``reset_store``, bound later at source line 44,
# or the build dunders at line 125).  Everything is therefore forwarded lazily.
def __getattr__(name: str):
    """Forward any attribute through the canonical ``dataaccess`` package.

    ``dataaccess`` imports this shim at the very top of its own ``__init__``,
    so when a caller reaches for ``data_access.X`` the source package may still
    be mid-initialization.  Deferring resolution until attribute access means
    by the time the name is actually read the canonical module is fully loaded
    (or, for names still being defined, we rely on the caller not racing the
    import).  This covers both the build-identity dunders and every public name
    (e.g. ``reset_store``, ``get_store``).
    """
    import dataaccess as _src  # noqa: PLC0415

    try:
        return getattr(_src, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
