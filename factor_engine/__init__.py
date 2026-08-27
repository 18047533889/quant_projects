"""P0-03: install the legacy-name alias finder as part of the package import.

``deprecated_shims`` is a pure shim and is imported lazily the first time a
legacy import is attempted; importing it from the package ``__init__`` makes
the alias resolution behaviour deterministic for any process that imports
``factor_engine`` (which every FE submodule does).  It carries no state.
"""

from . import deprecated_shims as _deprecated_shims  # noqa: F401

# Re-export the static mapping tables for tests/patching without importing the
# full engine.
_EXPOSE = None
__version__ = "0.3.1"
__all__ = ["__version__"]