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
__all__ = ["__version__", "get_engine"]


def get_engine(*, profile_path=None, execution=None):
    """Return the v2 durable facade using the administrator's approved profile."""
    from .runtime.default_engine import get_engine as build_default_engine

    return build_default_engine(profile_path=profile_path, execution=execution)
