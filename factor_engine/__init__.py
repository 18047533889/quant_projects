"""factor_engine — umbrella namespace entry point.

This package is the top-level importable entry for the FactorEngine suite in
the ``quant-projects`` umbrella wheel.  The engine's actual code lives in the
top-level packages ``runtime``, ``api``, ``ir``, ``expr``, ``modeling``,
``backend``, ``cleaned_operators``, etc. (mirrored under ``factor_engine/``),
and those submodules import each other via top-level absolute imports.

This ``__init__`` is intentionally minimal: it must be importable in a clean
venv WITHOUT pulling in numpy/pandas or any heavy engine module.  Submodules
are imported lazily on demand (``import factor_engine.runtime`` etc.).
"""

__version__ = "0.3.1"

__all__ = ["__version__"]
