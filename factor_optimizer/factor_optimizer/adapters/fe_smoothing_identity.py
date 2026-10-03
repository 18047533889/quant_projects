"""Execution identity for the FactorEngine-backed lagged-SMA adapter."""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
import platform

import numpy as np
import polars as pl
import pandas as pd


def _source_hash(function) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot certify the selected FE SMA function source") from exc
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _module_source_hash(function) -> str:
    source_path = inspect.getsourcefile(function)
    if not source_path:
        raise RuntimeError("cannot locate the selected FE SMA implementation module")
    try:
        source = Path(source_path).read_bytes()
    except OSError as exc:
        raise RuntimeError("cannot read the selected FE SMA implementation module") from exc
    return hashlib.sha256(source).hexdigest()


def build_fe_smoothing_identity() -> dict:
    """Bind the exact direct FE function dynamically called by the SMA adapter.

    The module source digest also covers FE's local helpers, including
    _lagged_rolling. Imported helpers outside this module are not a full
    transitive runtime closure and are deliberately documented as such.
    """
    from factor_optimizer.adapters.fe_smoothing import execute_lagged_sma

    long_smoothing = importlib.import_module("factor_engine.backend.long_smoothing")
    selected = getattr(long_smoothing, "lagged_mean", None)
    if not callable(selected):
        raise RuntimeError("FactorEngine long_smoothing.lagged_mean is unavailable")

    return {
        "selector": {
            "module": execute_lagged_sma.__module__,
            "qualname": execute_lagged_sma.__qualname__,
            "implementation_hash": _source_hash(execute_lagged_sma),
        },
        "selected_function": {
            "module": getattr(selected, "__module__", None),
            "qualname": getattr(selected, "__qualname__", None),
            "implementation_hash": _source_hash(selected),
            "module_source_hash": _module_source_hash(selected),
        },
        "versions": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "polars_version": pl.__version__,
        },
        "identity_scope": (
            "selected lagged_mean function and its source module; imported "
            "dependencies outside long_smoothing.py are not transitively hashed"
        ),
    }


__all__ = ["build_fe_smoothing_identity"]
