"""Scoped execution identity for selected sparse layered decay."""
from __future__ import annotations
import hashlib
import inspect
import platform
import sys

def _source_hash(obj, label):
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        raise ValueError(f"cannot certify {label} source") from exc
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"cannot certify empty {label} source")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()

def build_layered_decay_execution_identity():
    import numpy as np
    import pandas as pd
    from factor_optimizer.adapters import layered_decay as adapter
    from factor_optimizer.adapters import layered_decay_long as sparse
    from factor_optimizer.adapters import repair_execution
    from factor_preprocess.transforms import layered_decay_state as state
    from quant_evaluator.metrics import quantile

    runner = getattr(sparse, "_execute_sparse_layered_decay_validated", None)
    state_class = getattr(state, "LayeredDecayState", None)
    quantile_fn = getattr(quantile, "assign_quantiles_batch", None)
    if not callable(runner) or not inspect.isclass(state_class) or not callable(quantile_fn):
        raise ValueError("selected layered-decay implementation is unavailable")
    methods = {name: getattr(state_class, name, None) for name in (
        "__init__", "step", "step_sparse", "_step_sparse_trusted", "_apply_inputs")}
    if any(not callable(method) for method in methods.values()):
        raise ValueError("selected LayeredDecayState method is unavailable")
    selected = {"runner": runner, "state_class": state_class,
        **{f"state_method:{name}": method for name, method in methods.items()},
        "selected_frame_validator": adapter._validate_frame,
        "qe_validate_tie_policy": quantile.validate_tie_policy,
        "qe_validate_quantile_count": quantile._validate_quantile_count,
        "qe_searchsorted_bins": quantile._searchsorted_bins,
        "assign_quantiles_batch": quantile_fn, "plan_execute": adapter.LayeredDecayPlan.execute}
    modules = {"plan_adapter": adapter, "sparse_runner": sparse,
        "state": state, "qe_quantile": quantile, "frame_validation": repair_execution}
    return {
        "scope": "selected_callables_and_containing_module_sources",
        "route": "factor_optimizer.adapters.layered_decay_long._execute_sparse_layered_decay_validated",
        "mapping_version": "layered-decay.v1",
        "selected": {name: {"module": getattr(obj, "__module__", None),
            "qualname": getattr(obj, "__qualname__", None),
            "source_sha256": _source_hash(obj, name)} for name, obj in selected.items()},
        "modules": {name: _source_hash(module, name) for name, module in modules.items()},
        "runtime_versions": {"python": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "numpy": np.__version__, "pandas": pd.__version__},
        "coverage_note": "Scoped source/version evidence, not a transitive dependency closure of NumPy, Pandas, QE, or imported helpers.",
    }
