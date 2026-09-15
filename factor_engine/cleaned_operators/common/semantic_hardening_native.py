"""Physical adapters for audited rolling reductions and grouped percentiles."""
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def calculate(name, x, **params):
    from factor_engine.cleaned_operators import semantic_hardening as source
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
    if name == "group_percentile":
        group = params.pop("group")
        if group is not None:
            verify_frames_share_identity(name, x, group)
        out = source._group_percentile_numpy(x.select(cols).to_numpy().astype(float),
            None if group is None else group.select(cols).to_numpy(), **params)
        return x.with_columns([pl.Series(c, out[:, i], dtype=pl.Float64) for i,c in enumerate(cols)])
    w, mp = source._validate_window(params["window"], params["min_periods"])
    if name == "ts_product":
        control = source._strict_skipna(params["skipna"])
        kernel = lambda values: source._stable_product(values, skipna=control)
    else:
        control = source._strict_scale(params["scale"])
        kernel = lambda values: source._median_abs_deviation(values, scale=control)
    result = []
    for c in cols:
        out = source._rolling_numpy_panel(x[c].to_numpy().astype(float).reshape(-1,1),
            window=w, min_periods=mp, func=kernel)
        result.append(pl.Series(c, out[:,0], dtype=pl.Float64))
    return x.with_columns(result)

def physical_spec(name):
    from factor_engine.cleaned_operators import semantic_hardening
    digest = hashlib.sha256(Path(semantic_hardening.__file__).read_bytes() +
        Path(__file__).read_bytes()).hexdigest()
    # The grouped implementation genuinely uses Pandas Series rank/grouping,
    # unlike the rolling reductions. Do not advertise it as Pandas-free.
    kind = (ExecutionKind.POLARS_PANDAS_DELEGATE if name == "group_percentile"
        else ExecutionKind.POLARS_NUMPY_KERNEL)
    return PhysicalImplementationSpec(canonical=name, backend="polars",
        execution_kind=kind, materializes_full_panel=True,
        supports_nan=True, supports_nulls=True, supports_inf=True,
        implementation_source_hash=digest, kernel_identity="semantic_hardening." + name,
        notes="CPU audited reductions; grouped rank uses Pandas Series. No GPU claim.")
