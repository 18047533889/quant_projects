"""Exact rank-weighted and Benford CPU kernels; no Pandas panel conversion."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator, PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
SCORE = "ts_score_rank_weighted_mean"
BENFORD = "report_benford_js_divergence"

def reference(name):
    from factor_engine.cleaned_operators import advanced_information as source
    return (source.TsScoreRankWeightedMean if name == SCORE else source.ReportBenfordJsDivergence)()

def metadata(name):
    result = copy.deepcopy(reference(name).metadata)
    result.tags = [*result.tags, "numpy_kernel"]
    return result

def calculate(name, *args, **kwargs):
    import inspect
    from factor_engine.cleaned_operators import advanced_information as source
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    bound = inspect.signature(reference(name)._calculate_series).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    first = values["target" if name == SCORE else "amount"]
    w = strict_int(values["window"], "window", minimum=1 if name == SCORE else 10)
    if name == SCORE:
        second = values["score"]
        verify_frames_share_identity(name, first, second)
        decay = strict_finite_scalar(values["decay"], "decay", minimum=0, maximum=1)
        if decay <= 0:
            raise ValueError("decay must be strictly positive")
    result = []
    for c in first.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        a = first[c].to_numpy().astype(float)
        if name == SCORE:
            b = second[c].to_numpy().astype(float)
        out = np.full(len(a), np.nan)
        for t in range(len(a)):
            start = max(0, t-w+1)
            out[t] = (source._score_rank_weighted_mean(a[start:t+1], b[start:t+1], decay)
                if name == SCORE else source._benford_js(a[start:t+1]))
        result.append(pl.Series(c, out, dtype=pl.Float64))
    return first.with_columns(result)

def physical_spec(name):
    from factor_engine.cleaned_operators import advanced_information
    digest = hashlib.sha256(Path(advanced_information.__file__).read_bytes() +
        Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name, backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
        implementation_source_hash=digest, kernel_identity="advanced_information." + name,
        notes="Per-column NumPy CPU reference kernels; no Pandas panel conversion or GPU claim.")

def make(name):
    class InformationRankNative(SeriesOperator):
        metadata = globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self, *args, **kwargs):
            return calculate(name, *args, **kwargs)
        def physical_spec(self):
            return physical_spec(name)
    return InformationRankNative

def register(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    OperatorRegistry.register(make(name)(), canonical=name, backend="polars",
        source="information_rank_numpy", backend_explicit=True,
        status="experimental" if name == BENFORD else "implemented")
