"""Exact Polars adapters for R22 filter/topology pandas references."""

import hashlib
from pathlib import Path

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import OperatorMetadata


def reference(name):
    if name == "ts_causal_local_linear_smoother":
        from factor_engine.cleaned_operators.filter_smooth import TSCausalLocalLinearSmoother
        return TSCausalLocalLinearSmoother()
    if name == "ts_causal_savgol_endpoint":
        from factor_engine.cleaned_operators.technical.frequency_filters import CausalSavGolEndpoint
        return CausalSavGolEndpoint()
    if name == "ts_betti_1_max_persistence":
        from factor_engine.cleaned_operators.advanced_topology import TsBetti1MaxPersistence
        return TsBetti1MaxPersistence()
    raise KeyError(name)


def metadata(name):
    source = reference(name).metadata
    return OperatorMetadata(
        name=name,
        category=source.category,
        description=source.description,
        param_names=list(source.param_names),
        panel_params=source.panel_params,
        scalar_params=source.scalar_params,
        param_specs=dict(source.param_specs),
        relational_specs=list(source.relational_specs),
        tags=[*source.tags, "pandas_delegate"],
    )


def calculate(name, *args, **kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import (
        from_pandas_panel,
        to_pandas_panel,
        verify_frames_share_identity,
    )

    panels = [value for value in (*args, *kwargs.values()) if isinstance(value, pl.DataFrame)]
    if not panels:
        raise TypeError(f"{name}: a Polars panel is required")
    verify_frames_share_identity(name, *panels)

    def convert(value):
        return to_pandas_panel(value) if isinstance(value, pl.DataFrame) else value

    result = reference(name).calculate(
        *(convert(value) for value in args),
        **{key: convert(value) for key, value in kwargs.items()},
    )
    return from_pandas_panel(panels[0], result)


def physical_spec(name):
    from factor_engine.backend.evidence_provenance import semantic_hashes_for

    source = reference(name).__class__.__module__
    module = __import__(source, fromlist=["__file__"])
    digest = hashlib.sha256(
        Path(module.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    return PhysicalImplementationSpec(
        canonical=name,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=digest,
        emitter_identity="r22_reference_delegate:polars_to_pandas_to_polars:v1",
        kernel_identity=f"{source}.{reference(name).__class__.__name__}._calculate_series",
        semantic_contract_hash=semantic_hashes_for(name)["semantic_contract_hash"],
        notes="Exact eager pandas-reference delegation; not native Polars acceleration.",
    )
