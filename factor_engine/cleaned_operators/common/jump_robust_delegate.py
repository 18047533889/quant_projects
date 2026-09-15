"""Explicit Polars-to-pandas delegates for jump-robust estimators."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import SeriesOperator


def reference(name):
    from factor_engine.cleaned_operators import jump_robust as source
    classes = (source.IntradayMedRV, source.IntradayMinRV, source.IntradayJumpTestStat)
    return {cls.metadata.name: cls for cls in classes}[name]()


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
    convert = lambda value: to_pandas_panel(value) if isinstance(value, pl.DataFrame) else value
    out = reference(name).calculate(
        *(convert(value) for value in args),
        **{key: convert(value) for key, value in kwargs.items()},
    )
    return from_pandas_panel(panels[0], out)


def physical_spec(name):
    from factor_engine.cleaned_operators import jump_robust

    digest = hashlib.sha256(
        Path(jump_robust.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    return PhysicalImplementationSpec(
        canonical=name,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=digest,
        kernel_identity="jump_robust." + name,
        notes="Exact full-panel conversion to the pandas authority; not native acceleration.",
    )


def register(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    class JumpRobustPolars(SeriesOperator):
        metadata = copy.deepcopy(reference(name).metadata)

        @property
        def _contract_callable(self):
            return reference(name)._calculate_series

        def _calculate_series(self, *args, **kwargs):
            return calculate(name, *args, **kwargs)

        def physical_spec(self):
            return physical_spec(name)

    OperatorRegistry.register(
        JumpRobustPolars(), canonical=name, backend="polars",
        source="jump_robust_reference_polars", status="implemented", backend_explicit=True,
    )
