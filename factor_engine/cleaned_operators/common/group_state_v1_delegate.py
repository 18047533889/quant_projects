"""Exact Polars-to-pandas delegates for the group-state-v1 signals."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import SeriesOperator


def reference(name):
    from factor_engine.cleaned_operators.technical import group_state_v1 as source

    for spec_name, _params, _fn, _desc in source._SPECS:
        if spec_name == name:
            from factor_engine.cleaned_operators.registry import OperatorRegistry
            return OperatorRegistry.get(name, backend="pandas_numpy")
    raise KeyError(name)


def calculate(name, *args, **kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import (
        from_pandas_panel,
        to_pandas_panel,
        verify_frames_share_identity,
    )

    panels = [value for value in (*args, *kwargs.values()) if isinstance(value, pl.DataFrame)]
    if len(panels) < 2:
        raise TypeError(f"{name} requires two Polars panels")
    verify_frames_share_identity(name, *panels)
    convert = lambda value: to_pandas_panel(value) if isinstance(value, pl.DataFrame) else value
    out = reference(name).calculate(
        *(convert(value) for value in args),
        **{key: convert(value) for key, value in kwargs.items()},
    )
    return from_pandas_panel(panels[0], out)


def physical_spec(name):
    from factor_engine.cleaned_operators.technical import group_state_v1

    digest = hashlib.sha256(
        Path(group_state_v1.__file__).read_bytes() + Path(__file__).read_bytes()
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
        kernel_identity=f"technical.group_state_v1.{name}",
        notes="Exact full-panel pandas authority delegate; CPU only, not native acceleration.",
    )


def make(name):
    class GroupStateV1Polars(SeriesOperator):
        metadata = copy.deepcopy(reference(name).metadata)
        _physical_spec = physical_spec(name)

        @property
        def _contract_callable(self):
            return reference(name)._calculate_series

        def _calculate_series(self, *args, **kwargs):
            return calculate(name, *args, **kwargs)

        def physical_spec(self):
            return self._physical_spec

    return GroupStateV1Polars
