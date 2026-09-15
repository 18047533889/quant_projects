"""Exact Polars-to-pandas delegate for session recovery."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import SeriesOperator


def reference():
    from factor_engine.cleaned_operators.session_recovery import SessionEventRecoveryScore
    return SessionEventRecoveryScore()


def metadata():
    return copy.deepcopy(reference().metadata)


def calculate(*args, **kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import (
        to_pandas_panel,
        verify_frames_share_identity,
    )

    panels = [value for value in (*args, *kwargs.values()) if isinstance(value, pl.DataFrame)]
    if len(panels) < 2:
        raise TypeError("session_event_recovery_score requires x and event Polars panels")
    verify_frames_share_identity("session_event_recovery_score", *panels)
    convert = lambda value: to_pandas_panel(value) if isinstance(value, pl.DataFrame) else value
    out = reference().calculate(
        *(convert(value) for value in args),
        **{key: convert(value) for key, value in kwargs.items()},
    )
    # This is a genuine minute->daily row reduction, so the output cannot be
    # projected back onto the minute input axis.  Carry the daily axis as the
    # canonical bridge time-coordinate column.
    return pl.from_pandas(out.rename_axis("__fe_time__").reset_index())


def physical_spec():
    from factor_engine.cleaned_operators import session_recovery

    digest = hashlib.sha256(
        Path(session_recovery.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    return PhysicalImplementationSpec(
        canonical="session_event_recovery_score",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=digest,
        kernel_identity="session_recovery.session_event_recovery_score",
        notes="Exact full-panel conversion to the pandas authority; not native acceleration.",
    )


def register():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    class SessionRecoveryPolars(SeriesOperator):
        metadata = metadata()

        @property
        def _contract_callable(self):
            return reference()._calculate_series

        def _calculate_series(self, *args, **kwargs):
            return calculate(*args, **kwargs)

        def physical_spec(self):
            return physical_spec()

    OperatorRegistry.register(
        SessionRecoveryPolars(), canonical="session_event_recovery_score", backend="polars",
        source="session_recovery_reference_polars", status="implemented", backend_explicit=True,
    )
