# -*- coding: utf-8 -*-
"""R6 P0-13: shared ``RecursiveKernel`` interface for full-history and
incremental/checkpointed execution.

The recursive recurrence math (EMA, Wilder RSI/ATR/ADX, EWM moments, MACD) is
implemented once in ``stateful_runtime.execute_stateful_segment`` and is proven
bit-exact against the full-history pandas implementations by
``tests/operators/test_r6_stateful_split_parity.py`` (every split point).  This
module gives that single implementation a first-class kernel interface so both
execution modes — ``bootstrap_full`` (full history from origin) and ``step`` /
``serialize`` / ``restore`` (incremental with checkpoints) — are explicitly the
same object, not two duplicated code paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from factor_engine.stateful_contract import StateCheckpoint, StatefulCheckpointRegistry
from factor_engine.stateful_runtime import execute_stateful_segment


@dataclass(frozen=True)
class RecursiveKernel:
    """One canonical's recursive kernel, shared by full-history and incremental.

    Attributes:
        canonical: the registered canonical name (``ts_ema``, ``RSI_WILDER`` …).
        minimum_history: rows required before the first meaningful output.
        checkpoint_fields: state fields the checkpoint persists.
    """

    canonical: str
    minimum_history: int
    checkpoint_fields: tuple[str, ...]

    @classmethod
    def for_canonical(cls, canonical: str) -> "RecursiveKernel":
        spec = StatefulCheckpointRegistry.get(canonical)
        if spec is None:
            raise ValueError(f"{canonical} has no stateful spec; not a recursive kernel")
        return cls(
            canonical=canonical,
            minimum_history=spec.minimum_history,
            checkpoint_fields=tuple(spec.checkpoint_fields),
        )

    # -- full history ------------------------------------------------------
    def bootstrap_full(
        self,
        inputs: Mapping[str, Sequence[Any]],
        *,
        timestamps: Sequence[Any],
        instrument: str,
        input_identity: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> "KernelRun":
        """Run from the dataset origin (no checkpoint) over the whole series."""
        result = execute_stateful_segment(
            self.canonical, inputs=inputs, timestamps=timestamps,
            instrument=instrument, input_identity=input_identity,
            params=params, checkpoint=None, starts_at_dataset_origin=True,
        )
        return KernelRun(values=result.values, checkpoint=result.checkpoint)

    # -- incremental -------------------------------------------------------
    def step(
        self,
        inputs: Mapping[str, Sequence[Any]],
        *,
        timestamps: Sequence[Any],
        instrument: str,
        input_identity: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
        checkpoint: StateCheckpoint | None = None,
        starts_at_dataset_origin: bool = False,
    ) -> "KernelRun":
        """Run one segment; with a ``checkpoint`` it resumes state, without it
        (and not at origin) the contract fails closed."""
        result = execute_stateful_segment(
            self.canonical, inputs=inputs, timestamps=timestamps,
            instrument=instrument, input_identity=input_identity,
            params=params, checkpoint=checkpoint,
            starts_at_dataset_origin=starts_at_dataset_origin,
        )
        return KernelRun(values=result.values, checkpoint=result.checkpoint)

    def serialize(self, checkpoint: StateCheckpoint) -> str:
        """Serialise a checkpoint for durable storage."""
        return checkpoint.to_json()

    @staticmethod
    def restore(payload: str) -> StateCheckpoint:
        """Restore a checkpoint from its serialised form."""
        return StateCheckpoint.from_json(payload)

    @staticmethod
    def validate_identity(
        checkpoint: StateCheckpoint,
        *,
        input_identity: Mapping[str, Any],
        instrument: str,
    ) -> None:
        """Reject a checkpoint whose source identity changed (R6 P0-14 / test D)."""
        StatefulCheckpointRegistry.validate(
            checkpoint, input_identity=input_identity, expected_instrument=instrument
        )


@dataclass(frozen=True)
class KernelRun:
    values: np.ndarray
    checkpoint: StateCheckpoint

    @property
    def state(self) -> Mapping[str, Any]:
        return self.checkpoint.state


__all__ = ["RecursiveKernel", "KernelRun"]
