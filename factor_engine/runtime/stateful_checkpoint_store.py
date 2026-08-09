# -*- coding: utf-8 -*-
"""Persistent checkpoint store for stateful incremental factor execution.

Audit §11: recursive (stateful) operators can be resumed from a per-instrument
checkpoint instead of re-reading the full causal history.  This store persists
``StateCheckpoint`` JSON sidecars under the factor lake, keyed by factor id,
operator canonical and instrument, and returns the latest checkpoint at or
before a watermark.  A missing, stale, or invalid checkpoint fails closed to
``None`` so the caller falls back to full-history replay — never a wrong value.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from stateful_contract import (
    CheckpointSerializationError,
    StateCheckpoint,
    StatefulCheckpointRegistry,
    StatefulContractError,
)


def _safe_key(value: Any) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class StatefulCheckpointStore:
    """File-backed checkpoint registry scoped to one factor lake."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            from workspace_paths import default_factor_lake_root

            root = Path(default_factor_lake_root()) / "stateful_checkpoints"
        self.root = Path(root)

    def _dir(self, factor_id: str) -> Path:
        return self.root / _safe_key(factor_id)

    def _path(self, factor_id: str, canonical: str, instrument: str) -> Path:
        return self._dir(factor_id) / f"{_safe_key(canonical)}__{_safe_key(instrument)}.json"

    def load_latest(
        self,
        factor_id: str,
        canonical: str,
        instrument: str,
        *,
        before: Any,
    ) -> StateCheckpoint | None:
        """Return the latest valid checkpoint with ``as_of`` strictly before ``before``."""
        spec = StatefulCheckpointRegistry.get(canonical)
        if spec is None:
            return None
        path = self._path(factor_id, canonical, instrument)
        if not path.is_file():
            return None
        try:
            checkpoint = StateCheckpoint.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, TypeError, json.JSONDecodeError, CheckpointSerializationError):
            return None
        if checkpoint.operator != canonical:
            return None
        cutoff = _as_utc(before)
        try:
            as_of = _as_utc(checkpoint.as_of)
        except (ValueError, TypeError):
            return None
        # execute_stateful_segment requires the segment to start strictly after
        # the checkpoint, so only strictly-before checkpoints are usable.
        if not as_of < cutoff:
            return None
        try:
            StatefulCheckpointRegistry.validate(checkpoint)
        except StatefulContractError:
            return None
        # NEW-P0-30 (defense-in-depth): a checkpoint whose record claims coverage
        # up to/through a bar the state did not actually cover (as_of !=
        # state.last_timestamp) is a false coverage claim — fail closed so a
        # direct resume cannot silently interpolate forward over the gap.
        state_last = (checkpoint.state or {}).get("last_timestamp")
        if state_last is not None:
            try:
                if _as_utc(state_last) != _as_utc(checkpoint.as_of):
                    return None
            except (ValueError, TypeError):
                return None
        return checkpoint

    def save(self, factor_id: str, checkpoint: StateCheckpoint) -> None:
        """Persist a single validated checkpoint atomically.

        NOTE (audit #377): this is a single-checkpoint write — it is atomic for
        one path but NOT a multi-file transaction.  When several checkpoints must
        be committed together (one per instrument in a segmented batch) use
        ``commit_batch`` so a failure leaves every official path untouched.
        """
        StatefulCheckpointRegistry.validate(checkpoint)
        path = self._path(factor_id, checkpoint.operator, checkpoint.instrument)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(checkpoint.to_json(), encoding="utf-8")
        tmp.replace(path)

    def commit_batch(
        self,
        factor_id: str,
        checkpoints: list[StateCheckpoint],
        *,
        execution_id: str | None = None,
    ) -> None:
        """Atomically persist a batch of checkpoints: all committed or none.

        Every checkpoint is validated and serialized *before* any disk write.
        Then each payload is written to a unique temporary sidecar (under a
        ``.tmp`` sibling directory, names disambiguated by a UUID so concurrent
        commits never collide) and only after ALL temporaries have been written
        successfully are they renamed into place with ``os.replace``.  A failure
        at any point removes the temporaries already written and re-raises,
        leaving the official checkpoint paths untouched (fail-closed atomicity:
        the 1001st instrument failing to serialize does not leave the first 1000
        persisted).
        """
        if not checkpoints:
            return
        # Phase 1 — validate + serialize everything before touching the paths.
        staged: list[tuple[Path, str]] = []
        for checkpoint in checkpoints:
            StatefulCheckpointRegistry.validate(checkpoint)
            path = self._path(factor_id, checkpoint.operator, checkpoint.instrument)
            staged.append((path, checkpoint.to_json()))

        tmp_paths: list[Path] = []
        try:
            # Phase 2 — write every payload to a unique temporary sidecar.
            for path, payload in staged:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_dir = path.parent / ".tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{execution_id}-" if execution_id else ""
                tmp = tmp_dir / f"{prefix}{uuid.uuid4().hex}-{path.name}.tmp"
                tmp.write_text(payload, encoding="utf-8")
                tmp_paths.append(tmp)
            # Phase 3 — only after every temporary is durable, move them in.
            for tmp, (path, _) in zip(tmp_paths, staged):
                os.replace(tmp, path)
        except Exception:
            for tmp in tmp_paths:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def latest_as_of(self, factor_id: str, canonical: str, instrument: str) -> Any:
        """Return the stored checkpoint's as_of (for diagnostics) or None."""
        checkpoint = self._load_any(factor_id, canonical, instrument)
        return None if checkpoint is None else checkpoint.as_of

    def _load_any(
        self, factor_id: str, canonical: str, instrument: str
    ) -> StateCheckpoint | None:
        path = self._path(factor_id, canonical, instrument)
        if not path.is_file():
            return None
        try:
            return StateCheckpoint.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, TypeError, json.JSONDecodeError, CheckpointSerializationError):
            return None


__all__ = ["StatefulCheckpointStore"]
