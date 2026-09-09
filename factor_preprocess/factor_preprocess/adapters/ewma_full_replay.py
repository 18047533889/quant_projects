"""Exact, bounded restart replay for the public lagged FP EWMA.

This adapter deliberately stores frozen input history and calls the existing
``transforms.ewma`` batch authority on every append.  It is FULL_REPLAY_ONLY,
not an O(1) sufficient-state implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from factor_preprocess.transforms import ewma


SCHEMA = "fp_ewma_full_replay.v1"
EXECUTION_MODE = "FULL_REPLAY_ONLY"
KERNEL_IMPLEMENTATION_SHA256 = hashlib.sha256(inspect.getsource(ewma).encode()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class FrozenEwmaReplaySpec:
    state_key: str
    source_identity: str
    feature_identity: str
    security_identity: str
    time_identity: str
    halflife: float
    min_periods: int = 1
    max_history_rows: int = 100_000

    def __post_init__(self) -> None:
        for name in ("state_key", "source_identity", "feature_identity", "security_identity", "time_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.halflife, bool) or not isinstance(self.halflife, (int, float)):
            raise ValueError("halflife must be a finite positive number")
        if not np.isfinite(float(self.halflife)) or float(self.halflife) <= 0:
            raise ValueError("halflife must be a finite positive number")
        for name in ("min_periods", "max_history_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def identity(self) -> str:
        return _digest({
            "schema": SCHEMA, "state_key": self.state_key,
            "source_identity": self.source_identity,
            "feature_identity": self.feature_identity,
            "security_identity": self.security_identity,
            "time_identity": self.time_identity,
            "halflife": float(self.halflife), "min_periods": int(self.min_periods),
            "max_history_rows": int(self.max_history_rows),
            "kernel_implementation_sha256": KERNEL_IMPLEMENTATION_SHA256,
        })


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("date must be explicitly UTC timezone-aware")
    if timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError("date timezone must be UTC")
    return timestamp.tz_convert("UTC")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["asset_id", "date", "value"]
    if list(frame.columns) != required:
        raise ValueError(f"segment must have exact columns {required}")
    out = frame.copy()
    out["date"] = [_utc_timestamp(value) for value in out["date"]]
    if out["date"].isna().any() or out["asset_id"].isna().any():
        raise ValueError("NaT and missing asset identities are forbidden")
    if not out["asset_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("asset_id must use the frozen non-empty string security identity")
    out["asset_id"] = out["asset_id"].astype(str)
    if out.duplicated(["asset_id", "date"]).any():
        raise ValueError("duplicate asset/date coordinate")
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if np.isinf(out["value"].to_numpy(dtype=float)).any():
        raise ValueError("infinite EWMA inputs are forbidden")
    return out


def _payload_without_hash(spec: FrozenEwmaReplaySpec, history: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in history.itertuples(index=False):
        value = None if pd.isna(row.value) else float(row.value)
        rows.append([str(row.asset_id), row.date.isoformat(), value])
    return {"schema": SCHEMA, "spec_identity": spec.identity, "rows": rows}


def _load(path: Path, spec: FrozenEwmaReplaySpec) -> pd.DataFrame:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("EWMA replay checkpoint is unreadable") from exc
    integrity = payload.pop("integrity_sha256", None)
    if integrity != _digest(payload):
        raise ValueError("EWMA replay checkpoint integrity mismatch")
    if payload.get("schema") != SCHEMA or payload.get("spec_identity") != spec.identity:
        raise ValueError("EWMA replay checkpoint identity mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("EWMA replay checkpoint rows are invalid")
    return _normalize(pd.DataFrame(rows, columns=["asset_id", "date", "value"]))


def _save(path: Path, spec: FrozenEwmaReplaySpec, history: pd.DataFrame) -> None:
    payload = _payload_without_hash(spec, history)
    payload["integrity_sha256"] = _digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(payload).decode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def apply_frozen_ewma_full_replay(
    segment: pd.DataFrame,
    *,
    spec: FrozenEwmaReplaySpec,
    checkpoint_path: str | Path,
    bootstrap: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """Append one immutable segment and return its exact lagged FP EWMA values."""
    current = _normalize(segment)
    requested = segment.copy()
    requested["date"] = [_utc_timestamp(value) for value in requested["date"]]
    requested["asset_id"] = requested["asset_id"].astype(str)
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("EWMA replay checkpoint has a concurrent writer") from exc
        if bootstrap:
            if path.exists():
                raise ValueError("bootstrap refuses to overwrite an existing checkpoint")
            history = current
        else:
            if not path.is_file():
                raise ValueError("resume requires an existing checkpoint")
            previous = _load(path, spec)
            watermarks = previous.groupby("asset_id", sort=False)["date"].max().to_dict()
            for asset, group in current.groupby("asset_id", sort=False):
                if asset in watermarks and group["date"].min() <= watermarks[asset]:
                    raise ValueError("non-append revision or duplicate timestamp is forbidden")
            history = _normalize(pd.concat([previous, current], ignore_index=True))
        if len(history) > spec.max_history_rows:
            raise ValueError("EWMA full-replay history budget exceeded")
        values = ewma(history, halflife=spec.halflife, min_periods=spec.min_periods)
        keyed = pd.Series(
            values.to_numpy(),
            index=pd.MultiIndex.from_frame(history[["asset_id", "date"]]),
        )
        requested_keys = pd.MultiIndex.from_frame(requested[["asset_id", "date"]])
        output = pd.Series(keyed.reindex(requested_keys).to_numpy(), index=segment.index)
        _save(path, spec, history)
    return output, {
        "execution_mode": EXECUTION_MODE,
        "spec_identity": spec.identity,
        "history_rows_replayed": len(history),
        "checkpoint_path": str(path),
    }
