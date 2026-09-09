"""Reference-only, bounded QE multi-instance job manifests."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


_TIERS = frozenset({"core", "extended", "research"})
_OUTPUT_MODE_ORDER = ("SUMMARY_ONLY", "SERIES", "FULL_DIAGNOSTIC")
_OUTPUT_MODES = frozenset(_OUTPUT_MODE_ORDER)


@dataclass(frozen=True)
class QEJobShard:
    factor_refs: tuple[str, ...]
    label_ref: str
    metric_instances: tuple[dict, ...]
    scenario_refs: tuple[tuple[str, str], ...]
    result_sink_ref: str
    shard_index: int
    tier: str = "core"
    cost_budget: float | None = None
    allowed_output_modes: tuple[str, ...] = (
        "SUMMARY_ONLY", "SERIES", "FULL_DIAGNOSTIC")

    def __post_init__(self):
        if not self.factor_refs or len(self.factor_refs) > 256:
            raise ValueError("QE shard factor_refs must contain 1..256 references")
        if not self.label_ref or not self.result_sink_ref:
            raise ValueError("label and result sink references are required")
        if self.tier not in _TIERS:
            raise ValueError(f"Unknown QE tier {self.tier!r}")
        if (self.cost_budget is not None and
                (isinstance(self.cost_budget, bool) or
                 not isinstance(self.cost_budget, (int, float)) or
                 not math.isfinite(self.cost_budget) or self.cost_budget < 0)):
            raise ValueError("cost_budget must be a finite nonnegative number or None")
        object.__setattr__(self, "allowed_output_modes", tuple(self.allowed_output_modes))
        if (not self.allowed_output_modes or
                len(set(self.allowed_output_modes)) != len(self.allowed_output_modes) or
                not set(self.allowed_output_modes) <= _OUTPUT_MODES):
            raise ValueError("allowed_output_modes must be unique supported output modes")

    def to_dict(self):
        return {"factor_refs": list(self.factor_refs), "label_ref": self.label_ref,
                "metric_instances": list(self.metric_instances),
                "scenario_refs": [list(x) for x in self.scenario_refs],
                "result_sink_ref": self.result_sink_ref, "shard_index": self.shard_index,
                "tier": self.tier, "cost_budget": self.cost_budget,
                "allowed_output_modes": list(self.allowed_output_modes)}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value["factor_refs"]), value["label_ref"],
                   tuple(value["metric_instances"]),
                   tuple(tuple(x) for x in value["scenario_refs"]),
                   value["result_sink_ref"], value["shard_index"],
                   value.get("tier", "core"), value.get("cost_budget"),
                   tuple(value.get("allowed_output_modes", _OUTPUT_MODE_ORDER)))


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _atomic_write(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_manifest(path, shard: QEJobShard):
    """Write an immutable content-addressed manifest; return its file ref."""
    payload = _canonical_bytes(shard.to_dict())
    digest = hashlib.sha256(payload).hexdigest()
    destination = Path(path) / f"qe_manifest_{digest}.json"
    if not destination.exists() or destination.read_bytes() != payload:
        _atomic_write(destination, payload)
    return str(destination)


def read_manifest(ref):
    path = Path(ref)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if path.name != f"qe_manifest_{digest}.json":
        raise ValueError("QE manifest path/content hash mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("QE manifest is not valid UTF-8 JSON") from exc
    return QEJobShard.from_dict(value)


def shard_factor_refs(factor_refs, *, shard_size=256):
    if type(shard_size) is not int or not 1 <= shard_size <= 256:
        raise ValueError("shard_size must be in [1, 256]")
    refs = tuple(factor_refs)
    return tuple(refs[i:i+shard_size] for i in range(0, len(refs), shard_size))


__all__ = ["QEJobShard", "write_manifest", "read_manifest", "shard_factor_refs"]
