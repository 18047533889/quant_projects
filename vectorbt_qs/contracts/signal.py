"""Signal contract for the vectorbt_qs backtest contract layer.

``SignalArtifact`` captures a decision timeline that the backtest simulator
consumes: when a signal was decided, its per-asset position targets, and the
policy that governs when it may be acted on.  It is a pure, frozen, stdlib-only
dataclass with a deterministic content hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _canonical_json(payload: Dict[str, Any]) -> str:
    """Deterministic JSON for content hashing (sorted keys, stable floats)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def content_hash_of(payload: Dict[str, Any]) -> str:
    """Hex SHA-256 of a canonical JSON payload."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SignalArtifact:
    """An immutable, content-addressed record of a decision-time signal.

    Attributes
    ----------
    signal_ref:
        Stable reference (e.g. strategy/signal id) this artifact is derived from.
    timestamps:
        Decision timestamps, one per rebalance event.
    position_targets:
        Position targets in column-major order matching ``universe`` ordering:
        ``[n_events, n_assets]`` of float weights (long-only for A-share).
    universe:
        Ordered ticker list.
    decision_time_policy:
        Policy naming when the signal is generated (e.g. ``"close"``).
    signal_available_policy:
        Policy naming when the signal becomes tradable (e.g. ``"next_open"``).
    content_hash:
        SHA-256 over the canonical payload.  Recomputable via :meth:`recompute_hash`.
    """

    signal_ref: str
    timestamps: List[str]
    position_targets: List[List[float]]
    universe_ids: List[str]
    decision_time_policy: str = "close"
    signal_available_policy: str = "next_open"
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.signal_ref:
            raise ValueError("signal_ref must be a non-empty string")
        if not self.timestamps:
            raise ValueError("timestamps cannot be empty")
        if not self.universe_ids:
            raise ValueError("universe_ids cannot be empty")
        n_events = len(self.timestamps)
        if len(self.position_targets) != n_events:
            raise ValueError(
                "position_targets rows must match timestamps length "
                f"({len(self.position_targets)} != {n_events})"
            )
        for row in self.position_targets:
            if len(row) != len(self.universe_ids):
                raise ValueError(
                    "each position_target row must match universe width "
                    f"({len(row)} != {len(self.universe_ids)})"
                )
        if not self.decision_time_policy.strip():
            raise ValueError("decision_time_policy must be non-empty")
        if not self.signal_available_policy.strip():
            raise ValueError("signal_available_policy must be non-empty")

    def raw_payload(self) -> Dict[str, Any]:
        return {
            "signal_ref": self.signal_ref,
            "timestamps": self.timestamps,
            "position_targets": self.position_targets,
            "universe_ids": self.universe_ids,
            "decision_time_policy": self.decision_time_policy,
            "signal_available_policy": self.signal_available_policy,
        }

    def recompute_hash(self) -> str:
        return content_hash_of(self.raw_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.raw_payload(),
            "content_hash": self.content_hash or self.recompute_hash(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SignalArtifact":
        obj = cls(
            signal_ref=payload["signal_ref"],
            timestamps=list(payload["timestamps"]),
            position_targets=[list(r) for r in payload["position_targets"]],
            universe_ids=list(payload["universe_ids"]),
            decision_time_policy=payload.get("decision_time_policy", "close"),
            signal_available_policy=payload.get(
                "signal_available_policy", "next_open"
            ),
            content_hash=payload.get("content_hash"),
        )
        expected = obj.recompute_hash()
        supplied = payload.get("content_hash")
        if supplied and supplied != expected:
            raise ValueError(
                f"content_hash mismatch for SignalArtifact {obj.signal_ref}: "
                f"supplied {supplied} != recomputed {expected}"
            )
        return obj


@dataclass(frozen=True)
class SignalSeriesArtifact:
    """A time-indexed signal ledger (one per event) — a thin, hash-addressed view."""

    signal_ref: str
    events: List[SignalArtifact]
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.signal_ref:
            raise ValueError("signal_ref must be non-empty")

    def recompute_hash(self) -> str:
        return content_hash_of(
            {
                "signal_ref": self.signal_ref,
                "events": [s.to_dict() for s in self.events],
            }
        )


__all__ = [
    "SignalArtifact",
    "SignalSeriesArtifact",
    "content_hash_of",
]
