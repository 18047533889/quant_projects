"""DesirabilityPolicyRegistry (DLIB-FO-003).

The historical ``DEFAULT_MAPS`` in ``desirability.py`` hardcodes GLOBAL anchors
(RankIC 0.01-0.05, ICIR 0.2-0.4, Turnover 0.05-0.15).  Those are reasonable
bootstrap defaults but they are NOT universal economics: an A-share daily
PRICE_VOLUME factor and a US daily FUNDAMENTAL factor have very different
turnover desirability curves.

This module adds a ``DesirabilityPolicyRegistry`` keyed by a
``DesirabilityContext`` (market, asset_type, frequency, factor_family,
label_horizon, universe_profile, liquidity_tier, consumer_profile).  Each
search session FREEZES its ``desirability_policy_id`` so the anchors cannot
drift mid-search.  Calibration may be suggested from historical admitted
factors / raw distribution / trial distribution, but the frozen policy is what
the session uses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from factor_optimizer.search.desirability import (
    DEFAULT_MAPS,
    Anchor,
    Desirability,
    desirability_for,
)

# The context dimensions that key a desirability policy.  A policy is frozen
# for a search session by its ``policy_id``.
CONTEXT_KEYS = (
    "market",
    "asset_type",
    "frequency",
    "factor_family",
    "label_horizon",
    "universe_profile",
    "liquidity_tier",
    "consumer_profile",
)


@dataclass(frozen=True)
class DesirabilityContext:
    """The market/asset/frequency context that selects a desirability policy.

    Every field is a non-empty string.  ``None`` is not allowed: a policy must
    be keyed by an explicit context so a session cannot silently fall back to
    a global default that does not match its economics.
    """

    market: str
    asset_type: str
    frequency: str
    factor_family: str
    label_horizon: str
    universe_profile: str
    liquidity_tier: str
    consumer_profile: str

    def __post_init__(self) -> None:
        for name in CONTEXT_KEYS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"DesirabilityContext.{name} must be a non-empty string"
                )

    def to_dict(self) -> Dict[str, str]:
        return {name: getattr(self, name) for name in CONTEXT_KEYS}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DesirabilityContext":
        if not isinstance(data, dict):
            raise TypeError("DesirabilityContext.from_dict requires a dict")
        missing = set(CONTEXT_KEYS) - set(data)
        if missing:
            raise ValueError(
                f"DesirabilityContext missing keys: {sorted(missing)}"
            )
        return cls(**{name: data[name] for name in CONTEXT_KEYS})


@dataclass(frozen=True)
class DesirabilityPolicy:
    """A frozen set of metric->anchor maps for a specific context.

    Attributes:
        policy_id: Stable identifier for this policy (frozen per session).
        context: The context this policy was calibrated for.
        maps: metric -> {direction, anchors} (same shape as DEFAULT_MAPS).
        source: Where the anchors came from ("bootstrap_default",
            "historical_admitted", "raw_distribution", "trial_distribution",
            "manual").
    """

    policy_id: str
    context: DesirabilityContext
    maps: Dict[str, Dict[str, Sequence[Anchor]]]
    source: str = "bootstrap_default"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(self.context, DesirabilityContext):
            raise TypeError("context must be a DesirabilityContext")
        if not isinstance(self.maps, dict) or not self.maps:
            raise ValueError("maps must be a non-empty dict")
        for metric, spec in self.maps.items():
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("metric name must be a non-empty string")
            if not isinstance(spec, dict) or "direction" not in spec or "anchors" not in spec:
                raise ValueError(
                    f"map for metric {metric!r} must have 'direction' and 'anchors'"
                )
            if spec["direction"] not in ("increasing", "decreasing"):
                raise ValueError(
                    f"map for metric {metric!r} has unknown direction "
                    f"{spec['direction']!r}"
                )
            anchors = spec["anchors"]
            if not anchors:
                raise ValueError(f"map for metric {metric!r} has empty anchors")
            for x, d in anchors:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise TypeError(
                        f"anchor x for {metric!r} must be numeric, got {x!r}"
                    )
                if isinstance(d, bool) or not isinstance(d, (int, float)):
                    raise TypeError(
                        f"anchor d for {metric!r} must be numeric, got {d!r}"
                    )
                if not 0.0 <= float(d) <= 1.0:
                    raise ValueError(
                        f"anchor desirability for {metric!r} must be in [0, 1]"
                    )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

    def score(self, metric: str, value: float) -> float:
        """Desirability of ``value`` for ``metric`` under this policy."""
        if metric not in self.maps:
            raise KeyError(
                f"no desirability map for metric {metric!r} in policy "
                f"{self.policy_id!r}"
            )
        spec = self.maps[metric]
        return desirability_for(spec["direction"], spec["anchors"], value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "context": self.context.to_dict(),
            "maps": {
                metric: {
                    "direction": spec["direction"],
                    "anchors": [list(anchor) for anchor in spec["anchors"]],
                }
                for metric, spec in self.maps.items()
            },
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DesirabilityPolicy":
        if not isinstance(data, dict):
            raise TypeError("DesirabilityPolicy.from_dict requires a dict")
        maps = {
            metric: {
                "direction": spec["direction"],
                "anchors": [tuple(anchor) for anchor in spec["anchors"]],
            }
            for metric, spec in data["maps"].items()
        }
        return cls(
            policy_id=data["policy_id"],
            context=DesirabilityContext.from_dict(data["context"]),
            maps=maps,
            source=data.get("source", "bootstrap_default"),
        )


def _policy_id_for(context: DesirabilityContext, maps: Dict) -> str:
    """Deterministic policy id from the context + anchor payload."""
    payload = json.dumps(
        {
            "context": context.to_dict(),
            "maps": {
                metric: {
                    "direction": spec["direction"],
                    "anchors": [list(a) for a in spec["anchors"]],
                }
                for metric, spec in maps.items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DesirabilityPolicyRegistry:
    """Registry of desirability policies keyed by DesirabilityContext.

    A search session FREEZES its ``desirability_policy_id`` at the start and
    uses that policy for the whole session, so the anchors cannot drift
    mid-search.  The registry is append-only: a policy, once registered, is
    immutable and cannot be overwritten.
    """

    def __init__(self) -> None:
        self._policies: Dict[str, DesirabilityPolicy] = {}
        self._by_context: Dict[Tuple[str, ...], str] = {}

    def register(
        self,
        context: DesirabilityContext,
        maps: Optional[Dict[str, Dict[str, Sequence[Anchor]]]] = None,
        *,
        source: str = "bootstrap_default",
        policy_id: Optional[str] = None,
    ) -> DesirabilityPolicy:
        """Register (or return) a policy for ``context``.

        If a policy already exists for this context, it is returned unchanged
        (append-only: the existing anchors are never overwritten).  When
        ``maps`` is None, the bootstrap ``DEFAULT_MAPS`` are used.
        """
        if not isinstance(context, DesirabilityContext):
            raise TypeError("context must be a DesirabilityContext")
        key = tuple(getattr(context, name) for name in CONTEXT_KEYS)
        existing_id = self._by_context.get(key)
        if existing_id is not None:
            return self._policies[existing_id]
        maps = maps if maps is not None else DEFAULT_MAPS
        pid = policy_id or _policy_id_for(context, maps)
        policy = DesirabilityPolicy(
            policy_id=pid,
            context=context,
            maps=maps,
            source=source,
        )
        self._policies[pid] = policy
        self._by_context[key] = pid
        return policy

    def get(self, policy_id: str) -> DesirabilityPolicy:
        """Return the frozen policy with ``policy_id`` (fail-closed)."""
        if policy_id not in self._policies:
            raise KeyError(f"no desirability policy registered with id {policy_id!r}")
        return self._policies[policy_id]

    def get_for_context(self, context: DesirabilityContext) -> DesirabilityPolicy:
        """Return the policy for ``context``, or raise if none registered."""
        key = tuple(getattr(context, name) for name in CONTEXT_KEYS)
        pid = self._by_context.get(key)
        if pid is None:
            raise KeyError(
                f"no desirability policy registered for context "
                f"{context.to_dict()!r}"
            )
        return self._policies[pid]

    def freeze_session(self, context: DesirabilityContext) -> str:
        """Freeze a desirability_policy_id for a search session.

        Returns the policy_id.  The session must use this id for its entire
        run; the anchors cannot drift because the policy is immutable.
        """
        policy = self.register(context)
        return policy.policy_id

    def suggest_calibration(
        self,
        context: DesirabilityContext,
        *,
        historical_admitted: Optional[Sequence[Dict[str, float]]] = None,
        raw_distribution: Optional[Dict[str, Sequence[float]]] = None,
        trial_distribution: Optional[Dict[str, Sequence[float]]] = None,
    ) -> Dict[str, str]:
        """Suggest a calibration source for ``context``.

        Returns a dict of metric -> suggested source, based on which evidence
        is available.  This is advisory only: the frozen policy is what the
        session actually uses.
        """
        suggestions: Dict[str, str] = {}
        if historical_admitted:
            suggestions["rank_ic"] = "historical_admitted"
        if raw_distribution:
            suggestions["turnover"] = "raw_distribution"
        if trial_distribution:
            suggestions["icir"] = "trial_distribution"
        return suggestions

    def __len__(self) -> int:
        return len(self._policies)


__all__ = [
    "DesirabilityContext",
    "DesirabilityPolicy",
    "DesirabilityPolicyRegistry",
    "CONTEXT_KEYS",
]
