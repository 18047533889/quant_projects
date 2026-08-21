"""ObjectiveSpec: first-class, frozen, serializable search objective contract.

The objective of a search is more than a bare direction string.  This module
defines ``ObjectiveSpec`` — the single authoritative description of what a
search is optimizing (which metric, in which direction).  It is:

- **frozen**: constructed with ``(metric_name, direction)`` and immutable;
- **validated fail-closed**: unknown directions and empty metric names raise
  ``ValueError`` at construction (and again on deserialization);
- **serializable**: ``to_dict`` / ``from_dict`` round-trips through JSON;
- **picklable**: plain frozen dataclass content; and
- **content-addressable**: equality/hash by value, so two specs built from
  the same payload are interchangeable.

``SearchConfig`` keeps the historical bare ``objective_direction`` field as a
backward-compatible convenience, but the spec is the authority: the runner's
direction is derived from the spec, never from a parallel field.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal

# Valid objective directions.  Reused by SearchConfig's convenience field.
OBJECTIVE_DIRECTIONS = ("maximize", "minimize")
ObjectiveDirection = Literal["maximize", "minimize"]

# Default metric name when only a direction is given.
DEFAULT_METRIC_NAME = "score"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Frozen, validated contract naming the search target metric and direction.

    Attributes:
        metric_name: Name of the metric being optimized (non-empty string).
        direction: ``"maximize"`` or ``"minimize"``.
    """

    metric_name: str
    direction: ObjectiveDirection

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str):
            raise ValueError(
                f"objective metric_name must be a non-empty string, "
                f"got {type(self.metric_name).__name__}"
            )
        if not self.metric_name.strip():
            raise ValueError("objective metric_name must be a non-empty string")
        if self.direction not in OBJECTIVE_DIRECTIONS:
            raise ValueError(
                f"objective direction must be 'maximize' or 'minimize', "
                f"got {self.direction!r}"
            )

    def to_dict(self) -> Dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {
            "metric_name": self.metric_name,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectiveSpec":
        """Deserialize (and re-validate fail-closed) from a dict."""
        if not isinstance(data, dict):
            raise TypeError(
                f"ObjectiveSpec.from_dict requires a dict, got {type(data).__name__}"
            )
        metric = data.get("metric_name", DEFAULT_METRIC_NAME)
        direction = data.get("direction")
        if direction is None:
            raise ValueError(
                "ObjectiveSpec.from_dict requires a 'direction' "
                "('maximize' or 'minimize')"
            )
        return cls(metric_name=metric, direction=direction)

    @classmethod
    def from_direction(cls, direction: str) -> "ObjectiveSpec":
        """Build the spec for the default metric from a bare direction string.

        Used by ``SearchConfig`` to reconcile the legacy ``objective_direction``
        convenience field with the authoritative spec.
        """
        return cls(metric_name=DEFAULT_METRIC_NAME, direction=direction)
