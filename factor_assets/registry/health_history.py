"""Caller-attested coverage metadata for point-in-time health replay."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TypeAlias

from factor_assets.contracts.lifecycle import HealthState


BaselineInput: TypeAlias = Mapping[str, HealthState] | Iterable[tuple[str, HealthState]]


def normalize_utc_timestamp(value: str, *, field_name: str = "timestamp") -> datetime:
    """Parse an ISO timestamp, interpreting legacy naive values as UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError as exc:
        raise ValueError(f"{field_name} is outside the supported UTC timestamp range") from exc


@dataclass(frozen=True, init=False)
class HealthHistoryCoverage:
    """A caller's completeness attestation for one health-event interval.

    Each baseline is the health state immediately before ``covered_from``;
    events exactly at that boundary are included in the covered interval.
    This metadata records the caller's claim and cannot establish external
    event-store completeness on its own.
    """

    covered_from: str
    covered_through: str
    baseline_health_by_factor: Mapping[str, HealthState]

    def __init__(
        self,
        covered_from: str,
        covered_through: str,
        baseline_health_by_factor: BaselineInput,
    ) -> None:
        start = normalize_utc_timestamp(covered_from, field_name="covered_from")
        end = normalize_utc_timestamp(covered_through, field_name="covered_through")
        if start > end:
            raise ValueError("covered_from must be less than or equal to covered_through")
        items = (
            baseline_health_by_factor.items()
            if isinstance(baseline_health_by_factor, Mapping)
            else baseline_health_by_factor
        )
        baselines: dict[str, HealthState] = {}
        try:
            iterator = iter(items)
        except TypeError as exc:
            raise TypeError("baseline_health_by_factor must be a mapping or iterable of pairs") from exc
        for pair in iterator:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError("baseline entries must be (factor_id, HealthState) pairs")
            factor_id, state = pair
            if not isinstance(factor_id, str) or not factor_id.strip():
                raise ValueError("baseline factor_id must be a non-empty string")
            if factor_id in baselines:
                raise ValueError(f"duplicate health baseline for {factor_id}")
            if not isinstance(state, HealthState):
                raise TypeError(f"baseline health for {factor_id} must be HealthState")
            baselines[factor_id] = state
        object.__setattr__(self, "covered_from", start.isoformat())
        object.__setattr__(self, "covered_through", end.isoformat())
        object.__setattr__(self, "baseline_health_by_factor", MappingProxyType(baselines))
