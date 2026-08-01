"""Cold-start coverage for PIT analyst-expectation operators."""
from __future__ import annotations

from .model import ColdStartFactor

# Event-level surprise transforms are the default for forward-filled realized
# financial panels.  The daily-observation z-score remains covered separately
# and is intentionally labelled compatibility_daily.
_FORMULAS = {
    "fin_surprise": (
        "fin_surprise(actual,expected,scale_base)",
        "event",
    ),
    "fin_surprise_zscore": (
        "fin_surprise_zscore(actual,expected,scale_base,252)",
        "compatibility_daily",
    ),
    "fin_surprise_event_zscore": (
        "fin_surprise_event_zscore(actual,expected,scale_base,period_id,8)",
        "event_history",
    ),
    "fin_surprise_event_percentile": (
        "fin_surprise_event_percentile(actual,expected,scale_base,period_id,8)",
        "event_history",
    ),
    "fin_expectation_revision": (
        "fin_expectation_revision(expected,target_period_id)",
        "revision_event",
    ),
    "fin_expectation_revision_pct": (
        "fin_expectation_revision_pct(expected,target_period_id)",
        "revision_event",
    ),
    "fin_expectation_revision_speed": (
        "fin_expectation_revision_speed(expected,target_period_id,60)",
        "revision_window",
    ),
    "fin_expectation_revision_count": (
        "fin_expectation_revision_count(expected,target_period_id,60)",
        "revision_window",
    ),
    "fin_expectation_revision_magnitude": (
        "fin_expectation_revision_magnitude(expected,target_period_id,60)",
        "revision_window",
    ),
    "fin_days_since_expectation_revision": (
        "fin_days_since_expectation_revision(expected,target_period_id,252)",
        "revision_age",
    ),
    "fin_expectation_dispersion": (
        "fin_expectation_dispersion(expected_std,expected_mean)",
        "dispersion",
    ),
    "fin_actual_expectation_divergence": (
        "fin_actual_expectation_divergence(actual,expected,scale_base)",
        "event",
    ),
    "fin_beat_streak": (
        "fin_beat_streak(actual,expected,period_id,8)",
        "event_history",
    ),
    "fin_miss_streak": (
        "fin_miss_streak(actual,expected,period_id,8)",
        "event_history",
    ),
}


def expectation_seeds(market: str) -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    return tuple(
        ColdStartFactor(
            factor_id=f"{market}_expectation_{name}",
            market=market,
            surface="extended",
            formula=formula,
            family="analyst_expectation",
            subfamily=subfamily,
            horizon=None,
            complexity="moderate",
            availability_tier="analyst",
            rationale=f"Generic PIT analyst-expectation transform {name}.",
            direction_hint="unknown",
            metadata={
                "source": "expectation_seeds",
                "operator": name,
                "requires_analyst_data": True,
                "event_semantics": subfamily != "compatibility_daily",
            },
        )
        for name, (formula, subfamily) in _FORMULAS.items()
    )
