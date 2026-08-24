"""Factor health state machine.

Pure standard-library module (``enum``, ``dataclasses``, ``typing``,
``datetime``).

A factor's health is assessed each time we observe a fresh evaluation snapshot.
The monitor keeps a rolling history of (timestamp, IC) per factor and maps it
to one of a small set of coarse states using simple thresholds:

- ``DECAYING``   -- recent IC is below its own historical baseline by a margin.
- ``DRIFTED``    -- recent IC has turned negative / unstable (volatility up).
- ``STALE``      -- no fresh snapshot for too long.
- ``REVIVED``    -- was decayed/drifted, now back above baseline.
- ``FAILED``     -- IC collapsed below an absolute failure floor.
- ``HEALTHY``    -- otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from .artifacts import EvaluationBundle, _normalize_dt, utcnow


class FactorHealthState(Enum):
    HEALTHY = "healthy"
    DECAYING = "decaying"
    DRIFTED = "drifted"
    STALE = "stale"
    REVIVED = "revived"
    FAILED = "failed"


@dataclass
class FactorHealthMonitor:
    """Tracks per-factor health over evaluation snapshots.

    Parameters (configurable thresholds):
        stale_after_days   -- if no update for this long, factor is STALE.
        decay_ratio        -- recent mean IC < ratio * baseline => DECAYING.
        failure_floor      -- recent mean IC below this absolute value => FAILED.
        drift_floor        -- recent mean IC below this (often negative) => DRIFTED.
        history_keep       -- how many (time, ic) samples to remember.
        decay_window       -- how many recent samples form the "recent" set.
    """

    stale_after_days: float = 60.0
    decay_ratio: float = 0.5
    failure_threshold: float = -0.02
    drift_floor: float = -0.01
    history_keep: int = 200
    decay_window: int = 10

    _history: Dict[str, List[Tuple[datetime, float]]] = field(
        default_factory=dict
    )
    _last_state: Dict[str, FactorHealthState] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def update(
        self,
        factor_id: str,
        snapshot: EvaluationBundle,
        at: Optional[datetime] = None,
    ) -> FactorHealthState:
        """Register a fresh evaluation and return the new health state."""
        now = _normalize_dt(at) or utcnow()
        hist = self._history.setdefault(factor_id, [])
        hist.append((now, snapshot.ic))
        # keep bounded
        if len(hist) > self.history_keep:
            hist[:] = hist[-self.history_keep :]

        state = self._classify(factor_id, hist)
        self._last_state[factor_id] = state
        return state

    def _classify(self, factor_id: str, hist: List[Tuple[datetime, float]]) -> FactorHealthState:
        if not hist:
            return FactorHealthState.HEALTHY

        latest_ts = hist[-1][0]
        age_days = (utcnow() - latest_ts).total_seconds() / 86400.0
        if age_days > self.stale_after_days:
            return FactorHealthState.STALE

        ics = [ic for _, ic in hist]
        baseline = mean(ics)
        window = ics[-self.decay_window :]
        recent = mean(window)

        # REVIVED: the factor was previously FAILED/DRIFTED/DECAYING but has
        # now returned above its historical baseline (recent >= baseline).
        prev = self._last_state.get(factor_id)
        if prev is not None and prev in (
            FactorHealthState.FAILED,
            FactorHealthState.DRIFTED,
            FactorHealthState.DECAYING,
            FactorHealthState.STALE,
        ):
            if recent >= baseline and recent > 0:
                return FactorHealthState.REVIVED

        if recent < self.failure_threshold:
            return FactorHealthState.FAILED
        if recent < self.drift_floor:
            return FactorHealthState.DRIFTED
        if recent < baseline * self.decay_ratio:
            return FactorHealthState.DECAYING
        return FactorHealthState.HEALTHY

    def state(self, factor_id: str) -> FactorHealthState:
        """Last known state for a factor (HEALTHY if unseen)."""
        return self._last_state.get(factor_id, FactorHealthState.HEALTHY)

    def recent_ic(self, factor_id: str) -> Optional[float]:
        hist = self._history.get(factor_id)
        return hist[-1][1] if hist else None
