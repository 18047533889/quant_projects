# -*- coding: utf-8 -*-
"""State-maturity / survival primitives (2026-08 CTA pack).

These answer "how mature is the current sustained state" from the *completed*
episodes that finished before it.  The current (still-running) episode is
never included in the reference sample: the completed-run list is updated only
when a run *ends* (on the first inactive row after it), so an active row always
sees exactly the runs that ended in the past.

State is three-valued: ACTIVE (finite non-zero), INACTIVE (finite zero) and
UNKNOWN (NaN).  UNKNOWN is a data gap, never a state exit: it breaks and
re-baselines the run without recording a completed episode, so suspensions and
provider misses cannot pollute the completed-run history.

All three share one causal single-pass kernel; ``history_window`` caps the
number of most-recent completed runs kept.

* ``ts_state_age_percentile`` — ECDF of the current age among completed runs.
* ``ts_state_exit_hazard``    — smoothed historical hazard of the state ending
  at the current age (``(D_l + alpha) / (N_l + 2*alpha)``).
* ``ts_state_residual_life``  — expected remaining ``L - l`` among completed
  runs that lasted at least ``l``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like
from factor_engine.cleaned_operators.stateful._common import metadata

_EPS = 1e-12


def _survival_kernel(
    state_col: np.ndarray,
    history_window: int,
    min_pct: int,
    min_haz: int,
    min_res: int,
    alpha: float,
    *,
    inactive_policy: str = "nan",
    gap_policy: str = "nan",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal single-pass survival statistics for one column.

    Returns ``(age, pct, hazard, residual)`` arrays.  ``age`` is the current
    active-run length; on an inactive row it is ``0`` under
    ``inactive_policy="zero"`` and ``NaN`` (not-applicable) under the default
    ``inactive_policy="nan"`` (R24-109 — an inactive row is NOT a zero-length
    episode).  Each statistic is NaN while its completed-run sample is below the
    corresponding minimum.

    R24-105..109:
    * LEFT-CENSOR: a run whose true start predates the sample (or whose entry
      was not observed — first row active, or entry inside a provider gap) is
      ``left_censored`` and is NEVER added to the completed-run distribution
      unless a real episode start was observed.
    * GAP-CENSOR: a provider gap inside an active run makes its length unknown;
      the run is not recorded and the exact age stays NaN (``gap_policy="nan"``)
      or is reported as a ``lower_bound`` floor.
    * INACTIVE: ``inactive_policy="nan"`` (R24-109, default) emits NaN on
      inactive rows (they are NOT applicable, not a 0-percentile outcome);
      ``inactive_policy="zero"`` keeps the legacy 0 reading.
    """
    if inactive_policy not in ("nan", "zero"):
        raise ValueError(f"inactive_policy must be 'nan' or 'zero', got {inactive_policy!r}")
    if gap_policy not in ("nan", "lower_bound"):
        raise ValueError(f"gap_policy must be 'nan' or 'lower_bound', got {gap_policy!r}")
    rows = state_col.shape[0]
    age = np.zeros(rows, dtype=float)
    pct = np.zeros(rows, dtype=float)
    hazard = np.zeros(rows, dtype=float)
    residual = np.zeros(rows, dtype=float)
    completed: list[float] = []
    cur = 0
    prev_active = False
    prev_finite = False
    run_observed_entry = False   # True when the current run's start was observed
    gap_censored = False         # True when a provider gap broke the run
    for row in range(rows):
        s = state_col[row]
        if not np.isfinite(s):
            # P0-005 + R24-107/108: a missing state is UNKNOWN, not a state
            # exit.  A data gap inside an active run makes the run length
            # unknown (gap_censored): it is never recorded as a completed
            # episode, and the age is NaN (or a lower_bound under gap_policy).
            if cur > 0 and prev_active:
                gap_censored = True
            # R30 §17 (P0-012): a provider gap UNKNOWNS the run state.  Reset the
            # episode counter so a later ACTIVE row starts a NEW (left-censored)
            # episode at age 1 instead of inheriting the pre-gap age.  Without
            # this, ``0,1,NaN,1,0`` reported the post-gap episode as continuing at
            # age 2 from the pre-gap run — a gap silently merged two episodes.
            cur = 0
            age[row] = np.nan
            pct[row] = np.nan
            hazard[row] = np.nan
            residual[row] = np.nan
            prev_active = False
            prev_finite = False
            continue
        active = s != 0.0
        if active:
            if cur == 0:
                # New run begins.  The entry is OBSERVED only if the previous
                # row was a confirmed inactive row; otherwise the run is
                # left-censored (R24-105) — never counted as completed.
                run_observed_entry = prev_finite and not prev_active
            cur += 1
            if gap_censored:
                # R24-108: exact age is unknown after a provider gap.
                age[row] = float(cur) if gap_policy == "lower_bound" else np.nan
            else:
                age[row] = float(cur)
            comp = np.asarray(completed, dtype=float)
            # percentile
            if comp.size >= min_pct:
                pct[row] = float(np.sum(comp <= cur)) / comp.size
            else:
                pct[row] = np.nan
            # hazard
            if comp.size >= min_haz:
                d_l = float(np.sum(comp == cur))
                n_l = float(np.sum(comp >= cur))
                # P1-77: gate on the *age-specific* risk set N_l, not just the
                # total completed-run count.  A single survivor reaching age l
                # must not produce a hazard estimate (hazard would read as a
                # robust probability when it is really 1 sample).
                if n_l >= min_haz:
                    hazard[row] = (d_l + alpha) / (n_l + 2.0 * alpha)
                else:
                    hazard[row] = np.nan
            else:
                hazard[row] = np.nan
            # residual life
            reached = comp[comp >= cur]
            if reached.size >= min_res:
                residual[row] = float(np.mean(reached - cur))
            else:
                residual[row] = np.nan
        else:
            if prev_active:
                # Run ended.  Only an observed-entry, non-gap-censored run is a
                # COMPLETED episode (R24-106/108).
                if run_observed_entry and not gap_censored:
                    completed.append(float(cur))
                    if len(completed) > history_window:
                        completed.pop(0)
                cur = 0
            if inactive_policy == "nan":
                # R24-109: inactive is NOT a 0-percentile outcome.
                age[row] = np.nan
                pct[row] = np.nan
                hazard[row] = np.nan
                residual[row] = np.nan
            else:
                age[row] = 0.0
                pct[row] = 0.0
                hazard[row] = 0.0
                residual[row] = 0.0
            gap_censored = False
            run_observed_entry = False
        prev_active = active
        prev_finite = True
    return age, pct, hazard, residual


@register_operator(
    name="ts_state_age_percentile",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_state_age_percentile",
    source="stateful.survival",
)
class TsStateAgePercentile(SeriesOperator):
    """ECDF of the current state age among previously *completed* episodes.

    ``0.95`` means the current episode has already lasted longer than 95% of
    the completed ones in memory.  Output 0 when not in an active state; NaN
    when the completed-run sample is below ``min_completed_runs``.
    """

    metadata = metadata(
        "ts_state_age_percentile",
        "当前状态年龄在历史已结束 episode 中的经验分位。",
        ["state", "history_window", "min_completed_runs", "inactive_policy", "gap_policy"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(
        self,
        state: pd.DataFrame,
        history_window: int = 60,
        min_completed_runs: int = 5,
        inactive_policy: str = "nan",
        gap_policy: str = "nan",
        **_: Any,
    ) -> pd.DataFrame:
        hw = max(2, int(history_window))
        mc = max(2, int(min_completed_runs))
        sv = state.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, pct, _, _ = _survival_kernel(
                sv[:, col], hw, mc, mc, mc, 1.0,
                inactive_policy=inactive_policy, gap_policy=gap_policy,
            )
            out[:, col] = pct
        return frame_like(state, out)


@register_operator(
    name="ts_state_exit_hazard",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_state_exit_hazard",
    source="stateful.survival",
)
class TsStateExitHazard(SeriesOperator):
    """Smoothed historical hazard of the state ending at the current age.

    ``h(l) = (D_l + alpha) / (N_l + 2*alpha)`` where ``D_l`` counts completed
    runs that ended exactly at ``l`` and ``N_l`` counts those that reached
    ``l``.  Output 0 when not in an active state; NaN when the sample is below
    ``min_completed_runs``.
    """

    metadata = metadata(
        "ts_state_exit_hazard",
        "当前状态年龄的历史退出风险(平滑 hazard)。",
        ["state", "history_window", "min_completed_runs", "alpha", "inactive_policy", "gap_policy"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(
        self,
        state: pd.DataFrame,
        history_window: int = 60,
        min_completed_runs: int = 5,
        alpha: float = 1.0,
        inactive_policy: str = "nan",
        gap_policy: str = "nan",
        **_: Any,
    ) -> pd.DataFrame:
        hw = max(2, int(history_window))
        mc = max(2, int(min_completed_runs))
        al = max(0.0, float(alpha))
        sv = state.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, _, hazard, _ = _survival_kernel(
                sv[:, col], hw, mc, mc, mc, al,
                inactive_policy=inactive_policy, gap_policy=gap_policy,
            )
            out[:, col] = hazard
        return frame_like(state, out)


@register_operator(
    name="ts_state_residual_life",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_state_residual_life",
    source="stateful.survival",
)
class TsStateResidualLife(SeriesOperator):
    """Expected remaining life ``E[L - l | L >= l]`` over completed episodes.

    Requires at least ``min_completed_runs`` completed runs that reached the
    current age, else NaN.  Output 0 when not in an active state.
    """

    metadata = metadata(
        "ts_state_residual_life",
        "当前状态的期望剩余寿命(基于已完成 episode)。",
        ["state", "history_window", "min_completed_runs", "inactive_policy", "gap_policy"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(
        self,
        state: pd.DataFrame,
        history_window: int = 60,
        min_completed_runs: int = 20,
        inactive_policy: str = "nan",
        gap_policy: str = "nan",
        **_: Any,
    ) -> pd.DataFrame:
        hw = max(2, int(history_window))
        mc = max(2, int(min_completed_runs))
        sv = state.to_numpy(dtype=float)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, _, _, residual = _survival_kernel(
                sv[:, col], hw, mc, mc, mc, 1.0,
                inactive_policy=inactive_policy, gap_policy=gap_policy,
            )
            out[:, col] = residual
        return frame_like(state, out)


def _register_surface() -> None:
    from factor_engine.cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        ["ts_state_age_percentile", "ts_state_exit_hazard", "ts_state_residual_life"]
    )


_register_surface()


# Round-11 #12 (#3): the survival trio are EPISODE-stateful — every active-row
# output depends on the *historical completed-run distribution* (accumulated
# since dataset origin) plus the current run's age, so a chunk computed without
# that state diverges from the full run.  Declare the execution contract as
# required_full_history (honest fail-closed): no segmented checkpoint-restore
# exists for these single-pass kernels, so we never claim ``checkpoint``.
from factor_engine.runtime.execution_contract import declare_stateful  # noqa: E402

for _survival_canonical in (
    "ts_state_age_percentile",
    "ts_state_exit_hazard",
    "ts_state_residual_life",
):
    declare_stateful(
        _survival_canonical,
        state_model="episode",
        chunking="required_full_history",
    )
