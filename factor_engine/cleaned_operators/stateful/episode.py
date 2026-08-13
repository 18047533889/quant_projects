# -*- coding: utf-8 -*-
"""Dynamic-episode / intrinsic-time primitives (2026-08 CTA pack).

* ``state_since_sum``         — sum of ``x`` over the current episode
  ``[tau, t]`` where ``tau`` is the last reset (dynamic episode window, unlike
  a fixed rolling window).  ``state_since_mean`` / ``state_since_count`` /
  ``state_since_last`` are the mean / count / last-value variants.  These four
  replace the retired ``state_since_reduce(mode=...)`` so each mode is an
  honest canonical with a fixed output unit (sum/mean/last carry ``x``'s unit,
  count is ``count``) and the mode-enum search dimension is gone.
* ``directional_change_state`` — binary/ternary Directional-Change event state:
  +1 while an up event is running, -1 while a down event runs, 0 before the
  first event confirms.
* ``directional_change_extent`` — normalized progress of the current DC event
  (signed by direction; magnitude = number of ``threshold`` moves since the
  event's origin, i.e. overshoot accumulates from the event start).
* ``state_since_trend_tstat``  — OLS slope t-statistic of ``x`` over the
  current episode since the last reset (dimensionless).

All are forward per-column recursions (prefix-causal, PIT).  A NaN input emits
NaN and re-baselines the recursion state (``missing_policy="break"``).  Every
``reset_condition`` slot is a ConditionBool: finite values must be exactly
0/1 (anything else raises), NaN is an unknown reset that re-baselines.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import assert_condition_bool, metadata

_EPS = 1e-12


def _state_since_kernel(
    x: pd.DataFrame,
    reset_condition: pd.DataFrame,
    min_episode: int,
    mode: str,
) -> pd.DataFrame:
    """Shared episode-reduce kernel over ``[tau, t]`` since the last reset.

    ``mode`` is one of ``{"sum", "mean", "count", "last"}`` (callers pre-validate
    ``reset_condition`` as a ConditionBool).  Reset rows (``reset_condition == 1``,
    and NaN reset input) emit NaN and start a new episode; a NaN ``x`` emits NaN
    that day but does not break the episode accumulation (the day exists, the
    value is missing).  Output is NaN until the episode has ``min_episode``
    valid rows.
    """
    xv = x.to_numpy(dtype=float)
    rv = reset_condition.to_numpy(dtype=float)
    min_e = max(1, int(min_episode))
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        acc = 0.0
        cnt = 0
        last = np.nan
        for row in range(rows):
            if not np.isfinite(rv[row, col]):
                out[row, col] = np.nan
                acc = 0.0
                cnt = 0
                last = np.nan
                continue
            if bool(rv[row, col] != 0.0):
                out[row, col] = np.nan
                acc = 0.0
                cnt = 0
                last = np.nan
                continue
            xt = xv[row, col]
            if not np.isfinite(xt):
                # P1-73: a missing x emits NaN that day (the docstring
                # contract) instead of continuing to print the old episode
                # statistics.  The episode accumulation itself is not
                # broken — acc/cnt/last are carried forward unchanged.
                out[row, col] = np.nan
                continue
            acc = acc + float(xt)
            cnt = cnt + 1
            last = float(xt)
            if cnt < min_e:
                out[row, col] = np.nan
                continue
            if mode == "sum":
                out[row, col] = acc
            elif mode == "mean":
                out[row, col] = acc / float(cnt)
            elif mode == "count":
                out[row, col] = float(cnt)
            else:  # last
                out[row, col] = last
    return frame_like(x, out)


@register_operator(
    name="state_since_sum",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_since_sum",
    source="stateful.episode",
)
class StateSinceSum(SeriesOperator):
    """Sum of ``x`` over the current episode since the last reset.

    Split from the retired ``state_since_reduce(mode="sum")``: the output unit
    is ``same_as:x`` (a sum of ``x`` carries ``x``'s unit), never a fixed
    ``level``.  ``reset_condition`` is a ConditionBool: finite values must be
    exactly 0/1 (anything else raises), NaN = unknown (re-baselines).
    """

    metadata = metadata(
        "state_since_sum",
        "自上次 reset 以来的 episode 内 x 的累计和(单位继承 x)。",
        ["x", "reset_condition", "min_episode"],
        domain="price_volume",
        unit="same_as:x",
    )
    metadata.output_unit = "same_as:x"

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_episode: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(reset_condition, name="reset_condition")
        return _state_since_kernel(x, reset_condition, int(min_episode), "sum")


@register_operator(
    name="state_since_mean",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_since_mean",
    source="stateful.episode",
)
class StateSinceMean(SeriesOperator):
    """Mean of ``x`` over the current episode since the last reset.

    Split from the retired ``state_since_reduce(mode="mean")``: the output unit
    is ``same_as:x`` (a mean of ``x`` carries ``x``'s unit), never a fixed
    ``level``.  ``reset_condition`` is a ConditionBool (see ``state_since_sum``).
    """

    metadata = metadata(
        "state_since_mean",
        "自上次 reset 以来的 episode 内 x 的均值(单位继承 x)。",
        ["x", "reset_condition", "min_episode"],
        domain="price_volume",
        unit="same_as:x",
    )
    metadata.output_unit = "same_as:x"

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_episode: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(reset_condition, name="reset_condition")
        return _state_since_kernel(x, reset_condition, int(min_episode), "mean")


@register_operator(
    name="state_since_count",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_since_count",
    source="stateful.episode",
)
class StateSinceCount(SeriesOperator):
    """Count of valid ``x`` rows over the current episode since the last reset.

    Split from the retired ``state_since_reduce(mode="count")``: the output unit
    is ``count`` (the old metadata's fixed ``level`` was wrong).  ``reset_condition``
    is a ConditionBool (see ``state_since_sum``).
    """

    metadata = metadata(
        "state_since_count",
        "自上次 reset 以来的 episode 内有效 x 的计数。",
        ["x", "reset_condition", "min_episode"],
        domain="price_volume",
        unit="count",
    )
    metadata.output_unit = "count"

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_episode: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(reset_condition, name="reset_condition")
        return _state_since_kernel(x, reset_condition, int(min_episode), "count")


@register_operator(
    name="state_since_last",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_since_last",
    source="stateful.episode",
)
class StateSinceLast(SeriesOperator):
    """Most recent valid ``x`` over the current episode since the last reset.

    Split from the retired ``state_since_reduce(mode="last")``: the output unit
    is ``same_as:x`` (the last value of ``x`` carries ``x``'s unit).  ``reset_condition``
    is a ConditionBool (see ``state_since_sum``).
    """

    metadata = metadata(
        "state_since_last",
        "自上次 reset 以来的 episode 内 x 的最新值(单位继承 x)。",
        ["x", "reset_condition", "min_episode"],
        domain="price_volume",
        unit="same_as:x",
    )
    metadata.output_unit = "same_as:x"

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_episode: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(reset_condition, name="reset_condition")
        return _state_since_kernel(x, reset_condition, int(min_episode), "last")


def _dc_states_and_extents(price: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    rows = price.shape[0]
    state = np.full(rows, np.nan, dtype=float)
    extent = np.full(rows, np.nan, dtype=float)
    direction = 0
    # P0-25: two distinct state variables.  ``origin`` is the price level where
    # the current event *started* (the extent reference, so overshoot
    # accumulates from the event origin); ``extreme`` is the running extreme of
    # the current direction (peak for up, trough for down) and only drives the
    # reversal test.  Reusing one variable for both made a new high reset the
    # extent to ~0 (extent = (p/turning - 1)/threshold with turning = p).
    origin = np.nan
    extreme = np.nan
    first = np.nan
    for row in range(rows):
        p = price[row]
        if not np.isfinite(p):
            state[row] = np.nan
            extent[row] = np.nan
            direction = 0
            origin = np.nan
            extreme = np.nan
            first = np.nan
            continue
        if direction == 0:
            if not np.isfinite(first):
                first = p
                state[row] = 0.0
                extent[row] = 0.0
                continue
            if p >= first * (1.0 + threshold):
                direction = 1
                origin = first
                extreme = p
            elif p <= first * (1.0 - threshold):
                direction = -1
                origin = first
                extreme = p
            state[row] = float(direction)
            if direction == 0:
                extent[row] = 0.0
            else:
                extent[row] = (p / origin - 1.0) / threshold
            continue
        if direction == 1:
            if p > extreme:
                extreme = p
            if p <= extreme * (1.0 - threshold):
                # reversal: a down event starts from the running peak
                direction = -1
                origin = extreme
                extreme = p
            state[row] = float(direction)
            extent[row] = (p / origin - 1.0) / threshold
            continue
        # direction == -1
        if p < extreme:
            extreme = p
        if p >= extreme * (1.0 + threshold):
            # reversal: an up event starts from the running trough
            direction = 1
            origin = extreme
            extreme = p
        state[row] = float(direction)
        extent[row] = (p / origin - 1.0) / threshold
    return state, extent


@register_operator(
    name="directional_change_state",
    category="time_series_state",
    business_category="time_series_state",
    canonical="directional_change_state",
    source="stateful.episode",
)
class DirectionalChangeState(SeriesOperator):
    """Directional-Change event state: +1 up event / -1 down event / 0 none.

    An up event runs while price keeps making new highs and ends when it falls
    ``threshold`` below the running peak (a down event then starts).  NaN price
    emits NaN and re-baselines the DC state machine.
    """

    metadata = metadata(
        "directional_change_state",
        "方向变化事件状态: +1 上行事件 / -1 下行事件 / 0 未确认。",
        ["price", "threshold"],
        domain="price_volume",
        unit="state",
    )

    def _calculate_series(self, price: pd.DataFrame, threshold: float = 0.03, **_: Any) -> pd.DataFrame:
        th = float(threshold)
        if th <= 0.0:
            raise ValueError("threshold must be > 0")
        pv = price.to_numpy(dtype=float)
        rows, cols = pv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            st, _ = _dc_states_and_extents(pv[:, col], th)
            out[:, col] = st
        return frame_like(price, out)


@register_operator(
    name="directional_change_extent",
    category="time_series_state",
    business_category="time_series_state",
    canonical="directional_change_extent",
    source="stateful.episode",
)
class DirectionalChangeExtent(SeriesOperator):
    """Normalized progress of the current Directional-Change event.

    Signed by direction: for an up event ``(price/origin - 1)/threshold``, for
    a down event the same ratio (negative), where ``origin`` is the price level
    at which the event started.  Magnitude = number of ``threshold`` moves since
    the event's origin (1.0 at confirmation, then overshoot accumulates; a new
    running high inside an up event does *not* reset the extent — P0-25).
    """

    metadata = metadata(
        "directional_change_extent",
        "方向变化事件进度: 距转折点多深(按 threshold 归一)。",
        ["price", "threshold"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, price: pd.DataFrame, threshold: float = 0.03, **_: Any) -> pd.DataFrame:
        th = float(threshold)
        if th <= 0.0:
            raise ValueError("threshold must be > 0")
        pv = price.to_numpy(dtype=float)
        rows, cols = pv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, ext = _dc_states_and_extents(pv[:, col], th)
            out[:, col] = ext
        return frame_like(price, out)


@register_operator(
    name="state_since_trend_tstat",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="state_since_trend_tstat",
    source="stateful.episode",
)
class StateSinceTrendTstat(SeriesOperator):
    """OLS slope t-statistic of ``x`` over the current episode since the last
    reset, computed incrementally over the episode.

    The regression time axis is the *physical* bar offset within the episode
    (P1-74): a NaN ``x`` row advances the clock but does not contribute a point,
    so a gap between valid observations is never compressed into an adjacent
    pair.  Requires at least ``min_obs`` valid rows in the episode.  Reset rows
    emit NaN and start a new episode; once the episode exceeds ``max_age`` rows
    it is censored and emits NaN until a real state reset (P1-75) rather than
    silently starting a synthetic fresh episode.
    """

    metadata = metadata(
        "state_since_trend_tstat",
        "episode 内 OLS 斜率 t 统计(事件以来趋势强度, 无量纲)。",
        ["x", "reset_condition", "min_obs", "max_age"],
        domain="price_volume",
        unit="dimensionless",
        category="time_series_regression",
    )
    metadata.output_unit = "dimensionless"

    def _calculate_series(
        self,
        x: pd.DataFrame,
        reset_condition: pd.DataFrame,
        min_obs: int = 5,
        max_age: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(reset_condition, name="reset_condition")
        xv = x.to_numpy(dtype=float)
        rv = reset_condition.to_numpy(dtype=float)
        min_o = max(3, int(min_obs))
        cap = None if max_age is None else int(max_age)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            Ss = 0.0
            Sy = 0.0
            Ssy = 0.0
            Ss2 = 0.0
            Syy = 0.0
            cnt = 0
            age = 0
            censored = False
            for row in range(rows):
                age += 1
                if not np.isfinite(rv[row, col]):
                    out[row, col] = np.nan
                    Ss = Sy = Ssy = Ss2 = Syy = 0.0
                    cnt = 0
                    age = 0
                    censored = False
                    continue
                if bool(rv[row, col] != 0.0):
                    out[row, col] = np.nan
                    Ss = Sy = Ssy = Ss2 = Syy = 0.0
                    cnt = 0
                    age = 0
                    censored = False
                    continue
                if censored:
                    # P1-75: once max_age is exceeded the episode is censored —
                    # stay NaN until a genuine reset, never start a synthetic
                    # fresh episode on the very next bar.
                    out[row, col] = np.nan
                    continue
                if cap is not None and age > cap:
                    out[row, col] = np.nan
                    censored = True
                    continue
                xt = xv[row, col]
                if np.isfinite(xt):
                    # P1-74: physical bar offset within the episode (age), not
                    # the count of valid x rows — missing x rows must advance
                    # the clock so the regression never compresses time.
                    s = float(age)
                    xf = float(xt)
                    Ss += s
                    Sy += xf
                    Ssy += s * xf
                    Ss2 += s * s
                    Syy += xf * xf
                    cnt += 1
                if cnt < min_o:
                    out[row, col] = np.nan
                    continue
                n = float(cnt)
                denom = n * Ss2 - Ss * Ss
                if denom <= 0.0:
                    out[row, col] = np.nan
                    continue
                b = (n * Ssy - Ss * Sy) / denom
                a = (Sy - b * Ss) / n
                sse = Syy - a * Sy - b * Ssy
                if sse < 0.0:
                    sse = 0.0
                mse = np.where((n - 2.0) if n > 2.0 else np.nan != 0, sse / (n - 2.0) if n > 2.0 else np.nan, np.nan)
                if not np.isfinite(mse) or mse <= 0.0:
                    out[row, col] = np.nan
                    continue
                se_b = np.sqrt(mse / (Ss2 - Ss * Ss / n))
                out[row, col] = b / (se_b + _EPS)
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        [
            "state_since_sum", "state_since_mean", "state_since_count",
            "state_since_last",
            "directional_change_state", "directional_change_extent",
            "state_since_trend_tstat",
        ]
    )
    # TASK 2: ``state_since_reduce`` is retired — the mode-enum dimension and
    # the dynamic output unit are gone.  The old name resolves to the sum
    # canonical (its default mode) so existing recipes that used the default
    # keep loading; mode-specific calls must migrate to the split canonicals.
    try:
        OperatorRegistry.register_alias("state_since_reduce", "state_since_sum")
    except (KeyError, ValueError):
        pass  # already registered
    # state_since_reduce is now a deprecated alias, not an active canonical —
    # retract it from the extended surface or finalize_layer_governance reports
    # an inactive static-surface entry.
    try:
        import cleaned_operators.operator_surface as _surface

        _surface.retract_extended_only({"state_since_reduce"})
    except ImportError:  # pragma: no cover - surface always present in-tree
        pass


_register_surface()


# Round-11 #12: the split episode-reduce canonicals are forward per-column
# recursions — a segmented checkpoint restore would restart the accumulator
# mid-episode.  Declare recursive / required_full_history (honest fail-closed).
from runtime.execution_contract import declare_stateful  # noqa: E402

for _episode_canon in (
    "state_since_sum",
    "state_since_mean",
    "state_since_count",
    "state_since_last",
):
    declare_stateful(_episode_canon, state_model="recursive", chunking="required_full_history")
