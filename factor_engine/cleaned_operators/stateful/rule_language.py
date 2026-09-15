# -*- coding: utf-8 -*-
"""Stateful rule-control primitives (2026-08 CTA pack).

* ``state_latch``      — set/reset SR latch (reset priority), regime memory.
* ``state_hold``       — snapshot-and-remember: keep the last value seen on an
                         update condition until reset (recursive memory, unlike
                         the rolling-window ``ts_last_if``).
* ``state_slew_limit`` — rate limiter: output moves at most ``limit`` per row
                         toward the target (slow-adjustment signal control).
* ``state_deadband``   — continuous hysteresis: changes smaller than ``band``
                         are ignored; larger changes move by the excess only.

All four are forward per-column recursions (prefix-causal, PIT).  A NaN input
emits NaN and re-baselines the internal state (``missing_policy="break"``): a
missing observation is never treated as False / no-change.
"""
from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
    validate_operator_call,
)
from factor_engine.cleaned_operators.common.daily_panel import _aligned
from factor_engine.cleaned_operators.rolling_pack import frame_like
from factor_engine.cleaned_operators.stateful._common import (
    assert_condition_bool,
    metadata,
    panel_or_scalar,
)

_EPS = 1e-12


def _nonnegative_scalar_or_panel(value: Any, name: str) -> None:
    """Validate scalar controls without rejecting supported panel controls."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real or a panel")
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _state_metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    panel_params: tuple[str, ...],
    param_specs: dict[str, ParamSpec],
    unit: str,
) -> OperatorMetadata:
    """Build the complete call topology for a recursive state operator."""
    result = metadata(
        name, description, params, domain="trading_state", unit=unit,
    )
    result.param_specs = param_specs
    result.panel_params = panel_params
    result.panel_arity = len(panel_params)
    result.scalar_params = tuple(name for name in params if name not in panel_params)
    if unit == "state":
        result.output_unit = "dimensionless"
    else:
        result.output_unit = "same_as:value" if name == "state_hold" else "same_as:x"
    result.window_semantics = "causal_unbounded"
    return result


@register_operator(
    name="state_latch",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_latch",
    source="stateful.rule_language",
)
class StateLatch(SeriesOperator):
    """Set/reset SR latch: ``reset`` wins, then ``set``, else carry.

    ``initial_state`` (0 or 1) is the neutral value the latch returns before
    any event and the value a NaN input re-baselines to.
    """

    metadata = _state_metadata(
        "state_latch",
        "SR 锁存状态: reset 优先, 其次 set, 否则保持。",
        ["set_condition", "reset_condition", "initial_state"],
        panel_params=("set_condition", "reset_condition"),
        param_specs={
            "initial_state": ParamSpec(
                dtype=float, choices=(0.0, 1.0), default=0.0, searchable=False,
                param_role=ParamRole.POLICY,
            ),
        },
        unit="state",
    )

    def _calculate_series(
        self,
        set_condition: pd.DataFrame,
        reset_condition: pd.DataFrame,
        initial_state: float = 0.0,
        **_: Any,
    ) -> pd.DataFrame:
        set_condition, reset_condition = _aligned(set_condition, reset_condition)
        assert_condition_bool(set_condition, name="set_condition")
        assert_condition_bool(reset_condition, name="reset_condition")
        sc = set_condition.to_numpy(dtype=float)
        rc = reset_condition.to_numpy(dtype=float)
        # P1-28: ``initial_state`` is strictly a boolean state — 2 or -1 must
        # not silently coerce to active.  Reject anything outside {0, 1}.
        init_v = float(initial_state)
        if init_v not in (0.0, 1.0):
            raise ValueError("state_latch requires initial_state in {0, 1}")
        init = 1.0 if init_v == 1.0 else 0.0
        rows, cols = sc.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            state = init
            for row in range(rows):
                if not (np.isfinite(sc[row, col]) and np.isfinite(rc[row, col])):
                    # NaN condition is unknown: it does not flip the latch and
                    # re-baselines to the initial state.
                    out[row, col] = np.nan
                    state = init
                    continue
                if rc[row, col] == 1.0:
                    state = 0.0
                elif sc[row, col] == 1.0:
                    state = 1.0
                out[row, col] = state
        return frame_like(set_condition, out)


@register_operator(
    name="state_hold",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_hold",
    source="stateful.rule_language",
)
class StateHold(SeriesOperator):
    """Recursive snapshot memory.

    ``Y_t = x_t`` when ``update_condition`` holds and ``x_t`` is finite;
    ``Y_t = NaN`` on ``reset_condition`` (or when not yet updated); otherwise
    ``Y_t = Y_{t-1}`` (the last snapshot is remembered).
    """

    metadata = _state_metadata(
        "state_hold",
        "条件触发时记录 x, 之后一直保持最近一次记录值, reset 清空。",
        ["value", "update_condition", "reset_condition"],
        panel_params=("value", "update_condition", "reset_condition"),
        param_specs={
            "reset_condition": ParamSpec(
                default=None, searchable=False, param_role=ParamRole.POLICY,
            ),
        },
        unit="level",
    )

    def _calculate_series(
        self,
        value: pd.DataFrame,
        update_condition: pd.DataFrame,
        reset_condition: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        if reset_condition is not None:
            value, update_condition, reset_condition = _aligned(value, update_condition, reset_condition)
        else:
            value, update_condition = _aligned(value, update_condition)
        assert_condition_bool(update_condition, name="update_condition")
        if reset_condition is not None:
            assert_condition_bool(reset_condition, name="reset_condition")
        vv = value.to_numpy(dtype=float)
        uc = update_condition.to_numpy(dtype=float)
        rv = None if reset_condition is None else reset_condition.to_numpy(dtype=float)
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            held = np.nan
            for row in range(rows):
                u_ok = bool(np.isfinite(uc[row, col]))
                r_ok = True if rv is None else bool(np.isfinite(rv[row, col]))
                if not (u_ok and r_ok):
                    # NaN condition is unknown: it must not flip the hold and
                    # re-baselines the memory.
                    out[row, col] = np.nan
                    held = np.nan
                    continue
                if rv is not None and rv[row, col] == 1.0:
                    held = np.nan
                elif uc[row, col] == 1.0:
                    # P1-29: an explicit update whose payload is NaN means "we
                    # cannot confirm the new value" — it must clear the hold,
                    # not silently keep the previous snapshot.
                    held = float(vv[row, col]) if np.isfinite(vv[row, col]) else np.nan
                out[row, col] = held
        return frame_like(value, out)


@register_operator(
    name="state_slew_limit",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_slew_limit",
    source="stateful.rule_language",
)
class StateSlewLimit(SeriesOperator):
    """Rate limiter: output moves at most ``limit`` per row toward target.

    ``Y_t = Y_{t-1} + clip(X_t - Y_{t-1}, -L_t, L_t)``.  ``limit`` may be a
    scalar or a per-row panel.  The first finite observation initialises the
    output directly.
    """

    metadata = _state_metadata(
        "state_slew_limit",
        "输出每行最多向目标移动 limit, 实现慢调整信号控制。",
        ["x", "limit"],
        panel_params=("x", "limit"),
        param_specs={
            "limit": ParamSpec(
                dtype=float,
                min=0.0,
                default=0.01,
                param_role=ParamRole.STATE_THRESHOLD,
            ),
        },
        unit="level",
    )

    _HANDLES_CALL_CONTRACT = True

    def calculate(self, x: pd.DataFrame, limit: Any = 0.01, **kwargs: Any) -> pd.DataFrame:
        # Backward-compatible dynamic threshold panel.  The formal factor
        # parameter remains the scalar ``limit``; a labelled panel is a data
        # input and is validated/aligned cell-wise by the kernel.
        if isinstance(limit, (pd.DataFrame, pd.Series)):
            processed, processed_kwargs = validate_operator_call(
                self, (x,), {"limit": 0.01, **kwargs},
            )
            x = processed[0]
            return self._calculate_series(x, limit=limit, **kwargs)
        processed, processed_kwargs = validate_operator_call(
            self, (x,), {"limit": limit, **kwargs},
        )
        return self._calculate_series(*processed, **processed_kwargs)

    def _calculate_series(self, x: pd.DataFrame, limit: Any = 0.01, **_: Any) -> pd.DataFrame:
        _nonnegative_scalar_or_panel(limit, "limit")
        if isinstance(limit, pd.DataFrame):
            x, limit = _aligned(x, limit)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev = np.nan
            for row in range(rows):
                target = xv[row, col]
                lim = panel_or_scalar(limit, row, col)
                if not np.isfinite(target) or not np.isfinite(lim) or lim < 0.0:
                    out[row, col] = np.nan
                    prev = np.nan
                    continue
                if not np.isfinite(prev):
                    prev = target
                    out[row, col] = prev
                    continue
                delta = float(target) - prev
                prev = prev + float(np.clip(delta, -lim, lim))
                out[row, col] = prev
        return frame_like(x, out)


@register_operator(
    name="state_deadband",
    category="time_series_state",
    business_category="time_series_state",
    canonical="state_deadband",
    source="stateful.rule_language",
)
class StateDeadband(SeriesOperator):
    """Continuous hysteresis: ignore small changes, move by the excess only.

    With ``d = X_t - Y_{t-1}``: ``Y_t = Y_{t-1}`` when ``|d| <= band``, else
    ``Y_t = Y_{t-1} + sign(d) * (|d| - band)``.  ``band`` may be a scalar or a
    per-row panel.  The first finite observation initialises the output.
    """

    metadata = _state_metadata(
        "state_deadband",
        "连续迟滞: 变化小于 band 不动, 超过部分才移动。",
        ["x", "band"],
        panel_params=("x", "band"),
        param_specs={
            "band": ParamSpec(
                dtype=float,
                min=0.0,
                default=0.0,
                param_role=ParamRole.STATE_THRESHOLD,
            ),
        },
        unit="level",
    )

    _HANDLES_CALL_CONTRACT = True

    def calculate(self, x: pd.DataFrame, band: Any = 0.0, **kwargs: Any) -> pd.DataFrame:
        # See StateSlewLimit.calculate: panel thresholds are supported data
        # inputs, while scalar ``band`` is the declared search parameter.
        if isinstance(band, (pd.DataFrame, pd.Series)):
            processed, processed_kwargs = validate_operator_call(
                self, (x,), {"band": 0.0, **kwargs},
            )
            x = processed[0]
            return self._calculate_series(x, band=band, **kwargs)
        processed, processed_kwargs = validate_operator_call(
            self, (x,), {"band": band, **kwargs},
        )
        return self._calculate_series(*processed, **processed_kwargs)

    def _calculate_series(self, x: pd.DataFrame, band: Any = 0.0, **_: Any) -> pd.DataFrame:
        _nonnegative_scalar_or_panel(band, "band")
        if isinstance(band, pd.DataFrame):
            x, band = _aligned(x, band)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev = np.nan
            for row in range(rows):
                target = xv[row, col]
                b = panel_or_scalar(band, row, col)
                if not np.isfinite(target) or not np.isfinite(b) or b < 0.0:
                    out[row, col] = np.nan
                    prev = np.nan
                    continue
                if not np.isfinite(prev):
                    prev = target
                    out[row, col] = prev
                    continue
                d = float(target) - prev
                if abs(d) <= b:
                    out[row, col] = prev
                    continue
                prev = prev + float(np.sign(d) * (abs(d) - b))
                out[row, col] = prev
        return frame_like(x, out)


def _register_surface() -> None:
    from factor_engine.cleaned_operators.stateful._common import register_stateful_surface
    from factor_engine.cleaned_operators.rolling_pack import register_polars_udf

    canonicals = ["state_latch", "state_hold", "state_slew_limit", "state_deadband"]
    register_stateful_surface(canonicals)
    for canonical in canonicals:
        register_polars_udf(canonical)


_register_surface()


# These recurrences have no checkpoint adapter.  Starting them at an arbitrary
# shard boundary changes their output, so incremental execution must replay the
# complete instrument history.
from factor_engine.runtime.execution_contract import declare_stateful  # noqa: E402

for _canonical in ("state_latch", "state_hold", "state_slew_limit", "state_deadband"):
    declare_stateful(
        _canonical,
        state_model="recursive",
        chunking="required_full_history",
        history_kind="full_history",
    )
