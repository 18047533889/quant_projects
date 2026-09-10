# -*- coding: utf-8 -*-
"""Single-source-of-truth recursive kernels for checkpoint stateful operators.

Full-history and checkpoint-incremental execution must be the *same* math.  The
full history path is the pandas reference — ``Series.ewm(alpha, adjust=False,
ignore_na=False)`` for EMA / Wilder / RSI / ATR / ADX / MACD — so every segment
kernel below reproduces that recurrence exactly, including its NaN handling:

* ``ignore_na=False`` weights are based on **absolute position**: a NaN decays
  the running ``old_wt`` by ``(1 - alpha)`` but leaves ``weighted_avg`` intact,
  and the output at a NaN position is the carried ``weighted_avg`` (not NaN);
* a valid observation updates
  ``weighted_avg = (old_wt*(1-alpha)*weighted_avg + alpha*x) / (old_wt*(1-alpha) + alpha)``
  and then **resets ``old_wt`` to 1.0** (that reset is what makes ``adjust=False``
  the familiar ``(1-alpha)*prev + alpha*x`` recursion when no NaN has intervened);
* ``min_periods`` counts **non-NaN observations**; output stays NaN until the
  count reaches the gate.

Because every path (full-history, checkpoint bootstrap, checkpoint resume) calls
these same loops, ``full_history(N) == stitch(segments, N)`` holds by
construction.  ``stateful_runtime._*_segment`` delegates here; the pandas
``Series.ewm`` reference remains the evidence-golden source and is reproduced
numerically (verified in tests/operators/test_stateful_segment_runtime.py).

Each kernel returns ``(values: np.ndarray, state: dict)`` where ``state`` is a
plain dict that survives JSON checkpoint serialization (no NaN / Inf payloads —
unseeded states are stored as ``None``, counts as ints).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EwmState:
    """Internal state of a pandas-equivalent ``ewm(adjust=False, ignore_na=False)``
    recurrence: the running weighted average, the decaying weight, and the number
    of non-NaN observations seen (drives ``min_periods`` gating)."""

    weighted_avg: float | None = None
    old_wt: float = 1.0
    valid_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weighted_avg": None if self.weighted_avg is None else float(self.weighted_avg),
            "old_wt": float(self.old_wt),
            "valid_count": int(self.valid_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EwmState":
        raw = payload.get("weighted_avg")
        # Only NaN is a missing state; ``Inf`` is a real weighted-average value
        # and must survive the in-memory round-trip (JSON persistence of Inf is
        # handled by the checkpoint store, which maps it to null — an Inf state
        # is pathological data and resumes then fail closed).
        weighted = float(raw) if raw is not None and not np.isnan(float(raw)) else None
        old_wt = float(payload.get("old_wt", 1.0))
        if np.isnan(old_wt):
            old_wt = 1.0
        return cls(
            weighted_avg=weighted,
            old_wt=old_wt,
            valid_count=max(0, int(payload.get("valid_count", 0) or 0)),
        )


def _ewm_step(state: EwmState, value: float | None, alpha: float) -> float | None:
    """One position of the pandas ``ewm(adjust=False, ignore_na=False)``
    recurrence.  Returns the output value at this position (the carried
    ``weighted_avg`` when the observation is missing, ``None`` when nothing has
    been seeded yet).  pandas treats *both* NaN and Inf as non-observations in
    the EWM (verified: ``Series([1,2,inf,4]).ewm(...).mean()`` carries the 2
    through the Inf instead of mixing it in), so ``np.isfinite`` is the gate."""
    if value is not None and np.isfinite(value):
        if state.weighted_avg is None:
            state.weighted_avg = float(value)
        else:
            old = state.old_wt * (1.0 - alpha)
            state.weighted_avg = (old * state.weighted_avg + alpha * float(value)) / (
                old + alpha
            )
        state.old_wt = 1.0
        state.valid_count += 1
        return state.weighted_avg
    state.old_wt *= 1.0 - alpha
    return state.weighted_avg


def ema_segment(x: np.ndarray, state: dict[str, Any], span: int) -> tuple[np.ndarray, dict[str, Any]]:
    """``ts_ema``: ``x.ewm(span=span, adjust=False).mean()`` with output carried
    across missing bars (matches pandas ``ignore_na=False`` default)."""
    alpha = 2.0 / (span + 1.0)
    ewm = EwmState.from_dict(state.get("ema", {}))
    out = np.full(x.shape, np.nan, dtype=float)
    for i, value in enumerate(x):
        carried = _ewm_step(ewm, value, alpha)
        if carried is not None:
            out[i] = carried
    state["ema"] = ewm.to_dict()
    return out, state


def _rsi(avg_gain: float | None, avg_loss: float | None) -> float:
    if avg_gain is None or avg_loss is None:
        return np.nan
    if avg_loss <= 0.0:
        return 50.0 if avg_gain <= 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi_wilder_segment(
    close: np.ndarray, state: dict[str, Any], window: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """``RSI_WILDER``: Wilder RSI over ``close`` matching pandas
    ``gain.ewm(alpha=1/w, adjust=False, min_periods=w).mean()`` — the seed is the
    *recursive* EWM value at the ``w``-th observation, NOT the SMA of the first
    ``w`` gains (review #5 R5-10).  ``close.diff()[0]`` is NaN, so the first
    position never seeds gain/loss."""
    alpha = 1.0 / window
    last_close = state.get("last_close")
    last_close = float(last_close) if last_close is not None and not np.isnan(last_close) else None
    gain_state = EwmState.from_dict(state.get("gain", {}))
    loss_state = EwmState.from_dict(state.get("loss", {}))
    out = np.full(close.shape, np.nan, dtype=float)
    for i, value in enumerate(close):
        if not np.isnan(value):
            value = float(value)
            # ``close.diff()`` uses the immediately-preceding row INCLUDING a
            # missing one: a NaN close at t-1 poisons delta[t] (R5-09).  So a
            # NaN previous close yields delta=None just like the first row.
            # An ``Inf`` close is a real observation (diff/tr become Inf).
            if last_close is None:
                delta: float | None = None  # diff[0] == NaN
            else:
                delta = value - last_close
            last_close = value
        else:
            delta = None
            last_close = np.nan  # previous close is *missing*, not skipped
        if delta is None:
            gain: float | None = None
            loss: float | None = None
        else:
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
        gavg = _ewm_step(gain_state, gain, alpha)
        lavg = _ewm_step(loss_state, loss, alpha)
        if (
            gain_state.valid_count >= window
            and loss_state.valid_count >= window
            and gavg is not None
            and lavg is not None
        ):
            out[i] = _rsi(gavg, lavg)
    # A trailing NaN previous-close is stored as nan (JSON null); on resume both
    # None and NaN mean "delta/TR unknown for the next bar", which is identical.
    state["last_close"] = np.nan if last_close is None else last_close
    state["gain"] = gain_state.to_dict()
    state["loss"] = loss_state.to_dict()
    return out, state


def _true_range(h: float, l: float, previous_close: float | None) -> float | None:
    if previous_close is None or np.isnan(previous_close):
        # pandas: no (or missing) prev_close -> tr1/tr2/tr3 = NaN via
        # ``(high - close.shift(1)).abs()`` and ``np.maximum`` NaN propagation.
        # Python's builtin ``max`` does NOT propagate NaN (``max(1, nan) == 1``),
        # so we must return None explicitly instead of relying on ``max``.
        # ``Inf`` previous_close is a real value (TR becomes Inf).
        return None
    return max(h - l, abs(h - previous_close), abs(l - previous_close))


def atr_wilder_segment(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    state: dict[str, Any],
    window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``ATR_WILDER``: pandas ``tr.ewm(alpha=1/w, adjust=False, min_periods=w)``
    where the first true-range is NaN (review #5 R5-11)."""
    alpha = 1.0 / window
    last_close = state.get("last_close")
    last_close = float(last_close) if last_close is not None and not np.isnan(last_close) else None
    tr_state = EwmState.from_dict(state.get("tr", {}))
    out = np.full(close.shape, np.nan, dtype=float)
    for i, (h, l, c) in enumerate(zip(high, low, close)):
        if np.isnan(h) or np.isnan(l) or np.isnan(c):
            tr: float | None = None
            last_close = np.nan  # missing bar -> next TR's prev_close unknown
        else:
            h, l, c = float(h), float(l), float(c)
            tr = _true_range(h, l, last_close)
            last_close = c
        carried = _ewm_step(tr_state, tr, alpha)
        if carried is not None and tr_state.valid_count >= window:
            out[i] = carried
    state["last_close"] = np.nan if last_close is None else last_close
    state["tr"] = tr_state.to_dict()
    return out, state


def adx_segment(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    state: dict[str, Any], window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Actual ADX winner: every EWM stage requires window valid observations.

    Physical high/low/close shifts are independent; a missing close must not
    erase a valid high/low movement. Zero directional support is undefined.
    """
    alpha = 1.0 / window
    last_h = state.get("last_high", np.nan)
    last_l = state.get("last_low", np.nan)
    last_c = state.get("last_close", np.nan)
    last_h = np.nan if last_h is None else float(last_h)
    last_l = np.nan if last_l is None else float(last_l)
    last_c = np.nan if last_c is None else float(last_c)
    tr_state = EwmState.from_dict(state.get("tr", {}))
    plus_state = EwmState.from_dict(state.get("plus_dm", {}))
    minus_state = EwmState.from_dict(state.get("minus_dm", {}))
    dx_state = EwmState.from_dict(state.get("dx", {}))
    out = np.full(close.shape, np.nan, dtype=float)
    for i, (h, l, c) in enumerate(zip(high, low, close)):
        h, l, c = float(h), float(l), float(c)
        with np.errstate(invalid="ignore", over="ignore"):
            tr = float(np.maximum.reduce([h - l, abs(h - last_c), abs(l - last_c)]))
            up, down = h - last_h, last_l - l
        plus_raw = up if up > down and up > 0 else 0.0
        minus_raw = down if down > up and down > 0 else 0.0
        last_h, last_l, last_c = h, l, c
        atr = _ewm_step(tr_state, tr, alpha)
        plus_sm = _ewm_step(plus_state, plus_raw, alpha)
        minus_sm = _ewm_step(minus_state, minus_raw, alpha)
        mature = min(tr_state.valid_count, plus_state.valid_count, minus_state.valid_count) >= window
        dx = None
        if mature and atr is not None and atr != 0 and plus_sm is not None and minus_sm is not None:
            plus_di, minus_di = 100.0 * plus_sm / atr, 100.0 * minus_sm / atr
            denom = plus_di + minus_di
            if denom != 0:
                dx = 100.0 * abs(plus_di - minus_di) / denom
        adx = _ewm_step(dx_state, dx, alpha)
        if adx is not None and dx_state.valid_count >= window:
            out[i] = adx
    state.update(last_high=last_h, last_low=last_l, last_close=last_c,
                 tr=tr_state.to_dict(), plus_dm=plus_state.to_dict(),
                 minus_dm=minus_state.to_dict(), dx=dx_state.to_dict())
    return out, state


def macd_segment(
    x: np.ndarray,
    state: dict[str, Any],
    fast: int,
    slow: int,
    signal: int,
    output: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``MACD_line`` / ``MACD_signal`` / ``MACD_hist``: three span-EWM recursions
    with ``min_periods=1``, matching ``x.ewm(span=fast, adjust=False).mean()``."""
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    alpha_f = 2.0 / (fast + 1.0)
    alpha_s = 2.0 / (slow + 1.0)
    alpha_i = 2.0 / (signal + 1.0)
    fast_state = EwmState.from_dict(state.get("fast_ema", {}))
    slow_state = EwmState.from_dict(state.get("slow_ema", {}))
    signal_state = EwmState.from_dict(state.get("signal_ema", {}))
    out = np.full(x.shape, np.nan, dtype=float)
    for i, value in enumerate(x):
        fast_v = _ewm_step(fast_state, value, alpha_f)
        slow_v = _ewm_step(slow_state, value, alpha_s)
        if fast_v is None or slow_v is None:
            line: float | None = None
        else:
            line = fast_v - slow_v
        sig_v = _ewm_step(signal_state, line, alpha_i)
        if sig_v is None or line is None:
            continue
        if output == "line":
            out[i] = line
        elif output == "signal":
            out[i] = sig_v
        else:
            out[i] = line - sig_v
    state["fast_ema"] = fast_state.to_dict()
    state["slow_ema"] = slow_state.to_dict()
    state["signal_ema"] = signal_state.to_dict()
    return out, state


__all__ = [
    "EwmState",
    "ema_segment",
    "rsi_wilder_segment",
    "atr_wilder_segment",
    "adx_segment",
    "macd_segment",
]
