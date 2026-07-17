# -*- coding: utf-8 -*-
"""Checkpoint-backed incremental execution for recursive factor operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stateful_contract import StateCheckpoint, StatefulCheckpointRegistry


@dataclass(frozen=True)
class StatefulSegmentResult:
    values: np.ndarray
    checkpoint: StateCheckpoint
    state: Mapping[str, Any]


def _array(values: Sequence[Any], name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return out


def _timestamps(values: Sequence[Any], expected: int) -> pd.DatetimeIndex:
    ts = pd.to_datetime(list(values), errors="raise", utc=True)
    if len(ts) != expected or expected < 1:
        raise ValueError("timestamps must be non-empty and match input length")
    if not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError("timestamps must be unique and monotonic increasing")
    return ts


def _positive(value: Any, name: str) -> int:
    out = int(value)
    if out < 1 or float(value) != float(out):
        raise ValueError(f"{name} must be a positive integer")
    return out


def _ema_step(value: float, previous: float | None, alpha: float) -> float:
    return value if previous is None or not np.isfinite(previous) else alpha * value + (1.0 - alpha) * previous


def _effective_identity(canonical, input_identity, inputs, params, spec):
    identity = dict(input_identity)
    identity["__stateful_runtime__"] = {
        "canonical": canonical,
        "semantic_version": spec.semantic_version,
        "state_schema_version": spec.state_schema_version,
        "missing_policy": spec.missing_policy,
        "input_names": sorted(str(name) for name in inputs),
        "params": {str(key): params[key] for key in sorted(params)},
    }
    return identity


def _ema_segment(x, state, span):
    alpha = 2.0 / (span + 1.0)
    last = state.get("last_ema")
    last = float(last) if last is not None and np.isfinite(last) else None
    out = np.full(x.shape, np.nan)
    for i, value in enumerate(x):
        if np.isfinite(value):
            last = _ema_step(float(value), last, alpha)
            out[i] = last
    state["last_ema"] = np.nan if last is None else last
    return out, state


def _rsi(avg_gain, avg_loss):
    if avg_loss <= 0:
        return 50.0 if avg_gain <= 0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _rsi_segment(close, state, window):
    out = np.full(close.shape, np.nan)
    last = state.get("last_close")
    last = float(last) if last is not None and np.isfinite(last) else None
    gain_avg, loss_avg = state.get("avg_gain"), state.get("avg_loss")
    seeded = all(v is not None and np.isfinite(v) for v in (gain_avg, loss_avg))
    gains, losses = list(state.get("_seed_gains", [])), list(state.get("_seed_losses", []))
    for i, value in enumerate(close):
        if not np.isfinite(value):
            continue
        value = float(value)
        if last is None:
            last = value
            continue
        delta, last = value - last, value
        gain, loss = max(delta, 0.0), max(-delta, 0.0)
        if not seeded:
            gains.append(gain); losses.append(loss)
            if len(gains) < window:
                continue
            gain_avg, loss_avg = float(np.mean(gains[-window:])), float(np.mean(losses[-window:]))
            seeded = True
        else:
            gain_avg = (float(gain_avg) * (window - 1) + gain) / window
            loss_avg = (float(loss_avg) * (window - 1) + loss) / window
        out[i] = _rsi(float(gain_avg), float(loss_avg))
    state.update({"last_close": np.nan if last is None else last, "avg_gain": np.nan if gain_avg is None else float(gain_avg), "avg_loss": np.nan if loss_avg is None else float(loss_avg), "_seed_gains": [] if seeded else gains[-window:], "_seed_losses": [] if seeded else losses[-window:]})
    return out, state


def _tr(high, low, previous_close):
    return high - low if previous_close is None or not np.isfinite(previous_close) else max(high - low, abs(high - previous_close), abs(low - previous_close))


def _atr_segment(high, low, close, state, window):
    out = np.full(close.shape, np.nan)
    atr = state.get("atr")
    seeded = atr is not None and np.isfinite(atr)
    atr = float(atr) if seeded else None
    seed = list(state.get("_seed_tr", []))
    last = state.get("last_close")
    last = float(last) if last is not None and np.isfinite(last) else None
    for i, (h, l, c) in enumerate(zip(high, low, close)):
        if not (np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            continue
        tr = _tr(float(h), float(l), last); last = float(c)
        if not seeded:
            seed.append(tr)
            if len(seed) < window:
                continue
            atr, seeded = float(np.mean(seed[-window:])), True
        else:
            atr = (float(atr) * (window - 1) + tr) / window
        out[i] = float(atr)
    state.update({"atr": np.nan if atr is None else atr, "last_close": np.nan if last is None else last, "_seed_tr": [] if seeded else seed[-window:]})
    return out, state


def _adx_segment(high, low, close, state, window):
    out = np.full(close.shape, np.nan)
    atr, plus_sm, minus_sm, adx = (state.get(k) for k in ("atr", "plus_dm", "minus_dm", "adx"))
    dm_seeded = all(v is not None and np.isfinite(v) for v in (atr, plus_sm, minus_sm))
    adx_seeded = adx is not None and np.isfinite(adx)
    atr = float(atr) if dm_seeded else None; plus_sm = float(plus_sm) if dm_seeded else None; minus_sm = float(minus_sm) if dm_seeded else None; adx = float(adx) if adx_seeded else None
    seed_tr, seed_plus, seed_minus, seed_dx = (list(state.get(k, [])) for k in ("_seed_tr", "_seed_plus_dm", "_seed_minus_dm", "_seed_dx"))
    last_h, last_l, last_c = (state.get(k) for k in ("last_high", "last_low", "last_close"))
    last_h = float(last_h) if last_h is not None and np.isfinite(last_h) else None; last_l = float(last_l) if last_l is not None and np.isfinite(last_l) else None; last_c = float(last_c) if last_c is not None and np.isfinite(last_c) else None
    for i, (h, l, c) in enumerate(zip(high, low, close)):
        if not (np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            continue
        h, l, c = float(h), float(l), float(c); tr = _tr(h, l, last_c)
        if last_h is None or last_l is None:
            plus_raw = minus_raw = 0.0
        else:
            up, down = h - last_h, last_l - l
            plus_raw = up if up > down and up > 0 else 0.0
            minus_raw = down if down > up and down > 0 else 0.0
        last_h, last_l, last_c = h, l, c
        if not dm_seeded:
            seed_tr.append(tr); seed_plus.append(plus_raw); seed_minus.append(minus_raw)
            if len(seed_tr) < window:
                continue
            atr, plus_sm, minus_sm = float(np.mean(seed_tr[-window:])), float(np.mean(seed_plus[-window:])), float(np.mean(seed_minus[-window:])); dm_seeded = True
        else:
            atr = (float(atr) * (window - 1) + tr) / window
            plus_sm = (float(plus_sm) * (window - 1) + plus_raw) / window
            minus_sm = (float(minus_sm) * (window - 1) + minus_raw) / window
        plus_di = 100.0 * float(plus_sm) / float(atr) if float(atr) > 0 else 0.0
        minus_di = 100.0 * float(minus_sm) / float(atr) if float(atr) > 0 else 0.0
        denom = plus_di + minus_di; dx = 0.0 if denom <= 0 else 100.0 * abs(plus_di - minus_di) / denom
        if not adx_seeded:
            seed_dx.append(dx)
            if len(seed_dx) < window:
                continue
            adx, adx_seeded = float(np.mean(seed_dx[-window:])), True
        else:
            adx = (float(adx) * (window - 1) + dx) / window
        out[i] = float(adx)
    state.update({"atr": np.nan if atr is None else atr, "plus_dm": np.nan if plus_sm is None else plus_sm, "minus_dm": np.nan if minus_sm is None else minus_sm, "adx": np.nan if adx is None else adx, "last_high": np.nan if last_h is None else last_h, "last_low": np.nan if last_l is None else last_l, "last_close": np.nan if last_c is None else last_c, "_seed_tr": [] if dm_seeded else seed_tr[-window:], "_seed_plus_dm": [] if dm_seeded else seed_plus[-window:], "_seed_minus_dm": [] if dm_seeded else seed_minus[-window:], "_seed_dx": [] if adx_seeded else seed_dx[-window:]})
    return out, state


def _macd_segment(x, state, fast, slow, signal, output):
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    af, asl, asi = 2.0 / (fast + 1), 2.0 / (slow + 1), 2.0 / (signal + 1)
    fast_ema, slow_ema, signal_ema = (state.get(k) for k in ("fast_ema", "slow_ema", "signal_ema"))
    fast_ema = float(fast_ema) if fast_ema is not None and np.isfinite(fast_ema) else None; slow_ema = float(slow_ema) if slow_ema is not None and np.isfinite(slow_ema) else None; signal_ema = float(signal_ema) if signal_ema is not None and np.isfinite(signal_ema) else None
    out = np.full(x.shape, np.nan)
    for i, value in enumerate(x):
        if not np.isfinite(value):
            continue
        value = float(value); fast_ema = _ema_step(value, fast_ema, af); slow_ema = _ema_step(value, slow_ema, asl)
        line = fast_ema - slow_ema; signal_ema = _ema_step(line, signal_ema, asi)
        out[i] = line if output == "line" else signal_ema if output == "signal" else line - signal_ema
    state.update({"fast_ema": np.nan if fast_ema is None else fast_ema, "slow_ema": np.nan if slow_ema is None else slow_ema, "signal_ema": np.nan if signal_ema is None else signal_ema})
    return out, state


def execute_stateful_segment(canonical: str, inputs: Mapping[str, Sequence[Any]], *, timestamps: Sequence[Any], instrument: str, input_identity: Mapping[str, Any], params: Mapping[str, Any] | None = None, checkpoint: StateCheckpoint | None = None, starts_at_dataset_origin: bool = False) -> StatefulSegmentResult:
    spec = StatefulCheckpointRegistry.get(canonical)
    if spec is None or not inputs:
        raise ValueError(f"unsupported or empty stateful request: {canonical}")
    n = len(next(iter(inputs.values())))
    if any(len(v) != n for v in inputs.values()):
        raise ValueError("all input arrays must have equal length")
    ts = _timestamps(timestamps, n)
    options = dict(params or {})
    effective_identity = _effective_identity(canonical, input_identity, inputs, options, spec)
    StatefulCheckpointRegistry.require_for_segment(canonical, starts_at_dataset_origin=bool(starts_at_dataset_origin), checkpoint=checkpoint, input_identity=effective_identity if checkpoint else None, expected_instrument=str(instrument) if checkpoint else None)
    if checkpoint is not None and pd.Timestamp(ts[0]) <= pd.Timestamp(checkpoint.as_of):
        raise ValueError("segment must start strictly after checkpoint.as_of")
    state = dict(checkpoint.state) if checkpoint else {}
    if canonical == "ts_ema":
        values, state = _ema_segment(_array(inputs["x"], "x"), state, _positive(options.get("span", options.get("window", 20)), "span"))
    elif canonical == "RSI_WILDER":
        values, state = _rsi_segment(_array(inputs["x"], "x"), state, _positive(options.get("window", 14), "window"))
    elif canonical == "ATR_WILDER":
        values, state = _atr_segment(_array(inputs["high"], "high"), _array(inputs["low"], "low"), _array(inputs["close"], "close"), state, _positive(options.get("window", 14), "window"))
    elif canonical == "ADX":
        values, state = _adx_segment(_array(inputs["high"], "high"), _array(inputs["low"], "low"), _array(inputs["close"], "close"), state, _positive(options.get("window", 14), "window"))
    elif canonical in {"MACD_line", "MACD_signal", "MACD_hist"}:
        values, state = _macd_segment(_array(inputs["x"], "x"), state, _positive(options.get("fast", 12), "fast"), _positive(options.get("slow", 26), "slow"), _positive(options.get("signal", 9), "signal"), {"MACD_line": "line", "MACD_signal": "signal", "MACD_hist": "hist"}[canonical])
    else:
        raise ValueError(f"stateful runtime not implemented for {canonical}")
    state["last_timestamp"] = ts[-1].isoformat()
    new_checkpoint = StatefulCheckpointRegistry.create_checkpoint(canonical, instrument=str(instrument), as_of=state["last_timestamp"], state=state, input_identity=effective_identity)
    return StatefulSegmentResult(values=values, checkpoint=new_checkpoint, state=state)


__all__ = ["StatefulSegmentResult", "execute_stateful_segment"]
