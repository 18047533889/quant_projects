# -*- coding: utf-8 -*-
"""Checkpoint-backed incremental execution for recursive factor operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stateful_contract import StateCheckpoint, StatefulCheckpointRegistry

# R5-09/10/11: the checkpoint incremental kernels ARE the full-history math now.
# The production full-history reference is the pandas ``ewm(adjust=False,
# ignore_na=False)`` recurrence; ``recursive_kernel`` reproduces it exactly
# (including old_wt reset, NaN/Inf missing handling, min_periods gating, first
# true-range = NaN, and recursive (not SMA) Wilder seeding), so
# full-history == checkpoint-bootstrap == checkpoint-resume holds by
# construction and is verified in tests/operators/test_recursive_kernel_parity.py.
from recursive_kernel import adx_segment as _adx_segment
from recursive_kernel import atr_wilder_segment as _atr_segment
from recursive_kernel import ema_segment as _ema_segment
from recursive_kernel import macd_segment as _macd_segment
from recursive_kernel import rsi_wilder_segment as _rsi_segment


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


def _implementation_hash(canonical: str) -> str:
    """R5-13: hash of the ACTUAL kernel implementation (code object, not the
    human-written ``semantic_version``).  The ``semantic_version`` is a manual
    contract a developer can forget to bump; ``implementation_hash`` is a machine
    fact — editing the recurrence changes the hash and invalidates checkpoints
    even when nobody bumps the version string."""
    import hashlib

    kernel = {
        "ts_ema": _ema_segment,
        "RSI_WILDER": _rsi_segment,
        "ATR_WILDER": _atr_segment,
        "ADX": _adx_segment,
        "MACD_line": _macd_segment,
        "MACD_signal": _macd_segment,
        "MACD_hist": _macd_segment,
        "ts_ewm_std": _ewm_moment_segment,
        "ts_ewm_var": _ewm_moment_segment,
        "ts_ewm_cov": _ewm_moment_segment,
        "ts_ewm_corr": _ewm_moment_segment,
    }.get(canonical)
    if kernel is None:
        return "unknown"
    code = getattr(kernel, "__code__", None)
    if code is None:
        return "unknown"
    # Deterministic digest: bytecode + sorted constants; never embeds memory
    # addresses, stable across interpreter restarts.
    payload = (code.co_code, tuple(sorted(map(repr, code.co_consts))))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _effective_identity(canonical, input_identity, inputs, params, spec):
    identity = dict(input_identity)
    identity["__stateful_runtime__"] = {
        "canonical": canonical,
        "semantic_version": spec.semantic_version,
        "state_schema_version": spec.state_schema_version,
        "missing_policy": spec.missing_policy,
        "input_names": sorted(str(name) for name in inputs),
        "params": {str(key): params[key] for key in sorted(params)},
        # R5-13: bind the kernel implementation hash so an edited recurrence (not
        # just a bumped version string) invalidates old checkpoints.
        "implementation_hash": _implementation_hash(canonical),
        # R5-12: any data-version / snapshot / lineage / calendar fields the
        # caller already places in ``input_identity`` flow into the fingerprint
        # verbatim.  A data revision (e.g. a restated close on the same date)
        # that changes these MUST change the identity so the stale checkpoint is
        # rejected and the factor replays from the earliest changed timestamp.
        "data_version": {
            str(key): input_identity[key]
            for key in sorted(input_identity)
            if any(tag in str(key).lower() for tag in (
                "version", "snapshot", "revision", "lineage", "calendar",
                "dataset_id", "asof", "as_of", "partition", "grain",
            ))
        },
    }
    return identity


def _finite_state(state, key, default):
    value = state.get(key, default)
    try:
        return float(value) if np.isfinite(float(value)) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _ewm_moment_segment(x, y, state, span, output):
    """Exact pandas ``ewm(span, adjust=False, bias=False)`` recurrence."""
    alpha = 2.0 / (span + 1.0)
    beta = 1.0 - alpha
    count = int(state.get("observation_count", 0) or 0)
    mean_x = state.get("mean_x")
    mean_y = state.get("mean_y")
    mean_x = float(mean_x) if mean_x is not None and np.isfinite(mean_x) else None
    mean_y = float(mean_y) if mean_y is not None and np.isfinite(mean_y) else None
    moment_x = _finite_state(state, "second_moment_x", 0.0)
    moment_y = _finite_state(state, "second_moment_y", 0.0)
    cross = _finite_state(state, "cross_moment", 0.0)
    weight_sum = _finite_state(state, "weight_sum", 1.0)
    weight_sq = _finite_state(state, "squared_weight_sum", 1.0)
    effective_weight = _finite_state(state, "effective_weight", 1.0)
    out = np.full(x.shape, np.nan)

    for i, (xv, yv) in enumerate(zip(x, y)):
        observed = bool(np.isfinite(xv) and np.isfinite(yv))
        if mean_x is not None:
            effective_weight *= beta
            weight_sum *= beta
            weight_sq *= beta * beta
        if observed:
            xv, yv = float(xv), float(yv)
            count += 1
            if mean_x is None:
                mean_x, mean_y = xv, yv
            else:
                old_x, old_y = mean_x, mean_y
                denom = effective_weight + alpha
                if old_x != xv:
                    mean_x = (effective_weight * old_x + alpha * xv) / denom
                if old_y != yv:
                    mean_y = (effective_weight * old_y + alpha * yv) / denom
                moment_x = (
                    effective_weight * (moment_x + (old_x - mean_x) ** 2)
                    + alpha * (xv - mean_x) ** 2
                ) / denom
                moment_y = (
                    effective_weight * (moment_y + (old_y - mean_y) ** 2)
                    + alpha * (yv - mean_y) ** 2
                ) / denom
                cross = (
                    effective_weight * (cross + (old_x - mean_x) * (old_y - mean_y))
                    + alpha * (xv - mean_x) * (yv - mean_y)
                ) / denom
                weight_sum += alpha
                weight_sq += alpha * alpha
                effective_weight += alpha
                weight_sum /= effective_weight
                weight_sq /= effective_weight * effective_weight
                effective_weight = 1.0
        correction_denom = weight_sum * weight_sum - weight_sq
        if count >= 2 and correction_denom > 0.0:
            correction = weight_sum * weight_sum / correction_denom
            var_x = correction * moment_x
            var_y = correction * moment_y
            cov = correction * cross
            if output == "var":
                out[i] = var_x
            elif output == "std":
                out[i] = np.sqrt(max(var_x, 0.0))
            elif output == "cov":
                out[i] = cov
            elif var_x > 0.0 and var_y > 0.0:
                out[i] = cov / np.sqrt(var_x * var_y)

    state.update({
        "effective_weight": effective_weight,
        "weight_sum": weight_sum,
        "squared_weight_sum": weight_sq,
        "observation_count": count,
        "mean_x": np.nan if mean_x is None else mean_x,
        "mean_y": np.nan if mean_y is None else mean_y,
        "second_moment_x": moment_x,
        "second_moment_y": moment_y,
        "cross_moment": cross,
    })
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
    elif canonical in {"ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr"}:
        x = _array(inputs["x"], "x")
        y = _array(inputs.get("y", inputs["x"]), "y")
        values, state = _ewm_moment_segment(
            x, y, state,
            _positive(options.get("span", options.get("window", 20)), "span"),
            canonical.removeprefix("ts_ewm_"),
        )
    else:
        raise ValueError(f"stateful runtime not implemented for {canonical}")
    state["last_timestamp"] = ts[-1].isoformat()
    new_checkpoint = StatefulCheckpointRegistry.create_checkpoint(canonical, instrument=str(instrument), as_of=state["last_timestamp"], state=state, input_identity=effective_identity)
    return StatefulSegmentResult(values=values, checkpoint=new_checkpoint, state=state)


__all__ = ["StatefulSegmentResult", "execute_stateful_segment"]
