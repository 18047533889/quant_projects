# -*- coding: utf-8 -*-
"""Checkpoint-backed segmented execution for recursive stateful factors.

Audit §11: a factor whose root operator is a segmented-execution stateful
canonical (``SEGMENTED_EXECUTION_CANONICALS``) can resume from a per-instrument
checkpoint instead of re-reading the full causal history.  This module decides
when the checkpoint path applies, executes the segment per instrument, persists
the new checkpoint, and returns ``None`` whenever the path cannot produce a
correct value so ``run_incremental`` falls back to full-history replay.

Two execution modes:

* ``bootstrap`` — no usable checkpoint yet: run the operator from the dataset
  origin over the look-back window (``starts_at_dataset_origin``), slice the
  warm-up prefix away, and persist the terminal checkpoint for the next run.
* ``incremental`` — a usable checkpoint exists before the output window: resume
  the recurrence over ``[output_start, output_end]`` only and persist the new
  checkpoint.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cleaned_operators.production_hardening import SEGMENTED_EXECUTION_CANONICALS
from stateful_contract import StatefulCheckpointRegistry
from stateful_runtime import execute_stateful_segment

logger = logging.getLogger(__name__)


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

# Positional leaf columns of each segmented canonical, in IR-input order, mapped
# to the ``execute_stateful_segment`` input-key contract.
_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "ts_ema": ("x",),
    "ts_ewm_std": ("x",),
    "ts_ewm_var": ("x",),
    "ts_ewm_cov": ("x", "y"),
    "ts_ewm_corr": ("x", "y"),
    "RSI_WILDER": ("x",),
    "ATR_WILDER": ("high", "low", "close"),
    "ADX": ("high", "low", "close"),
    "MACD_line": ("x",),
    "MACD_signal": ("x",),
    "MACD_hist": ("x",),
}


def stateful_canonicals_in_ir(ir) -> list[str]:
    """Return stateful-registry canonicals in the IR, topological (post-order)."""
    out: list[str] = []

    def walk(node) -> None:
        for child in node.inputs:
            walk(child)
        if StatefulCheckpointRegistry.get(node.op) is not None:
            out.append(node.op)

    walk(ir)
    return out


def segmented_incremental_available(*, ir) -> bool:
    """True when the factor root is a segmented-execution canonical and it is the
    only stateful operator in the DAG (a single checkpoint per instrument).

    Interior stateful nodes (stateful output feeding downstream arithmetic or
    ranking) still run the standard narrow-and-replay path.
    """
    canonicals = stateful_canonicals_in_ir(ir)
    if len(canonicals) != 1:
        return False
    return canonicals[0] == ir.op and ir.op in SEGMENTED_EXECUTION_CANONICALS


def _source_snapshot_scope(source: Any) -> str:
    """Window-independent data-snapshot scope for the checkpoint identity.

    R10-P0-019: ``compute_data_scope`` includes the query window (start/end),
    which must NOT be part of the checkpoint fingerprint — the SAME data
    snapshot queried over a different incremental window must resume the same
    checkpoint.  Only the dataset / read options / snapshot-id carry the data
    identity.
    """
    try:
        import copy

        from storage.data_scope import compute_data_scope

        probe = copy.copy(source)
        probe.start_date = None
        probe.end_date = None
        return compute_data_scope(probe)
    except Exception:  # pragma: no cover - scope failure must not block resume
        return "ephemeral"


def _boundary_timeline(source: Any, column_name: str, *, since: Any, start: Any) -> list[pd.Timestamp]:
    """NEW-P0-26: the source bar timeline in ``[checkpoint.as_of, segment_start]``.

    Probes ONLY the boundary window (never the full history) by widening the
    source from the checkpoint's last-covered bar up to the segment start, so the
    chunk auditor can see any bar that a direct resume would silently skip.
    Returns ``[]`` when the source cannot be probed (the auditor then cannot
    prove a gap — the as_of/state.last_timestamp consistency check still applies).
    """
    try:
        import copy

        probe = copy.copy(source)
        if hasattr(probe, "start_date"):
            probe.start_date = since
        if hasattr(probe, "end_date"):
            probe.end_date = start
        series = probe.load_column(column_name)
        idx = series.index
        if isinstance(idx, pd.MultiIndex):
            if "timestamp" in idx.names:
                idx = idx.get_level_values("timestamp")
            else:
                idx = idx.get_level_values(0)
        return sorted(pd.to_datetime(list(dict.fromkeys(idx)), utc=True))
    except Exception:
        return []


def audit_segment_continuity(
    *,
    checkpoint: Any,
    segment_start: Any,
    source_timeline: Any = None,
) -> bool:
    """Chunk auditor (NEW-P0-26 / NEW-P0-30): is a checkpoint chain CONTIGUOUS?

    Validates that resuming from ``checkpoint`` onto a segment beginning at
    ``segment_start`` does not silently skip any bar:

    * **false coverage claim** — ``checkpoint.as_of`` must equal the state's own
      ``last_timestamp``.  A checkpoint record that claims coverage up to/through
      a bar the state did not actually cover (e.g. as_of says Monday but the last
      processed bar was Friday) must NOT be resumed with forward-bar
      interpolation — the intervening Monday would be lost.
    * **gap** — when ``source_timeline`` exposes the boundary, any source bar
      STRICTLY between the checkpoint's last-covered bar and ``segment_start``
      (e.g. Monday between Friday and Tuesday) is a missing bar that a direct
      resume would skip.  The auditor fails closed: resume impossible.
    * **boundary mismatch** — the last observable source bar before
      ``segment_start`` must equal the checkpoint's last-covered bar; a stale /
      not-in-data checkpoint is rejected rather than resumed.

    Returns ``True`` = contiguous / resume OK; ``False`` = gap -> full replay.
    """
    state = getattr(checkpoint, "state", None) or {}
    state_last = state.get("last_timestamp") if isinstance(state, Mapping) else None
    try:
        as_of = _as_utc(checkpoint.as_of)
    except (ValueError, TypeError):
        return False
    if state_last is not None:
        try:
            if _as_utc(state_last) != as_of:
                return False
        except (ValueError, TypeError):
            return False
    try:
        start_ts = _as_utc(segment_start)
    except (ValueError, TypeError):
        return False
    if start_ts <= as_of:
        return False
    if source_timeline is not None:
        try:
            bars = sorted(_as_utc(t) for t in source_timeline)
        except (ValueError, TypeError):
            bars = []
        if bars:
            between = [t for t in bars if as_of < t < start_ts]
            if between:
                return False
            boundary = [t for t in bars if t < start_ts]
            if boundary and boundary[-1] != as_of:
                return False
    return True


def _root_series_and_params(ir, canonical: str) -> tuple[list[str], dict[str, Any]] | None:
    """Extract the root's source columns and operator parameters.

    Positional arguments lower to ``literal`` inputs and keyword arguments land
    in the node ``attrs``; both are mapped to parameter names through the
    operator's declared ``param_names`` (series inputs are the names in
    ``_INPUT_KEYS[canonical]``).  Returns ``None`` when a series position does
    not hold a plain source column (the checkpoint path cannot fabricate
    intermediate data).
    """
    keys = _INPUT_KEYS.get(canonical)
    if not keys:
        return None
    from cleaned_operators.registry import OperatorRegistry

    implementation = OperatorRegistry.get(canonical)
    param_names = tuple(
        getattr(implementation.metadata, "param_names", ()) or ()
    ) if implementation is not None else ()
    series: list[str] = []
    params: dict[str, Any] = {}
    for index, child in enumerate(ir.inputs):
        name = param_names[index] if index < len(param_names) else None
        if child.op == "literal":
            if name is not None and name not in keys:
                params[name] = child.attrs.get("value")
            continue
        if child.op != "column":
            return None
        if name is not None and name not in keys:
            return None
        column_name = child.attrs.get("name")
        if not str(column_name):
            return None
        series.append(str(column_name))
    for key, value in ir.attrs.items():
        if key != "dtype":
            params[key] = value
    if len(series) != len(keys):
        return None
    return series, params


def try_stateful_segmented_incremental(
    *,
    factor_id: str,
    ir,
    source,
    store,
    start,
    end,
    bootstrap: bool,
) -> tuple[pd.Series, dict[str, Any]] | None:
    """Attempt checkpoint-backed segmented execution over ``[start, end]``.

    With ``bootstrap=False`` every instrument must already have a checkpoint
    strictly before ``start``; otherwise the attempt returns ``None`` and the
    caller falls back.  With ``bootstrap=True`` the operator is run from the
    dataset origin over the full supplied window and the terminal checkpoint is
    persisted (used to seed the checkpoint store on the first run).

    Audit #378: a persisted checkpoint's ``as_of`` is the LAST FULLY COMMITTED
    input timestamp.  The output window always re-computes its terminal bar
    (1-bar inclusive overlap), so the committed checkpoint is the state one bar
    before the end (multi-bar) and a single-bar segment has no new fully
    committed bar and never overwrites the checkpoint.  All instruments run
    first; their pending checkpoints are then committed together atomically via
    ``StatefulCheckpointStore.commit_batch`` — a failure of any instrument (or
    of the batch commit itself) abandons the whole segment and returns ``None``,
    so no checkpoint is ever persisted for a partially-successful segment.
    """
    canonical = ir.op
    if canonical not in SEGMENTED_EXECUTION_CANONICALS:
        return None
    extracted = _root_series_and_params(ir, canonical)
    if extracted is None:
        return None
    input_names, params = extracted
    input_keys = _INPUT_KEYS[canonical]
    try:
        series = {name: source.load_column(name) for name in input_names}
    except Exception as exc:  # source unavailability -> standard path
        logger.warning("stateful segmented source load failed: %s", exc)
        return None
    if not series:
        return None

    # Align every input onto the shared (timestamp, instrument) anchor grid.
    anchor = series[input_names[0]].index
    if not isinstance(anchor, pd.MultiIndex):
        return None
    frames = {name: series[name].unstack(level="instrument") for name in input_names}
    reference = frames[input_names[0]]
    instruments = list(reference.columns)
    if len(reference.index) == 0:
        return None
    # The segment API requires tz-aware monotonic timestamps; the output panel
    # keeps the source's original (naive) index so the result matches a full run.
    segment_timestamps = pd.to_datetime(reference.index, utc=True)

    # R10-P0-019: the checkpoint identity must be bound to the DATA SNAPSHOT,
    # not just the formula.  ``_effective_identity`` fingerprints everything in
    # ``input_identity``, and ``require_for_segment`` re-checks it on resume —
    # so a checkpoint written against an older dataset / read-mode / snapshot
    # is rejected (fail-closed to full replay) instead of resuming stale state.
    # The scope is computed WITHOUT the query window (start/end) so an
    # incremental resume across a different output window still matches.
    source_scope = _source_snapshot_scope(source)
    input_identity: dict[str, Any] = {
        "factor_id": str(factor_id),
        "canonical": canonical,
        "input_columns": input_names,
        "params": params,
        "source_snapshot_scope": source_scope,
    }

    # Audit #379: the complete (timestamp, instrument) grid the source anchors.
    # ``incremental`` must match the full-history shape exactly — rows missing
    # from a panel cell are preserved as NaN, never dropped.
    anchor_index = pd.MultiIndex.from_product(
        [reference.index, instruments],
        names=["timestamp", "instrument"],
    )

    out = np.full((len(segment_timestamps), len(instruments)), np.nan, dtype=float)
    # Audit #378: a checkpoint's ``as_of`` is the LAST FULLY COMMITTED input
    # timestamp.  All instruments run first; only after every one succeeds are
    # their pending checkpoints committed atomically via ``store.commit_batch``.
    pending: dict[str, Any] = {}
    # NEW-P0-26/NEW-P0-30: probe the boundary window once (using the first
    # instrument's checkpoint as_of) so the chunk auditor can reject a
    # non-contiguous resume — a bar strictly between the checkpoint's last-covered
    # bar and the segment start would otherwise be silently skipped.
    boundary_timeline: list[pd.Timestamp] | None = None
    for j, instrument in enumerate(instruments):
        checkpoint = None if bootstrap else store.load_latest(
            factor_id, canonical, instrument, before=start
        )
        if checkpoint is None and not bootstrap:
            # At least one instrument lacks a usable checkpoint: full replay.
            return None
        if checkpoint is not None and not bootstrap:
            if boundary_timeline is None:
                boundary_timeline = _boundary_timeline(
                    source, input_names[0], since=checkpoint.as_of, start=start
                )
            if not audit_segment_continuity(
                checkpoint=checkpoint,
                segment_start=start,
                source_timeline=boundary_timeline,
            ):
                # NEW-P0-30: the checkpoint's last-covered bar is NOT the bar
                # immediately before the segment start (gap / false coverage
                # claim) — a direct resume would silently drop a bar.
                return None
        starts_at_origin = bootstrap or checkpoint is None
        inputs: Mapping[str, np.ndarray] = {
            key: frames[name][instrument].to_numpy(dtype=float)
            for key, name in zip(input_keys, input_names)
        }
        try:
            if len(segment_timestamps) >= 2:
                # Multi-bar branch: the output window re-computes its terminal
                # bar (1-bar inclusive overlap), so the checkpoint to persist is
                # the state just before the terminal bar — ``checkpoint.as_of ==
                # last fully committed input timestamp == t(n-1)`` and the next
                # segment resumes from that boundary.
                first = execute_stateful_segment(
                    canonical,
                    {key: values[:-1] for key, values in inputs.items()},
                    timestamps=segment_timestamps[:-1],
                    instrument=str(instrument),
                    input_identity=input_identity,
                    params=params,
                    checkpoint=checkpoint,
                    starts_at_dataset_origin=starts_at_origin,
                )
                last = execute_stateful_segment(
                    canonical,
                    {key: values[-1:] for key, values in inputs.items()},
                    timestamps=segment_timestamps[-1:],
                    instrument=str(instrument),
                    input_identity=input_identity,
                    params=params,
                    checkpoint=first.checkpoint,
                )
                out[:, j] = np.concatenate([first.values, last.values])
                pending[instrument] = first.checkpoint
            else:
                # Single-bar branch: the one bar is only re-computed (1-bar
                # inclusive overlap) — there is NO new fully-committed bar, so
                # ``checkpoint.as_of`` (the last fully committed input timestamp)
                # is unchanged.  We must NOT overwrite the checkpoint: the
                # incoming one (or, under bootstrap, the absence of one) remains
                # the resume point for the next segment.
                single = execute_stateful_segment(
                    canonical,
                    inputs,
                    timestamps=segment_timestamps,
                    instrument=str(instrument),
                    input_identity=input_identity,
                    params=params,
                    checkpoint=checkpoint,
                    starts_at_dataset_origin=starts_at_origin,
                )
                out[:, j] = single.values
        except Exception as exc:
            logger.warning("stateful segment failed for %s/%s: %s", factor_id, instrument, exc)
            return None

    # Every instrument succeeded.  Commit the batch atomically: any failure here
    # abandons the whole segment (no checkpoint is persisted), fail-closed.
    if pending:
        try:
            store.commit_batch(factor_id, list(pending.values()))
        except Exception as exc:
            logger.warning("stateful checkpoint batch commit failed for %s: %s", factor_id, exc)
            return None

    panel = pd.DataFrame(out, index=reference.index, columns=instruments)
    result_series = panel.stack(future_stack=True)
    result_series = result_series.reindex(anchor_index)
    result_series.index = result_series.index.set_names(["timestamp", "instrument"])
    mode = {
        "mode": "stateful_segmented",
        "bootstrap": bool(bootstrap),
        "canonical": canonical,
        "instruments": len(instruments),
    }
    return result_series, mode


__all__ = [
    "segmented_incremental_available",
    "stateful_canonicals_in_ir",
    "try_stateful_segmented_incremental",
]
