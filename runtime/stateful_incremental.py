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

import enum
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cleaned_operators.production_hardening import SEGMENTED_EXECUTION_CANONICALS
from stateful_contract import StatefulCheckpointRegistry
from stateful_runtime import _implementation_hash, execute_stateful_segment

logger = logging.getLogger(__name__)


class SegmentedFallbackReason(str, enum.Enum):
    """R40 #249：segmented 路径各失败路径的**类型化**回退原因。

    前四类（CHECKPOINT_CORRUPTION / STATE_CORRUPTION / PIT_VIOLATION /
    CALENDAR_ERROR）是系统损坏，必须 hard fail——绝不静默回退 full replay
    （那会掩盖损坏并用重算覆盖疑似损坏的 checkpoint）。其余是可接受的正常
    fallback（回退 full replay）。
    """

    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    STATE_CORRUPTION = "state_corruption"
    PIT_VIOLATION = "pit_violation"
    CALENDAR_ERROR = "calendar_error"
    NO_CHECKPOINT = "no_checkpoint"
    INCOMPATIBLE = "incompatible"
    UNSUPPORTED = "unsupported"


#: 系统损坏类：hard fail（raise），不 fallback。
_CORRUPTION_REASONS = frozenset(
    {
        SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
        SegmentedFallbackReason.STATE_CORRUPTION,
        SegmentedFallbackReason.PIT_VIOLATION,
        SegmentedFallbackReason.CALENDAR_ERROR,
    }
)


class StatefulSegmentedCorruptionError(RuntimeError):
    """R40 #249：segmented 路径遇到系统损坏（checkpoint/state/PIT/calendar）。

    抛出而非返回 None，强制 hard fail——调用方不得把它当普通 fallback 吞掉。
    """


class BoundaryProbeResult(str, enum.Enum):
    """R40 #245：边界连续性探测的类型化结果。

    * ``PROVEN_CONTIGUOUS`` — 源时间线可探且无 gap（可 resume）；
    * ``GAP`` — 时间线内存在严格位于 checkpoint.as_of 与 segment_start 之间的 bar；
    * ``UNKNOWN`` — 源无法探测（timeout / 空 / 异常）→ production 禁止 resume。
    """

    PROVEN_CONTIGUOUS = "proven_contiguous"
    GAP = "gap"
    UNKNOWN = "unknown"


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


class SourceSnapshotIdentityUnavailableError(RuntimeError):
    """R40 #244：production 下源 snapshot identity 无法解析。

    production 语义：identity 未解析 → checkpoint resume disabled + full replay
    （不生成可复用 identity）。research 语义允许 unique run-scoped ephemeral ID
    但禁止跨 run resume（每次生成不同 ID，checkpoint 永不命中）。
    """


def _source_snapshot_scope(source: Any, *, mode: str = "research") -> str:
    """Window-independent data-snapshot scope for the checkpoint identity.

    R10-P0-019: ``compute_data_scope`` includes the query window (start/end),
    which must NOT be part of the checkpoint fingerprint — the SAME data
    snapshot queried over a different incremental window must resume the same
    checkpoint.  Only the dataset / read options / snapshot-id carry the data
    identity.

    R40 #244: 旧 ``except Exception: return "ephemeral"`` 让两个源同时
    resolver 故障时共享同一个 ``"ephemeral"`` 身份，被当成同一份数据
    checkpoint resume。现在：

    * production：解析失败 → 抛 :class:`SourceSnapshotIdentityUnavailableError`
      （上层回退 full replay，不生成可复用 identity）；
    * research：返回 **unique run-scoped** ``"ephemeral:<uuid>"`` —— 每次调用
      不同，跨 run checkpoint 永不命中（禁止跨 run resume）。
    """
    try:
        import copy

        from storage.data_scope import compute_data_scope

        probe = copy.copy(source)
        probe.start_date = None
        probe.end_date = None
        return compute_data_scope(probe)
    except Exception as exc:
        if str(mode).strip().lower() == "production":
            raise SourceSnapshotIdentityUnavailableError(
                f"production source snapshot identity unresolved: "
                f"{type(exc).__name__}: {exc} — checkpoint resume disabled, "
                "full replay required (no reusable identity)"
            ) from exc
        return f"ephemeral:{uuid.uuid4().hex}"


def _boundary_timeline(
    source: Any, column_name: str, *, since: Any, start: Any, mode: str = "research"
) -> tuple[BoundaryProbeResult, list[pd.Timestamp]]:
    """NEW-P0-26 / R40 #245: probe the source bar timeline in
    ``[checkpoint.as_of, segment_start]``.

    Probes ONLY the boundary window (never the full history) by widening the
    source from the checkpoint's last-covered bar up to the segment start, so the
    chunk auditor can see any bar that a direct resume would silently skip.

    Returns a typed :class:`BoundaryProbeResult` + timeline list:

    * 探测成功且时间线非空 → ``(PROVEN_CONTIGUOUS, bars)``；
    * 探测成功但时间线为空 → ``(PROVEN_CONTIGUOUS, [])``（空边界窗口本身无可证明
      的 gap）；
    * 探测失败 → ``(UNKNOWN, [])`` —— production 禁止 checkpoint resume（旧行为
      返回 ``[]`` 让审计只看 as_of 检查就放行，这正是 #245 要堵的洞）。
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
        bars = sorted(pd.to_datetime(list(dict.fromkeys(idx)), utc=True))
        return BoundaryProbeResult.PROVEN_CONTIGUOUS, bars
    except Exception as exc:
        if str(mode).strip().lower() == "production":
            return BoundaryProbeResult.UNKNOWN, []
        # research: 探测失败允许只看 as_of/state 一致性（旧行为）。
        return BoundaryProbeResult.UNKNOWN, []


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


class AxisIdentityCertificate:
    """R40 #247：多输入 checkpoint 的轴身份证书。

    unstack 后旧代码只以 ``frames[input_names[0]]`` 为 anchor，不验证所有
    inputs 是否共享 timestamp index / instrument columns / 顺序 / source
    snapshot —— 一个 input 多一根列、少一个 instrument 或顺序不同都会被当作
    同一份数据继续 checkpoint。``verify_frames_share_identity`` 在 production
    下要求 exact match。
    """

    @staticmethod
    def frames_identity(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
        """抽取所有 input frame 的轴身份（供比较）。"""
        first = next(iter(frames.values()))
        return {
            "index": tuple(first.index),
            "columns": tuple(first.columns),
            "index_dtype": str(first.index.dtype),
        }

    @staticmethod
    def verify_frames_share_identity(
        frames: Mapping[str, pd.DataFrame], *, mode: str = "research"
    ) -> bool:
        """所有 input frames 必须共享同一 index / columns / 顺序 / dtype。

        production 要求 exact match；research 允许宽松（仅比对锚 frame）。
        返回 False 表示 axis identity 不一致（production 下调用方应回退 full
        replay，绝不带错位数据 checkpoint resume）。
        """
        if not frames:
            return True
        anchor = next(iter(frames.values()))
        for name, frame in frames.items():
            if mode == "production":
                if not anchor.index.equals(frame.index):
                    return False
                if not anchor.columns.equals(frame.columns):
                    return False
            else:
                # research：宽松——只要求长度与列数一致。
                if len(anchor.index) != len(frame.index) or len(anchor.columns) != len(frame.columns):
                    return False
        return True


def _checkpoint_input_identity(
    *,
    factor_id: str,
    canonical: str,
    input_names: list[str],
    params: dict[str, Any],
    source_scope: str,
    market: str = "",
    calendar_version: str = "",
    timezone: str = "",
    universe_membership_hash: str = "",
    price_basis_policy: str = "",
    field_contract_digests: Mapping[str, str] | None = None,
    operator_semantic_contract_digest: str = "",
    numeric_semantics_hash: str = "",
    missing_support_policy: str = "",
) -> dict[str, Any]:
    """R40 #248：checkpoint input identity（含 market / calendar / timezone /
    universe / price basis / field / operator contract / numeric semantics /
    missing policy）。任一上下文不同 => 不同 checkpoint identity。
    """
    return {
        "factor_id": str(factor_id),
        "canonical": canonical,
        "input_columns": input_names,
        "params": params,
        "source_snapshot_scope": source_scope,
        "market": str(market or ""),
        "calendar_version": str(calendar_version or ""),
        "timezone": str(timezone or ""),
        "universe_membership_hash": str(universe_membership_hash or ""),
        "price_basis_policy": str(price_basis_policy or ""),
        "field_contract_digests": dict(field_contract_digests or {}),
        "operator_semantic_contract_digest": str(operator_semantic_contract_digest or ""),
        "numeric_semantics_hash": str(numeric_semantics_hash or ""),
        "missing_support_policy": str(missing_support_policy or ""),
    }


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
    mode: str = "research",
    market: str = "",
    calendar_version: str = "",
    timezone: str = "",
    universe_membership_hash: str = "",
    price_basis_policy: str = "",
    field_contract_digests: Mapping[str, str] | None = None,
    operator_semantic_contract_digest: str = "",
    numeric_semantics_hash: str = "",
    missing_support_policy: str = "",
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

    R40 mode 语义：
    * ``mode="production"`` — 系统损坏（checkpoint/state/PIT/calendar）hard fail
      （抛 :class:`StatefulSegmentedCorruptionError`）；源 identity 未解析 /
      轴 identity 不一致 / boundary UNKNOWN → 回退 full replay（返回 None）。
    * ``mode="research"``（默认）— 所有失败路径回退 full replay（None）。

    R40 #244/#245/#246/#247/#248/#249 具体行为见各函数文档。
    """
    is_production = str(mode).strip().lower() == "production"

    def _hard_fail(reason: SegmentedFallbackReason, detail: str) -> None:
        if is_production and reason in _CORRUPTION_REASONS:
            raise StatefulSegmentedCorruptionError(
                f"stateful segmented corruption ({reason.value}): {detail}"
            )
        logger.warning("stateful segmented fallback reason=%s detail=%s", reason.value, detail)

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
        _hard_fail(SegmentedFallbackReason.NO_CHECKPOINT, f"source load failed: {exc}")
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
    # R40 #247: production 要求所有 inputs 共享同一轴身份（index/columns/order）。
    if not AxisIdentityCertificate.verify_frames_share_identity(frames, mode=mode):
        _hard_fail(
            SegmentedFallbackReason.INCOMPATIBLE,
            "input frames do not share identical axes (timestamp index / "
            "instrument columns / ordering)",
        )
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
    # R40 #244: production 下 scope 未解析 -> 回退 full replay。
    try:
        source_scope = _source_snapshot_scope(source, mode=mode)
    except SourceSnapshotIdentityUnavailableError as exc:
        _hard_fail(SegmentedFallbackReason.NO_CHECKPOINT, str(exc))
        return None
    # R40 #248: checkpoint identity 绑定 market / calendar / timezone /
    # universe / price basis / field / operator contract / numeric semantics /
    # missing policy —— 任一上下文不同 => 不同 checkpoint identity。
    input_identity = _checkpoint_input_identity(
        factor_id=factor_id,
        canonical=canonical,
        input_names=input_names,
        params=params,
        source_scope=source_scope,
        market=market,
        calendar_version=calendar_version,
        timezone=timezone,
        universe_membership_hash=universe_membership_hash,
        price_basis_policy=price_basis_policy,
        field_contract_digests=field_contract_digests,
        operator_semantic_contract_digest=operator_semantic_contract_digest,
        numeric_semantics_hash=numeric_semantics_hash,
        missing_support_policy=missing_support_policy,
    )

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
    # R40 #245/#246: per-(checkpoint.as_of, start) 边界探测缓存 —— 不再只用第一
    # 个 instrument 的 checkpoint.as_of probe 一次后对所有 instruments 共用。
    boundary_probe_cache: dict[tuple[str, str], tuple[BoundaryProbeResult, list[pd.Timestamp]]] = {}
    for j, instrument in enumerate(instruments):
        checkpoint = None if bootstrap else store.load_latest(
            factor_id, canonical, instrument, before=start
        )
        if checkpoint is None and not bootstrap:
            # At least one instrument lacks a usable checkpoint: full replay.
            _hard_fail(SegmentedFallbackReason.NO_CHECKPOINT, f"{instrument} lacks checkpoint")
            return None
        if checkpoint is not None and not bootstrap:
            try:
                as_of_key = _as_utc(checkpoint.as_of)
            except (ValueError, TypeError):
                _hard_fail(
                    SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
                    f"{instrument} checkpoint.as_of unparseable: {checkpoint.as_of!r}",
                )
                return None
            state = getattr(checkpoint, "state", None) or {}
            state_last = state.get("last_timestamp") if isinstance(state, Mapping) else None
            if state_last is not None:
                try:
                    if _as_utc(state_last) != as_of_key:
                        _hard_fail(
                            SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
                            f"{instrument} false coverage claim: checkpoint.as_of={as_of_key} "
                            f"!= state.last_timestamp={state_last}",
                        )
                        return None
                except (ValueError, TypeError):
                    _hard_fail(
                        SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
                        f"{instrument} state.last_timestamp unparseable: {state_last!r}",
                    )
                    return None
            cache_key = (str(as_of_key), str(start))
            if cache_key not in boundary_probe_cache:
                boundary_probe_cache[cache_key] = _boundary_timeline(
                    source, input_names[0], since=as_of_key, start=start, mode=mode
                )
            probe_result, boundary_timeline = boundary_probe_cache[cache_key]
            if probe_result is BoundaryProbeResult.UNKNOWN:
                # R40 #245: production UNKNOWN -> 禁止 checkpoint resume。
                _hard_fail(
                    SegmentedFallbackReason.CALENDAR_ERROR,
                    f"boundary probe UNKNOWN for {instrument} — cannot prove contiguity",
                )
                return None
            if not audit_segment_continuity(
                checkpoint=checkpoint,
                segment_start=start,
                source_timeline=boundary_timeline,
            ):
                # NEW-P0-30: gap / false coverage claim — direct resume would
                # silently drop a bar.
                _hard_fail(
                    SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
                    f"{instrument} boundary gap / mismatch at {start}",
                )
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
        except StatefulSegmentedCorruptionError:
            raise
        except Exception as exc:
            # R40 #249: production 下 segment 执行异常 = STATE_CORRUPTION hard fail。
            _hard_fail(
                SegmentedFallbackReason.STATE_CORRUPTION,
                f"segment execution failed for {instrument}: {type(exc).__name__}: {exc}",
            )
            return None

    # Every instrument succeeded.  Commit the batch atomically: any failure here
    # abandons the whole segment (no checkpoint is persisted), fail-closed.
    if pending:
        try:
            store.commit_batch(factor_id, list(pending.values()))
        except Exception as exc:
            _hard_fail(
                SegmentedFallbackReason.CHECKPOINT_CORRUPTION,
                f"batch commit failed for {factor_id}: {type(exc).__name__}: {exc}",
            )
            return None

    panel = pd.DataFrame(out, index=reference.index, columns=instruments)
    result_series = panel.stack(future_stack=True)
    result_series = result_series.reindex(anchor_index)
    result_series.index = result_series.index.set_names(["timestamp", "instrument"])
    mode_info = {
        "mode": "stateful_segmented",
        "bootstrap": bool(bootstrap),
        "canonical": canonical,
        "instruments": len(instruments),
        "execution_mode": mode,
    }
    return result_series, mode_info


# ---------------------------------------------------------------------------
# R44: node-level incremental planning + interior-stateful execution.
# 全部 ADDITIVE —— 不触碰 segmented_incremental_available /
# try_stateful_segmented_incremental / _checkpoint_input_identity 及其调用方。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateNodeIdentity:
    """R44：一个 stateful 子 DAG 节点的**跨因子共享**身份。

    关键点：身份**不**以 factor_id 为键。两个不同因子 DAG 只要在同一个
    source_scope 上共享同一个 canonical + 相同 bound params + 相同算子实现，
    就产生**相同**的 :meth:`stable_key` —— 因此它们的 checkpoint 可以跨因子
    复用（state 是数据 + 公式 + 实现的函数，与"哪个因子在消费它"无关）。

    字段：
    * ``subdag_semantic_identity`` — stateful 子 DAG 的语义指纹（canonical +
      上游 column/op 形状），不含 bound params；
    * ``bound_parameter_identity`` — bound params 的指纹（span=20 vs 30 不同）；
    * ``source_policy_identity`` — 源 snapshot scope（market/dataset/read-policy），
      不含查询窗口；
    * ``physical_implementation_id`` — 算子实现哈希（``stateful_runtime._implementation_hash``）；
    * ``checkpoint_schema_version`` — checkpoint 状态 schema 版本。
    """

    subdag_semantic_identity: str
    bound_parameter_identity: str
    source_policy_identity: str
    market: str
    frequency: str
    physical_implementation_id: str
    checkpoint_schema_version: str

    def stable_key(self) -> str:
        """确定性、可排序、json-safe 的稳定键。

        用 ``sort_keys`` + 紧凑分隔符 + ``allow_nan=False`` 保证跨进程/跨调用
        稳定；字段顺序固定，键可直接作为 checkpoint 目录/缓存键。
        """
        payload = json.dumps(
            {
                "subdag_semantic_identity": self.subdag_semantic_identity,
                "bound_parameter_identity": self.bound_parameter_identity,
                "source_policy_identity": self.source_policy_identity,
                "market": self.market,
                "frequency": self.frequency,
                "physical_implementation_id": self.physical_implementation_id,
                "checkpoint_schema_version": self.checkpoint_schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    """把任意 json-safe 映射折叠成稳定 sha256 摘要。"""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_state_node_identity(
    canonical: str,
    params: Mapping[str, Any] | None,
    *,
    source_scope: str,
    market: str = "",
    frequency: str = "",
    calendar_version: str = "",
    timezone: str = "",
    universe_membership_hash: str = "",
    price_basis_policy: str = "",
    field_contract_digests: Mapping[str, str] | None = None,
    operator_semantic_contract_digest: str = "",
    numeric_semantics_hash: str = "",
    missing_support_policy: str = "",
) -> StateNodeIdentity:
    """R44：计算一个 stateful 节点的跨因子共享身份。

    复用 :func:`_checkpoint_input_identity` 的上下文构建块（market / calendar /
    timezone / universe / price basis / field / operator contract / numeric
    semantics / missing policy），但**剥离 factor_id** —— 身份只绑定
    canonical + params + source_scope + 实现 + schema，因此两个共享
    ``ts_ema(close, 20)`` 的因子得到同一个 :meth:`StateNodeIdentity.stable_key`。

    ``subdag_semantic_identity`` 编码 canonical + 上游 column/op 形状（不含
    bound params）；``bound_parameter_identity`` 单独编码 params，使 span=20
    与 span=30 产生不同身份。
    """
    spec = StatefulCheckpointRegistry.get(canonical)
    checkpoint_schema_version = (
        spec.state_schema_version if spec is not None else ""
    )
    # 上游 column/op 形状：canonical 本身 + 输入列名（不含 params）。
    input_names = _INPUT_KEYS.get(canonical, ())
    subdag_semantic_identity = _stable_digest(
        {
            "canonical": canonical,
            "input_columns": sorted(str(n) for n in input_names),
        }
    )
    bound_parameter_identity = _stable_digest(
        {str(k): params[k] for k in sorted(params or {})}
    )
    # source_policy_identity 直接复用调用方传入的 source_scope（已不含查询窗口）。
    return StateNodeIdentity(
        subdag_semantic_identity=subdag_semantic_identity,
        bound_parameter_identity=bound_parameter_identity,
        source_policy_identity=str(source_scope),
        market=str(market or ""),
        frequency=str(frequency or ""),
        physical_implementation_id=_implementation_hash(canonical),
        checkpoint_schema_version=checkpoint_schema_version,
    )


class NodeIncrementalMode(str, enum.Enum):
    """R44：单个 IR 节点的增量执行模式。

    * ``LOAD_TODAY`` — S0 无状态：只需当前 bar；
    * ``LOAD_TAIL`` — S1 有限窗口：加载 ``[output_start - backward_history, ...]``；
    * ``RESTORE_STATE`` — S2 checkpointed：恢复 state 后跑增量 segment；
    * ``EVENT_ASOF`` — S4 event/PIT 时钟；
    * ``FULL_REPLAY`` — S5 / fail-closed：全历史重放。
    """

    LOAD_TODAY = "LOAD_TODAY"
    LOAD_TAIL = "LOAD_TAIL"
    RESTORE_STATE = "RESTORE_STATE"
    EVENT_ASOF = "EVENT_ASOF"
    FULL_REPLAY = "FULL_REPLAY"


@dataclass(frozen=True)
class NodePlan:
    """R44：单个 IR 节点的增量计划。"""

    node_id: str
    op: str
    mode: NodeIncrementalMode
    backward_history: int
    forward_impact: int | None
    state_node_identity: StateNodeIdentity | None = None  # mode == RESTORE_STATE 时设置


@dataclass(frozen=True)
class NodeIncrementalPlan:
    """R44：整棵因子 IR 的节点级增量计划。

    ``supportable=False`` 时调用方必须回退到既有路径（full-history replay）。
    """

    factor_id: str
    nodes: tuple[NodePlan, ...]  # 每个 IR 节点一个，post-order
    stateful_node_ids: tuple[str, ...]  # mode == RESTORE_STATE 的 node_id
    supportable: bool
    unsupported_reason: str | None = None


def _node_params(node: Any) -> dict[str, Any]:
    """从 IRNode 提取 bound params（kwargs attrs + positional literal inputs）。"""
    params: dict[str, Any] = {}
    if getattr(node, "attrs", None):
        params.update(dict(node.attrs))
    try:
        from cleaned_operators.registry import OperatorRegistry

        meta = getattr(OperatorRegistry.get(getattr(node, "op", "")), "metadata", None)
    except Exception:
        meta = None
    if meta is None:
        return params
    param_names = tuple(getattr(meta, "param_names", None) or ())
    for index, child in enumerate(getattr(node, "inputs", ()) or ()):
        if getattr(child, "op", None) == "literal":
            name = param_names[index] if index < len(param_names) else None
            if name:
                params.setdefault(name, getattr(child, "attrs", {}).get("value"))
    return params


def _classify_node_mode(op: str, params: Mapping[str, Any]) -> NodeIncrementalMode:
    """R44：单节点模式分类（P1 IncrementalContract 未落地时的本地最小分类器）。

    优先尝试 ``runtime.incremental_contract.resolve_incremental_contract``；若
    P1 尚未落地（import 失败），回退到本地逻辑：stateful + checkpoint registry
    → RESTORE_STATE；event-clock → EVENT_ASOF；有限窗口 → LOAD_TAIL；无状态
    → LOAD_TODAY；未知 → FULL_REPLAY（fail-closed）。
    """
    # 叶子 source column / literal：无状态，只需当前 bar。
    if op in ("column", "literal"):
        return NodeIncrementalMode.LOAD_TODAY
    try:
        from runtime.incremental_contract import resolve_incremental_contract

        contract = resolve_incremental_contract(op, params)
        mode = getattr(contract, "incremental_mode", None)
        mode_name = getattr(mode, "value", None) or str(mode)
        if mode_name == "CHECKPOINTED_STATE":
            return NodeIncrementalMode.RESTORE_STATE
        if mode_name == "EVENT_ASOF":
            return NodeIncrementalMode.EVENT_ASOF
        if mode_name == "FINITE_WINDOW":
            return NodeIncrementalMode.LOAD_TAIL
        if mode_name == "LOAD_TODAY":
            return NodeIncrementalMode.LOAD_TODAY
        return NodeIncrementalMode.FULL_REPLAY
    except Exception:
        pass
    # 本地最小分类器。
    try:
        from runtime.execution_contract import execution_contract, history_requirement

        contract = execution_contract(op)
        if contract.requires_full_history or contract.state_model != "stateless":
            if StatefulCheckpointRegistry.get(op) is not None:
                return NodeIncrementalMode.RESTORE_STATE
            return NodeIncrementalMode.FULL_REPLAY
        req = history_requirement(op, params)
        if req.is_event_clock:
            return NodeIncrementalMode.EVENT_ASOF
        if req.is_full_history:
            return NodeIncrementalMode.FULL_REPLAY
        if req.rows > 0:
            return NodeIncrementalMode.LOAD_TAIL
        return NodeIncrementalMode.LOAD_TODAY
    except Exception:
        return NodeIncrementalMode.FULL_REPLAY


def _node_backward_history(op: str, params: Mapping[str, Any]) -> int:
    """R44：单节点 backward_history（有限窗口的 warm-up 行数；stateful 为 0）。"""
    try:
        from runtime.execution_contract import history_requirement

        req = history_requirement(op, params)
        if req.is_full_history or req.is_event_clock:
            return 0
        return max(0, int(req.rows))
    except Exception:
        return 0


def _node_forward_impact(op: str, params: Mapping[str, Any]) -> int | None:
    """R44：单节点 forward_impact（``None`` = 无界）。"""
    try:
        from runtime.execution_contract import forward_impact

        return forward_impact(op, params)
    except Exception:
        return None


def plan_node_level_incremental(
    ir,
    *,
    factor_id: str,
    source_scope: str,
    market: str = "",
    frequency: str = "",
    calendar_version: str = "",
    timezone: str = "",
    universe_membership_hash: str = "",
    price_basis_policy: str = "",
    field_contract_digests: Mapping[str, str] | None = None,
    operator_semantic_contract_digest: str = "",
    numeric_semantics_hash: str = "",
    missing_support_policy: str = "",
) -> NodeIncrementalPlan:
    """R44：post-order 遍历 IR，为每个节点计算增量模式。

    对 mode == RESTORE_STATE 的节点计算 :class:`StateNodeIdentity`（state 跨
    因子共享）。以下情况 ``supportable=False``（带原因，调用方回退 full replay）：
    * 需要 checkpointed state 的**内部**节点，其上游无法仅 tail 重放；
    * 两个不同 source 的 stateful 节点需要 composite state（尚未支持）；
    * 任一节点解析为 FULL_REPLAY 且存在 checkpointed 兄弟节点（整体回退——
      递归重算本来就是全量）。
    """
    nodes: list[NodePlan] = []
    stateful_ids: list[str] = []
    unsupported_reason: str | None = None
    counter = {"n": 0}

    def walk(node: Any) -> None:
        for child in getattr(node, "inputs", ()) or ():
            walk(child)
        op = str(getattr(node, "op", ""))
        params = _node_params(node)
        mode = _classify_node_mode(op, params)
        node_id = f"n{counter['n']}"
        counter["n"] += 1
        identity = None
        if mode == NodeIncrementalMode.RESTORE_STATE:
            identity = compute_state_node_identity(
                op, params, source_scope=source_scope, market=market,
                frequency=frequency, calendar_version=calendar_version,
                timezone=timezone, universe_membership_hash=universe_membership_hash,
                price_basis_policy=price_basis_policy,
                field_contract_digests=field_contract_digests,
                operator_semantic_contract_digest=operator_semantic_contract_digest,
                numeric_semantics_hash=numeric_semantics_hash,
                missing_support_policy=missing_support_policy,
            )
            stateful_ids.append(node_id)
        nodes.append(
            NodePlan(
                node_id=node_id, op=op, mode=mode,
                backward_history=_node_backward_history(op, params),
                forward_impact=_node_forward_impact(op, params),
                state_node_identity=identity,
            )
        )

    walk(ir)

    # 支持性判定。
    if unsupported_reason is None:
        # 任一 FULL_REPLAY 节点 + 存在 checkpointed 兄弟 → 整体回退。
        has_stateful = any(n.mode == NodeIncrementalMode.RESTORE_STATE for n in nodes)
        has_full = any(n.mode == NodeIncrementalMode.FULL_REPLAY for n in nodes)
        if has_full and has_stateful:
            unsupported_reason = (
                "FULL_REPLAY node coexists with a checkpointed sibling; "
                "recursive recompute is full anyway"
            )
        elif has_full:
            unsupported_reason = "a node requires full-history replay"
        elif len(stateful_ids) > 1:
            # 多个 stateful 节点：仅当它们共享同一 source 才可能（当前不支持
            # composite state），保守回退。
            unsupported_reason = (
                "multiple stateful nodes require composite state (not yet supported)"
            )

    return NodeIncrementalPlan(
        factor_id=str(factor_id),
        nodes=tuple(nodes),
        stateful_node_ids=tuple(stateful_ids),
        supportable=unsupported_reason is None,
        unsupported_reason=unsupported_reason,
    )


# ---------------------------------------------------------------------------
# R44: 内部 stateful 节点的最小下游 pandas 解释器。
# 只支持明确列出的算子；任何其它算子返回 None（调用方回退 full replay）。
# 正确性优先于覆盖度 —— 回退路径是正确性参考。
# ---------------------------------------------------------------------------
_DOWNSTREAM_BINARY = frozenset({"add", "subtract", "multiply", "divide"})
_DOWNSTREAM_UNARY = frozenset({"neg", "abs", "log"})
_DOWNSTREAM_TS = frozenset({"ts_delta", "ts_delay"})
_DOWNSTREAM_RANK = frozenset({"rank", "cs_rank_01"})


def _eval_downstream_node(
    node: Any, child_frames: list[pd.DataFrame]
) -> pd.DataFrame | None:
    """R44：用最小 pandas 解释器求值一个下游节点。

    ``child_frames`` 是已求值的子节点结果列表（与 ``node.inputs`` 位置对应，
    跳过 column/literal 叶子）。返回 ``None`` 表示该算子不在受支持集合内
    （调用方回退 full replay）。
    """
    op = str(getattr(node, "op", ""))
    if op in _DOWNSTREAM_BINARY:
        if len(child_frames) < 2:
            return None
        a, b = child_frames[0], child_frames[1]
        if op == "add":
            return a + b
        if op == "subtract":
            return a - b
        if op == "multiply":
            return a * b
        if op == "divide":
            with np.errstate(divide="ignore", invalid="ignore"):
                return a / b
        return None
    if op in _DOWNSTREAM_UNARY:
        if len(child_frames) != 1:
            return None
        a = child_frames[0]
        if op == "neg":
            return -a
        if op == "abs":
            return a.abs()
        if op == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(a)
        return None
    if op in _DOWNSTREAM_TS:
        if len(child_frames) != 1:
            return None
        a = child_frames[0]
        params = _node_params(node)
        n = int(params.get("n", params.get("d", params.get("lag", 1))))
        if op == "ts_delay":
            return a.shift(n)
        if op == "ts_delta":
            return a - a.shift(n)
        return None
    if op in _DOWNSTREAM_RANK:
        if len(child_frames) != 1:
            return None
        a = child_frames[0]
        # 每日期截面 rank（0-1），与 cs_rank_01 语义一致。
        valid = np.isfinite(a.to_numpy(dtype=float))
        masked = a.where(valid)
        r = masked.rank(axis=1, method="average")
        n = pd.Series(valid.sum(axis=1), index=a.index)
        denom = (n - 1).replace(0, np.nan)
        out = r.sub(1, axis=0).div(denom, axis=0)
        singleton = valid & np.broadcast_to((n <= 1).to_numpy()[:, None], out.shape)
        out = out.where(~singleton, 0.5)
        return out.where(valid, np.nan)
    return None


def _eval_downstream_dag(
    root: Any,
    stateful_node: Any,
    stateful_frame: pd.DataFrame,
    column_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """R44：post-order 求值 stateful 节点之后的下游子 DAG。

    ``stateful_node`` 是已由 stateful segment 求值的节点对象，其结果为
    ``stateful_frame``；``column_frames`` 是 ``{column_name: DataFrame}`` 的
    源列数据（供下游 ``column`` 叶子读取）。返回根节点结果；任何不受支持
    算子 → None。
    """
    def walk(node: Any) -> pd.DataFrame | None:
        if node is stateful_node:
            return stateful_frame
        if node.op == "column":
            name = str(node.attrs.get("name", ""))
            frame = column_frames.get(name)
            if frame is None:
                return None
            return frame
        child_frames: list[pd.DataFrame] = []
        for child in getattr(node, "inputs", ()) or ():
            child_result = walk(child)
            if child_result is None:
                return None
            child_frames.append(child_result)
        return _eval_downstream_node(node, child_frames)

    return walk(root)


def try_interior_stateful_incremental(
    *,
    factor_id: str,
    ir,
    source,
    store,
    start,
    end,
    bootstrap: bool,
    mode: str = "research",
    market: str = "",
    frequency: str = "",
    calendar_version: str = "",
    timezone: str = "",
    universe_membership_hash: str = "",
    price_basis_policy: str = "",
    field_contract_digests: Mapping[str, str] | None = None,
    operator_semantic_contract_digest: str = "",
    numeric_semantics_hash: str = "",
    missing_support_policy: str = "",
) -> tuple[pd.Series, dict[str, Any]] | None:
    """R44：内部 stateful 因子的 checkpoint 增量执行（ADDITIVE）。

    * 当 ``segmented_incremental_available`` 为 True（root-only stateful）时
      直接委托给既有 :func:`try_stateful_segmented_incremental` —— 对既有调用
      方零行为变化。
    * 否则，对 :func:`plan_node_level_incremental` 判定 supportable 的内部
      stateful DAG：恢复 stateful 节点的 per-instrument checkpoint，在输出窗口
      上跑 stateful segment（1-bar 重叠语义），再用最小 pandas 解释器在
      ``[output_start, output_end]`` 上求值下游节点，并重算终 bar。

    任何不受支持的下游算子 / 形状 → 返回 None（调用方回退 full-history replay，
    绝不返回错误值）。所有 checkpoint 写入经 ``store.commit_batch`` 原子提交。
    """
    if segmented_incremental_available(ir=ir):
        return try_stateful_segmented_incremental(
            factor_id=factor_id, ir=ir, source=source, store=store,
            start=start, end=end, bootstrap=bootstrap, mode=mode,
            market=market, calendar_version=calendar_version, timezone=timezone,
            universe_membership_hash=universe_membership_hash,
            price_basis_policy=price_basis_policy,
            field_contract_digests=field_contract_digests,
            operator_semantic_contract_digest=operator_semantic_contract_digest,
            numeric_semantics_hash=numeric_semantics_hash,
            missing_support_policy=missing_support_policy,
        )

    try:
        source_scope = _source_snapshot_scope(source, mode=mode)
    except SourceSnapshotIdentityUnavailableError:
        return None

    plan = plan_node_level_incremental(
        ir, factor_id=factor_id, source_scope=source_scope, market=market,
        frequency=frequency, calendar_version=calendar_version, timezone=timezone,
        universe_membership_hash=universe_membership_hash,
        price_basis_policy=price_basis_policy,
        field_contract_digests=field_contract_digests,
        operator_semantic_contract_digest=operator_semantic_contract_digest,
        numeric_semantics_hash=numeric_semantics_hash,
        missing_support_policy=missing_support_policy,
    )
    if not plan.supportable or not plan.stateful_node_ids:
        return None

    # 只支持单个 stateful 节点（composite state 未支持）。
    if len(plan.stateful_node_ids) != 1:
        return None
    stateful_node_id = plan.stateful_node_ids[0]
    stateful_plan = next(n for n in plan.nodes if n.node_id == stateful_node_id)
    canonical = stateful_plan.op
    if canonical not in SEGMENTED_EXECUTION_CANONICALS:
        return None
    if StatefulCheckpointRegistry.get(canonical) is None:
        return None

    # 定位 stateful 节点在 IR 中的位置（post-order 顺序）。
    order: list[Any] = []

    def collect(node: Any) -> None:
        for child in getattr(node, "inputs", ()) or ():
            collect(child)
        order.append(node)

    collect(ir)
    stateful_ir = None
    for node in order:
        if str(getattr(node, "op", "")) == canonical:
            stateful_ir = node
            break
    if stateful_ir is None:
        return None

    extracted = _root_series_and_params(stateful_ir, canonical)
    if extracted is None:
        return None
    input_names, params = extracted
    input_keys = _INPUT_KEYS[canonical]
    try:
        series = {name: source.load_column(name) for name in input_names}
    except Exception:
        return None
    if not series:
        return None
    anchor = series[input_names[0]].index
    if not isinstance(anchor, pd.MultiIndex):
        return None
    frames = {name: series[name].unstack(level="instrument") for name in input_names}
    reference = frames[input_names[0]]
    instruments = list(reference.columns)
    if len(reference.index) == 0:
        return None
    segment_timestamps = pd.to_datetime(reference.index, utc=True)

    input_identity = _checkpoint_input_identity(
        factor_id=factor_id, canonical=canonical, input_names=input_names,
        params=params, source_scope=source_scope, market=market,
        calendar_version=calendar_version, timezone=timezone,
        universe_membership_hash=universe_membership_hash,
        price_basis_policy=price_basis_policy,
        field_contract_digests=field_contract_digests,
        operator_semantic_contract_digest=operator_semantic_contract_digest,
        numeric_semantics_hash=numeric_semantics_hash,
        missing_support_policy=missing_support_policy,
    )

    out = np.full((len(segment_timestamps), len(instruments)), np.nan, dtype=float)
    pending: dict[str, Any] = {}
    stateful_frames: dict[str, pd.DataFrame] = {}
    for j, instrument in enumerate(instruments):
        checkpoint = None if bootstrap else store.load_latest(
            factor_id, canonical, instrument, before=start
        )
        if checkpoint is None and not bootstrap:
            return None
        starts_at_origin = bootstrap or checkpoint is None
        inputs: Mapping[str, np.ndarray] = {
            key: frames[name][instrument].to_numpy(dtype=float)
            for key, name in zip(input_keys, input_names)
        }
        try:
            if len(segment_timestamps) >= 2:
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
        except Exception:
            return None
        stateful_frames[stateful_node_id] = pd.DataFrame(
            out, index=reference.index, columns=instruments
        )

    # 求值下游子 DAG（stateful 节点之后）。
    downstream_root = None
    for node in order:
        if node is stateful_ir:
            continue
        if any(child is stateful_ir for child in getattr(node, "inputs", ()) or ()):
            downstream_root = node
            break
    if downstream_root is None:
        # 无下游节点：stateful 节点即根 —— 直接返回。
        if pending:
            try:
                store.commit_batch(factor_id, list(pending.values()))
            except Exception:
                return None
        panel = pd.DataFrame(out, index=reference.index, columns=instruments)
        result_series = panel.stack(future_stack=True)
        result_series = result_series.reindex(
            pd.MultiIndex.from_product(
                [reference.index, instruments], names=["timestamp", "instrument"]
            )
        )
        mode_info = {
            "mode": "interior_stateful",
            "bootstrap": bool(bootstrap),
            "canonical": canonical,
            "instruments": len(instruments),
            "execution_mode": mode,
        }
        return result_series, mode_info

    downstream = _eval_downstream_dag(
        downstream_root, stateful_ir, stateful_frames[stateful_node_id], frames
    )
    if downstream is None:
        return None
    if downstream.shape != (len(reference.index), len(instruments)):
        return None

    if pending:
        try:
            store.commit_batch(factor_id, list(pending.values()))
        except Exception:
            return None

    panel = downstream
    result_series = panel.stack(future_stack=True)
    result_series = result_series.reindex(
        pd.MultiIndex.from_product(
            [reference.index, instruments], names=["timestamp", "instrument"]
        )
    )
    mode_info = {
        "mode": "interior_stateful",
        "bootstrap": bool(bootstrap),
        "canonical": canonical,
        "instruments": len(instruments),
        "execution_mode": mode,
    }
    return result_series, mode_info


def shared_state_resume(
    *,
    canonical: str,
    params: Mapping[str, Any] | None,
    sources: list[Any],
    store,
    market: str = "",
    frequency: str = "",
    **identity_kwargs: Any,
) -> list[StateNodeIdentity]:
    """R44：跨因子 state 共享证明辅助（test-only）。

    对同一 canonical + params + source_scope 的多个不同因子 DAG（``sources``
    列表），计算各自的 :class:`StateNodeIdentity` —— 它们应产生**相同**的
    ``stable_key()``，证明 checkpoint 可跨因子复用。
    """
    out: list[StateNodeIdentity] = []
    for source in sources:
        try:
            source_scope = _source_snapshot_scope(source, mode="research")
        except Exception:
            source_scope = "ephemeral"
        out.append(
            compute_state_node_identity(
                canonical, params, source_scope=source_scope,
                market=market, frequency=frequency, **identity_kwargs,
            )
        )
    return out


__all__ = [
    "AxisIdentityCertificate",
    "BoundaryProbeResult",
    "NodeIncrementalMode",
    "NodeIncrementalPlan",
    "NodePlan",
    "SegmentedFallbackReason",
    "SourceSnapshotIdentityUnavailableError",
    "StateNodeIdentity",
    "StatefulSegmentedCorruptionError",
    "compute_state_node_identity",
    "plan_node_level_incremental",
    "segmented_incremental_available",
    "shared_state_resume",
    "stateful_canonicals_in_ir",
    "try_interior_stateful_incremental",
    "try_stateful_segmented_incremental",
]
