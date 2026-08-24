# -*- coding: utf-8 -*-
"""R13 checkpoint-identity regressions (NEW-P0-26 / P0-29 / P0-30 / P0-31 / P1-32).

* NEW-P0-26 — the stateful chunk auditor rejects a NON-contiguous resume: a bar
  strictly between the checkpoint's last-covered bar and the segment start makes
  the resume impossible (fail-closed to full replay).
* NEW-P0-29 — a fingerprint-computation failure is a hard error; an empty
  fingerprint never authorises a resume (``"" == ""`` must not skip recompute).
* NEW-P0-30 — no forward-bar interpolation: the checkpoint's last-covered bar
  must be the bar immediately before the segment start (Friday -> Monday ->
  Tuesday regression).
* NEW-P0-31 — the stateful runtime reuses the registry's disassembly identity
  (``_code_payload``) instead of a sorted-consts digest; distinct kernels hash
  differently.
* NEW-P1-32 — nested checkpoint schema validation checks sub-field TYPES and
  RANGES, not just required key names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stateful_contract import (
    StateCheckpoint,
    StatefulCheckpointRegistry,
    StatefulContractError,
)


# ---------------------------------------------------------------------------
# NEW-P0-31 — implementation hash reuses the registry identity
# ---------------------------------------------------------------------------

def test_implementation_hash_does_not_collide_on_constant_swap():
    """``2*x+3`` and ``3*x+2`` share the constant set {2,3} and structurally
    identical bytecode; the old sorted-consts digest collided.  The registry's
    disassembly identity must distinguish them, and the stateful runtime must
    delegate to it (never a third hash)."""
    from factor_engine.cleaned_operators.registry import _code_payload
    from recursive_kernel import ema_segment, rsi_wilder_segment
    from stateful_runtime import _implementation_hash

    def f1(x):
        return 2 * x + 3

    def f2(x):
        return 3 * x + 2

    assert _code_payload(f1.__code__, include_names=True) != _code_payload(
        f2.__code__, include_names=True
    )
    # The runtime's implementation hash IS the registry identity for the kernel.
    assert _implementation_hash("ts_ema") == _code_payload(
        ema_segment.__code__, include_names=True
    )
    assert _implementation_hash("ts_ema") != _implementation_hash("RSI_WILDER")
    assert _implementation_hash("ts_ema") != _code_payload(
        rsi_wilder_segment.__code__, include_names=True
    )


def test_implementation_hash_is_deterministic():
    from stateful_runtime import _implementation_hash

    assert _implementation_hash("ts_ema") == _implementation_hash("ts_ema")
    assert len(_implementation_hash("ts_ema")) == 16


# ---------------------------------------------------------------------------
# NEW-P0-30 / NEW-P0-26 — chunk auditor: no gap / no forward-bar interpolation
# ---------------------------------------------------------------------------

def _ema_checkpoint(as_of: str, *, state_last: str | None = None) -> StateCheckpoint:
    return StatefulCheckpointRegistry.create_checkpoint(
        "ts_ema",
        instrument="A",
        as_of=as_of,
        state={
            "ema": {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 1},
            "last_timestamp": state_last or as_of,
        },
        input_identity={"dataset": "unit"},
    )


def test_chunk_auditor_rejects_missing_bar_between_friday_and_tuesday():
    """NEW-P0-30: checkpoint ends Friday, the next expected bar is Monday, but a
    Tuesday segment would resume directly and silently skip Monday.  The auditor
    rejects the resume (fail-closed)."""
    from factor_engine.runtime.stateful_incremental import audit_segment_continuity

    cp = _ema_checkpoint("2024-01-05T00:00:00+00:00")  # Friday
    # Monday (2024-01-08) exists in the source timeline but is NOT the segment
    # start (Tuesday, 2024-01-09) — a direct resume would interpolate over it.
    assert (
        audit_segment_continuity(
            checkpoint=cp,
            segment_start="2024-01-09T00:00:00+00:00",  # Tuesday
            source_timeline=["2024-01-05", "2024-01-08", "2024-01-09"],
        )
        is False
    )
    # A contiguous resume (Monday directly after Friday) is allowed.
    assert (
        audit_segment_continuity(
            checkpoint=cp,
            segment_start="2024-01-08T00:00:00+00:00",  # Monday
            source_timeline=["2024-01-05", "2024-01-08"],
        )
        is True
    )


def test_chunk_auditor_rejects_boundary_mismatch_when_observable():
    """The last source bar before the segment start must equal the checkpoint's
    last-covered bar — a stale checkpoint whose bar is not the immediate
    predecessor is rejected (NEW-P0-26)."""
    from factor_engine.runtime.stateful_incremental import audit_segment_continuity

    cp = _ema_checkpoint("2024-01-05T00:00:00+00:00")  # Friday
    # The observable timeline shows the bar before the segment start is Monday,
    # not Friday -> the checkpoint is not the immediate predecessor.
    assert (
        audit_segment_continuity(
            checkpoint=cp,
            segment_start="2024-01-09T00:00:00+00:00",  # Tuesday
            source_timeline=["2024-01-08", "2024-01-09"],  # Monday, Tuesday
        )
        is False
    )


def test_chunk_auditor_rejects_false_coverage_claim():
    """NEW-P0-30: the checkpoint record claims coverage up to/through Monday but
    the state's own last_timestamp is Friday — a direct Tuesday resume must be
    rejected (no forward-bar interpolation)."""
    from factor_engine.runtime.stateful_incremental import audit_segment_continuity

    cp = StateCheckpoint(
        operator="ts_ema",
        instrument="A",
        as_of="2024-01-08T00:00:00+00:00",  # claims coverage through Monday
        state_schema_version="ema_state.v2",
        semantic_version="2.0",
        input_fingerprint="x",
        state={
            "ema": {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 1},
            "last_timestamp": "2024-01-05T00:00:00+00:00",  # actually Friday
        },
    )
    assert (
        audit_segment_continuity(
            checkpoint=cp,
            segment_start="2024-01-09T00:00:00+00:00",  # Tuesday
        )
        is False
    )


class _WindowedSource:
    """DataSource stub that respects start_date/end_date (tz-aware) so the
    boundary probe can recover the Monday bar the narrowed resume window hides."""

    def __init__(self, panel: pd.DataFrame):
        self._panel = panel
        self.start_date = None
        self.end_date = None

    @staticmethod
    def _ts(value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def load_column(self, name: str) -> pd.Series:
        panel = self._panel
        if self.start_date is not None:
            panel = panel[panel.index >= self._ts(self.start_date)]
        if self.end_date is not None:
            panel = panel[panel.index <= self._ts(self.end_date)]
        stacked = panel.stack()
        stacked.index = stacked.index.set_names(["timestamp", "instrument"])
        return stacked.rename(name)


def test_segmented_incremental_gap_returns_none(tmp_path):
    """NEW-P0-30 integration: a Tuesday resume from a Friday checkpoint with a
    Monday bar in the data is marked resume-impossible (returns None -> full
    replay) instead of silently skipping Monday."""
    from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
    from factor_engine.runtime.stateful_incremental import try_stateful_segmented_incremental
    from factor_engine.ir.nodes import IRNode

    # Data: Fri(01-05) -> Mon(01-08) -> Tue(01-09).
    full = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0],
        },
        index=pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09"], utc=True
        ),
    )
    ir = IRNode(
        op="ts_ema",
        inputs=(IRNode(op="column", attrs={"name": "close"}),),
        attrs={"span": 3},
    )
    store = StatefulCheckpointStore(root=tmp_path)

    # Bootstrap over [Fri, Mon]: the persisted checkpoint's last fully-committed
    # bar is Friday (the terminal bar Mon is re-computed by the next segment).
    boot_source = _WindowedSource(full)
    boot_source.start_date = "2024-01-05"
    boot_source.end_date = "2024-01-08"
    boot = try_stateful_segmented_incremental(
        factor_id="f1", ir=ir, source=boot_source, store=store,
        start="2024-01-05", end="2024-01-08", bootstrap=True,
    )
    assert boot is not None
    loaded = store.load_latest("f1", "ts_ema", "A", before="2024-01-09")
    assert loaded is not None
    assert pd.Timestamp(loaded.as_of) == pd.Timestamp("2024-01-05T00:00:00+00:00")

    # The resume window is narrowed to Tuesday only; Monday is NOT in it, but the
    # boundary probe widens the source back to Friday..Tuesday and recovers it.
    tue_source = _WindowedSource(full)
    tue_source.start_date = "2024-01-09"
    tue_source.end_date = "2024-01-09"

    attempt = try_stateful_segmented_incremental(
        factor_id="f1",
        ir=ir,
        source=tue_source,
        store=store,
        start="2024-01-09",
        end="2024-01-09",
        bootstrap=False,
    )
    assert attempt is None  # resume impossible -> full replay, never skip Monday


def test_segmented_incremental_contiguous_resume_still_works(tmp_path):
    """A legitimate Monday-after-Friday resume must still proceed (the auditor
    only rejects real gaps / false coverage)."""
    from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
    from factor_engine.runtime.stateful_incremental import try_stateful_segmented_incremental
    from factor_engine.ir.nodes import IRNode

    full = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0],
        },
        index=pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09"], utc=True
        ),
    )
    ir = IRNode(
        op="ts_ema",
        inputs=(IRNode(op="column", attrs={"name": "close"}),),
        attrs={"span": 3},
    )
    store = StatefulCheckpointStore(root=tmp_path)

    boot_source = _WindowedSource(full)
    boot_source.start_date = "2024-01-05"
    boot_source.end_date = "2024-01-08"
    boot = try_stateful_segmented_incremental(
        factor_id="f1", ir=ir, source=boot_source, store=store,
        start="2024-01-05", end="2024-01-08", bootstrap=True,
    )
    assert boot is not None

    # Resume over [Monday, Tuesday] directly after Friday -> contiguous.
    mon_tue = _WindowedSource(full)
    mon_tue.start_date = "2024-01-08"
    mon_tue.end_date = "2024-01-09"

    attempt = try_stateful_segmented_incremental(
        factor_id="f1",
        ir=ir,
        source=mon_tue,
        store=store,
        start="2024-01-08",
        end="2024-01-09",
        bootstrap=False,
    )
    assert attempt is not None
    assert attempt[1]["bootstrap"] is False


def test_execute_segment_rejects_false_coverage_checkpoint():
    """NEW-P0-30 at the runtime boundary: a checkpoint whose as_of does not match
    the state's own last_timestamp must hard-fail, never resume."""
    from stateful_runtime import execute_stateful_segment

    # Bootstrap a genuine checkpoint ending Friday (as_of == state.last_timestamp).
    boot = execute_stateful_segment(
        "ts_ema",
        {"x": np.array([1.0])},
        timestamps=["2024-01-05"],  # Friday only
        instrument="A",
        input_identity={"dataset": "unit"},
        params={"span": 3},
        starts_at_dataset_origin=True,
    )
    assert pd.Timestamp(boot.checkpoint.as_of) == pd.Timestamp(
        "2024-01-05T00:00:00+00:00"
    )
    # Tamper the record to FALSELY claim coverage through Monday while the state's
    # own last_timestamp stays Friday; the fingerprint is unchanged so the audit
    # reaches the as_of/state consistency gate.
    cp = boot.checkpoint
    forged = StateCheckpoint(
        operator=cp.operator,
        instrument=cp.instrument,
        as_of="2024-01-08T00:00:00+00:00",  # claims Monday
        state_schema_version=cp.state_schema_version,
        semantic_version=cp.semantic_version,
        input_fingerprint=cp.input_fingerprint,
        state=cp.state,
    )
    with pytest.raises(ValueError, match="last_timestamp"):
        execute_stateful_segment(
            "ts_ema",
            {"x": np.array([3.0, 4.0])},
            timestamps=["2024-01-09", "2024-01-10"],  # Tuesday, Wednesday
            instrument="A",
            input_identity={"dataset": "unit"},
            params={"span": 3},
            checkpoint=forged,
            starts_at_dataset_origin=False,
        )


# ---------------------------------------------------------------------------
# NEW-P0-29 — fingerprint failure is a hard error, never "" == "" skip
# ---------------------------------------------------------------------------

def test_partition_fingerprint_failure_raises_not_empty():
    from factor_engine.runtime.factor_identity import partition_input_fingerprint

    class _BrokenFrame:
        columns = ["datetime", "asset", "value"]

        def __getitem__(self, key):
            return self

        @property
        def empty(self):
            return False

        def astype(self, *a, **k):
            raise RuntimeError("boom")

        def sort_values(self, *a, **k):
            raise RuntimeError("boom")

        def reset_index(self, *a, **k):
            # NEW-P0-29: the fingerprint path's first unguarded call on the
            # broken frame must fail loudly (fail-closed to full recompute) —
            # never swallowed, never degraded to a fixed degenerate hash.
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        partition_input_fingerprint(_BrokenFrame())


def test_partition_fingerprint_empty_never_matches_resume():
    from factor_engine.runtime.factor_identity import (
        checkpoint_fingerprint_matches,
        partition_input_fingerprint,
    )

    # A genuinely empty frame is still a deterministic (non-empty) fingerprint.
    empty = pd.DataFrame(
        {"datetime": pd.to_datetime([]), "asset": [], "value": []}
    )
    fp = partition_input_fingerprint(empty)
    assert fp and fp == partition_input_fingerprint(
        pd.DataFrame({"datetime": pd.to_datetime([]), "asset": [], "value": []})
    )
    # An empty partition fingerprint on EITHER side is non-matching (recompute),
    # so two missing fingerprints never authorise a resume.
    stored = {
        "identity_digest": "a",
        "source_snapshot": "s",
        "source_dependency_hash": "d",
        "partition_input_fingerprint": "",
        "run_generation": "g",
    }
    current = dict(stored)
    assert checkpoint_fingerprint_matches(stored, current) is False
    current["partition_input_fingerprint"] = fp
    assert checkpoint_fingerprint_matches(stored, current) is False
    assert checkpoint_fingerprint_matches(current, current) is True


# ---------------------------------------------------------------------------
# NEW-P1-32 — recursive typed checkpoint schema
# ---------------------------------------------------------------------------

def _raw_ema_checkpoint(ema_state: dict) -> StateCheckpoint:
    return StateCheckpoint(
        operator="ts_ema",
        instrument="A",
        as_of="2024-01-01T00:00:00+00:00",
        state_schema_version="ema_state.v2",
        semantic_version="2.0",
        input_fingerprint="x",
        state={"ema": ema_state, "last_timestamp": "2024-01-01T00:00:00+00:00"},
    )


def test_nested_checkpoint_schema_validates_subfield_range():
    """NEW-P1-32: valid_count is a non-negative integer — a negative count is
    rejected on read (the old key-name-only nested_schema did not check it)."""
    cp = _raw_ema_checkpoint(
        {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": -1}
    )
    with pytest.raises(StatefulContractError, match="valid_count"):
        StatefulCheckpointRegistry.validate(cp)


def test_nested_checkpoint_schema_validates_subfield_type():
    """NEW-P1-32: valid_count must be an integer, not a float."""
    cp = _raw_ema_checkpoint(
        {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 1.5}
    )
    with pytest.raises(StatefulContractError, match="valid_count"):
        StatefulCheckpointRegistry.validate(cp)


def test_nested_checkpoint_schema_accepts_nan_missing_marker():
    """NEW-P1-32: weighted_avg/old_wt may hold a NaN "missing" marker
    (finite=False), and a valid non-negative count passes."""
    cp = _raw_ema_checkpoint(
        {"weighted_avg": float("nan"), "old_wt": float("nan"), "valid_count": 0}
    )
    StatefulCheckpointRegistry.validate(cp)  # must not raise


def test_nested_checkpoint_schema_rejects_missing_subfield():
    cp = _raw_ema_checkpoint({"weighted_avg": 1.0, "old_wt": 1.0})
    with pytest.raises(StatefulContractError, match="valid_count"):
        StatefulCheckpointRegistry.validate(cp)
