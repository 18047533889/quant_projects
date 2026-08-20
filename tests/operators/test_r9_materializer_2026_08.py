# -*- coding: utf-8 -*-
"""R9-P0-025 / R9-P0-026 / R9-P0-027 regression tests.

Covered review items for ``storage/materialize/materializer.py``:

- R9-P0-025  Any failed partition must raise: ``partitions_failed`` is
             adjudicated BEFORE the all-empty short-circuit, so an
             all-partitions-failed run raises ``MaterializePartitionError``
             instead of returning a ``rows_written=0`` success summary.  The
             failure path must also work when ``run_lineage`` is supplied (no
             NameError on the deferred-watermark guard).
- R9-P0-026  The committed watermark is advanced LAST.  Success-path order is:
             (1) all required partitions succeeded, (2) dependency writes
             succeeded, (3) lineage recorded + checkpoints cleared, (4) then and
             only then ``update_watermark``.  A failed run must leave the
             watermark byte-for-byte unchanged so the next incremental run
             cannot skip data.
- R9-P0-027  NaN rows are written as explicit tombstones instead of being
             ``dropna``'d.  Tombstone marker = ``value=NaN`` + ``is_valid=0`` +
             ``invalid_reason="inf_or_nan"``.  The ``[datetime, asset]``
             dedup keep="last" upsert then overwrites a stale finite value with
             the tombstone, so a valid->NaN revision is visible as NaN/invalid
             on read rather than leaving the old finite value forever.

Partition failures are injected by monkeypatching ``_upsert_partition``; no
real data access is required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storage.exceptions import MaterializePartitionError
from storage.materializer import ParquetMaterializer
from storage.partition_policy import partition_key


def _series(rows):
    """rows: [(date_str, asset, value)] -> MultiIndex(timestamp, instrument) Series.

    Dates are normalized to ``pd.Timestamp`` so the materializer's long-table
    ``datetime`` column stays a proper datetime dtype (its watermark path calls
    ``.isoformat()`` on the column min/max).
    """
    frame = pd.DataFrame(
        [(pd.Timestamp(d), a, v) for d, a, v in rows],
        columns=["timestamp", "instrument", "value"],
    )
    return frame.set_index(["timestamp", "instrument"])["value"]


@pytest.fixture
def mat(tmp_path):
    return ParquetMaterializer(lake_root=tmp_path)


def _read_partition(tmp_path, factor_id, year):
    path = tmp_path / "factors" / factor_id / f"year={year}" / "data.parquet"
    return pd.read_parquet(path)


def _lineage(run_id, factor_id):
    # Minimal valid lineage for catalog.record_run.
    return {"run_id": run_id, "factor_id": factor_id, "ast_hash": "h1"}


# ---------------------------------------------------------------------------
# R9-P0-025 / R9-P0-026 — partition-failure adjudication + watermark order
# ---------------------------------------------------------------------------


def test_partial_partition_failure_raises_and_leaves_watermark_unchanged(
    mat, tmp_path, monkeypatch
):
    """Partition 1 success, partition 2 fail, partition 3 success.

    The run must end in a failed status (MaterializePartitionError) and the
    committed watermark must remain exactly as it was before the run.
    ``run_lineage`` is supplied to exercise the failure-path lineage branch.
    """
    mat.materialize(
        factor_id="r9_partial",
        result=_series([("2024-01-15", "A", 1.0)]),
        ast_hash="h1",
    )
    wm_before = mat.catalog.get_watermark("r9_partial")
    assert wm_before is not None and wm_before["end_date"].startswith("2024-01-15")

    real = ParquetMaterializer._upsert_partition

    def _selective_fail(self, factor_dir, part_values, new_df, *, policy):
        if partition_key(part_values).startswith("year=2025"):
            raise OSError("simulated partition write failure")
        return real(self, factor_dir, part_values, new_df, policy=policy)

    monkeypatch.setattr(ParquetMaterializer, "_upsert_partition", _selective_fail)

    with pytest.raises(MaterializePartitionError):
        mat.materialize(
            factor_id="r9_partial",
            result=_series(
                [
                    ("2024-02-01", "A", 2.0),  # partition 1: success
                    ("2025-03-01", "A", 3.0),  # partition 2: fail (injected)
                    ("2026-04-01", "A", 4.0),  # partition 3: success
                ]
            ),
            ast_hash="h1",
            run_lineage=_lineage("run_partial_fail", "r9_partial"),
        )

    # Watermark must NOT have advanced past the failed run.
    assert mat.catalog.get_watermark("r9_partial") == wm_before


def test_all_partitions_failed_raises_not_normal_return(mat, monkeypatch):
    """Every partition fails -> MaterializePartitionError (never a normal return).

    R9-P0-025: the all-partitions-failed case must raise even though nothing was
    written (the all-empty short-circuit is checked only after the failure
    adjudication).
    """
    def _boom(*args, **kwargs):
        raise OSError("simulated partition write failure")

    monkeypatch.setattr(ParquetMaterializer, "_upsert_partition", _boom)

    with pytest.raises(MaterializePartitionError):
        mat.materialize(
            factor_id="r9_allfail",
            result=_series(
                [("2024-01-15", "A", 1.0), ("2025-02-15", "A", 2.0)]
            ),
            ast_hash="h1",
            run_lineage=_lineage("run_all_fail", "r9_allfail"),
        )

    # A failed run must never leave a committed watermark (no success summary).
    assert mat.catalog.get_watermark("r9_allfail") is None


# ---------------------------------------------------------------------------
# R9-P0-026 — deferred commit also advances the watermark LAST
# ---------------------------------------------------------------------------


def test_deferred_commit_records_lineage_and_checkpoints_before_watermark(
    mat, monkeypatch
):
    """commit_deferred_materialization must run lineage + checkpoint cleanup
    before advancing the watermark (watermark commits LAST)."""
    summary = mat.materialize(
        factor_id="r9_defer",
        result=_series([("2024-01-15", "A", 1.0)]),
        ast_hash="h1",
        defer_watermark=True,
        run_lineage=_lineage("run_deferred", "r9_defer"),
    )
    assert summary["watermark_deferred"] is True
    assert mat.catalog.get_watermark("r9_defer") is None  # not committed yet

    cat = mat.catalog
    calls: list[str] = []
    real_update = cat.update_watermark
    real_record = cat.record_run
    real_clear = cat.clear_partition_checkpoints

    def _spy_update(*a, **kw):
        calls.append("update_watermark")
        return real_update(*a, **kw)

    def _spy_record(*a, **kw):
        calls.append("record_run")
        return real_record(*a, **kw)

    def _spy_clear(*a, **kw):
        calls.append("clear_partition_checkpoints")
        return real_clear(*a, **kw)

    monkeypatch.setattr(cat, "update_watermark", _spy_update)
    monkeypatch.setattr(cat, "record_run", _spy_record)
    monkeypatch.setattr(cat, "clear_partition_checkpoints", _spy_clear)

    merged = mat.commit_deferred_materialization(summary)
    assert merged["dual_write_committed"] is True
    assert mat.catalog.get_watermark("r9_defer") is not None

    # lineage + checkpoint cleanup strictly before the watermark commit.
    assert calls.index("update_watermark") > calls.index("record_run")
    assert calls.index("update_watermark") > calls.index(
        "clear_partition_checkpoints"
    )


# ---------------------------------------------------------------------------
# R9-P0-027 — NaN tombstones overwrite stale finite values
# ---------------------------------------------------------------------------


def test_nan_replacement_roundtrip_overwrites_stale_value(mat, tmp_path):
    """valid->NaN revision must tombstone, not keep the stale value.

    Initial t1 A=1, t2 A=2; recompute t1 A=NaN, t2 A=3.  After materialize+read,
    t1 A must be NaN/invalid (NOT the stale 1), and t2 A must be 3.
    """
    mat.materialize(
        factor_id="r9_nan",
        result=_series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]),
        ast_hash="h1",
    )
    mat.materialize(
        factor_id="r9_nan",
        result=_series(
            [("2024-01-15", "A", np.nan), ("2024-01-16", "A", 3.0)]
        ),
        ast_hash="h1",
    )

    df = _read_partition(tmp_path, "r9_nan", 2024)
    key = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df["datetime"]), df["asset"]], names=["datetime", "asset"]
    )
    got = df.set_index(key)["value"]
    isv = df.set_index(key)["is_valid"]
    reason = df.set_index(key)["invalid_reason"]

    # t1 A is tombstoned (NaN + is_valid=0), NOT the stale 1.0.
    assert pd.isna(got.loc[(pd.Timestamp("2024-01-15"), "A")])
    assert isv.loc[(pd.Timestamp("2024-01-15"), "A")] == 0
    assert reason.loc[(pd.Timestamp("2024-01-15"), "A")] == "inf_or_nan"

    # t2 A keeps the new valid value 3.0.
    assert got.loc[(pd.Timestamp("2024-01-16"), "A")] == pytest.approx(3.0)
    assert isv.loc[(pd.Timestamp("2024-01-16"), "A")] == 1


def test_nan_tombstone_overwrites_stale_value_with_null_overwrite(mat, tmp_path):
    """R9-P0-027: an all-NaN recompute with explicit tombstone intent
    (null_overwrite=True) writes a tombstone that overwrites the stale finite
    value, and the tombstone marker (value=NaN + is_valid=0) is persisted.

    (An all-NaN run WITHOUT explicit intent is still short-circuited for
    backward compatibility with the historical ``test_empty_after_cleaning``
    behavior — the mixed valid+NaN replacement case is covered by
    ``test_nan_replacement_roundtrip_overwrites_stale_value``.)"""
    mat.materialize(
        factor_id="r9_nan_explicit",
        result=_series([("2024-01-15", "A", 1.5)]),
        ast_hash="h1",
    )
    result = mat.materialize(
        factor_id="r9_nan_explicit",
        result=_series([("2024-01-15", "A", np.nan)]),
        ast_hash="h1",
        null_overwrite=True,
    )
    # null_overwrite path reports total rows kept (valid + tombstones).
    assert result["rows_written"] == 1

    df = _read_partition(tmp_path, "r9_nan_explicit", 2024)
    assert len(df) == 1
    assert pd.isna(df["value"].iloc[0])
    assert df["is_valid"].iloc[0] == 0
