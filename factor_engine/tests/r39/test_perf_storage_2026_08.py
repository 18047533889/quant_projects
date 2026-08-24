# -*- coding: utf-8 -*-
"""R39 storage performance remediation tests (PERF-044/046/047/048/049/050/052/053/054/059/083/084).

Covers (task spec):
(a) PERF-048 — normal materialize never rescans the full factor history
    (``full_factor_rescan_count == 0`` and ``_count_partition_metrics`` is not
    called on the normal write path — stronger than the "scanned once" dedup).
(b) PERF-049 — ``PartitionCommitStats`` correctness (rows_added/rows_replaced/
    rows_after) on a small temp lake.
(c) PERF-046/047 — ``materialize_block(pa.Table)`` equals ``materialize(Series)``
    output on the same data (values / is_valid mask / watermark).
(d) PERF-050 — ``CatalogBatchTransaction`` performs N updates in one commit.
(e) PERF-052/083 — delta mode: two incremental writes → delta fragments +
    manifest, ``historical_rewrite_bytes == 0``, merged read correct (incl.
    legacy monolithic readable).
(f) PERF-084 — compaction produces one sorted base equal to the merged result.
(g) PERF-044 — ``WriteAmplificationTracker`` accounting.

Run serially (small RAM)::
    cd /home/shw/quant_projects && PYTHONPATH=factor_engine .venv/bin/python \
        -m pytest factor_engine/tests/r39/test_perf_storage_2026_08.py -q
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.delta_store import (
    DeltaManifest,
    compact_partition,
    read_delta_partition,
    should_compact,
)
from factor_engine.storage.materialize import ParquetMaterializer
from factor_engine.storage.partition_policy import PartitionPolicy
from factor_engine.storage.write_amplification import WriteAmplificationTracker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _series(rows: list[tuple[str, str, float]]) -> pd.Series:
    """rows: [(date_str, asset, value)] → MultiIndex(timestamp, instrument) Series."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), a) for d, a, _ in rows], names=["timestamp", "instrument"]
    )
    return pd.Series([v for _, _, v in rows], index=idx)


def _long(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: [(date_str, asset, value)] → long-table DataFrame for _upsert_partition."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([d for d, _, _ in rows]),
            "asset": [a for _, a, _ in rows],
            "value": [v for _, _, v in rows],
            "is_valid": [1] * len(rows),
            "invalid_reason": [""] * len(rows),
        }
    )


# ---------------------------------------------------------------------------
# (a) PERF-048: no full-factor rescan on the normal write path
# ---------------------------------------------------------------------------


class TestPerf048NoFullRescan:
    def test_normal_materialize_never_calls_count_partition_metrics(self, tmp_path, monkeypatch):
        real = ParquetMaterializer._count_partition_metrics
        calls = {"n": 0}

        def counting(factor_dir):
            calls["n"] += 1
            return real(factor_dir)

        monkeypatch.setattr(
            ParquetMaterializer, "_count_partition_metrics", staticmethod(counting)
        )
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize("f", _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), ast_hash="h1")

        # The normal write path derives metrics from in-memory commit stats —
        # it must never trigger a full-history value-cell re-scan.
        assert calls["n"] == 0
        assert mat.full_factor_rescan_count == 0

    def test_watermark_row_count_consistent(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize("f", _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), ast_hash="h1")
        wm = mat.catalog.get_watermark("f")
        assert wm["row_count"] == 2
        # Incremental add 2 more rows → row_count becomes 4 without a rescan.
        mat.materialize(
            "f", _series([("2024-01-17", "A", 3.0), ("2024-01-18", "A", 4.0)]), ast_hash="h1"
        )
        wm2 = mat.catalog.get_watermark("f")
        assert wm2["row_count"] == 4
        assert mat.full_factor_rescan_count == 0


# ---------------------------------------------------------------------------
# (b) PERF-049: PartitionCommitStats correctness
# ---------------------------------------------------------------------------


class TestPartitionCommitStats:
    def test_add_replace_accounting(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        policy = PartitionPolicy.from_config(partition_columns=["year"])
        factor_dir = tmp_path / "factors" / "f"

        stats1 = mat._upsert_partition(
            factor_dir, {"year": 2024}, _long([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), policy=policy
        )
        assert stats1.rows_before == 0
        assert stats1.rows_added == 2
        assert stats1.rows_replaced == 0
        assert stats1.rows_after == 2
        assert stats1.valid_after == 2
        assert stats1.partition_key == "year=2024"

        # 1 new key + 1 overwrite of an existing key.
        stats2 = mat._upsert_partition(
            factor_dir, {"year": 2024}, _long([("2024-01-16", "A", 22.0), ("2024-01-17", "A", 3.0)]), policy=policy
        )
        assert stats2.rows_before == 2
        assert stats2.rows_added == 1  # 2024-01-17
        assert stats2.rows_replaced == 1  # 2024-01-16
        assert stats2.rows_after == 3
        assert stats2.valid_after == 3

        # Overwriting an existing key with a tombstone reduces valid count.
        stats3 = mat._upsert_partition(
            factor_dir,
            {"year": 2024},
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2024-01-17"]),
                    "asset": ["A"],
                    "value": [np.nan],
                    "is_valid": [0],
                    "invalid_reason": ["inf_or_nan"],
                }
            ),
            policy=policy,
        )
        assert stats3.rows_replaced == 1
        assert stats3.valid_after == 2

    def test_stats_persisted_to_catalog(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize("f", _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), ast_hash="h1")
        stats = mat.catalog.get_partition_stats("f", "year=2024")
        assert stats is not None
        assert stats["rows"] == 2
        assert stats["valid_rows"] == 2


# ---------------------------------------------------------------------------
# (c) PERF-046/047: materialize_block(pa.Table) == materialize(Series)
# ---------------------------------------------------------------------------


class TestMaterializeBlock:
    def _make_table(self) -> pa.Table:
        return pa.table(
            {
                "datetime": pa.array(
                    pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-16", "2024-01-16"])
                ),
                "asset": pa.array(["A", "B", "A", "B"]),
                "value": pa.array([1.0, float("inf"), float("nan"), 4.0]),
            }
        )

    def test_arrow_block_matches_series_reference(self, tmp_path):
        series = _series([("2024-01-15", "A", 1.0), ("2024-01-15", "B", float("inf")),
                          ("2024-01-16", "A", float("nan")), ("2024-01-16", "B", 4.0)])
        m_ref = ParquetMaterializer(lake_root=tmp_path / "ref")
        m_blk = ParquetMaterializer(lake_root=tmp_path / "blk")

        r_ref = m_ref.materialize("f", series, ast_hash="h1", preserve_invalid_rows=True)
        r_blk = m_blk.materialize_block("f", self._make_table(), ast_hash="h1", preserve_invalid_rows=True)

        df_ref = pd.read_parquet(tmp_path / "ref" / "factors" / "f" / "year=2024" / "data.parquet")
        df_blk = pd.read_parquet(tmp_path / "blk" / "factors" / "f" / "year=2024" / "data.parquet")

        # Values / index / masks must be identical.
        assert df_ref["datetime"].equals(df_blk["datetime"])
        assert df_ref["asset"].equals(df_blk["asset"])
        assert df_ref["value"].equals(df_blk["value"])
        assert df_ref["is_valid"].equals(df_blk["is_valid"])
        assert df_ref["invalid_reason"].equals(df_blk["invalid_reason"])
        assert r_ref["rows_written"] == r_blk["rows_written"] == 4

        # Watermark identical (last_updated is a wall-clock, excluded).
        for key in ("start_date", "end_date", "row_count"):
            assert r_ref["watermark"][key] == r_blk["watermark"][key]

    def test_arrow_block_tombstone_masks(self, tmp_path):
        m = ParquetMaterializer(lake_root=tmp_path)
        m.materialize_block("f", self._make_table(), ast_hash="h1", preserve_invalid_rows=True)
        df = pd.read_parquet(tmp_path / "factors" / "f" / "year=2024" / "data.parquet")
        invalid = df.loc[df["is_valid"] == 0]
        assert set(invalid["invalid_reason"]) == {"inf_or_nan"}
        assert len(invalid) == 2  # inf + nan rows


# ---------------------------------------------------------------------------
# (d) PERF-050: CatalogBatchTransaction single durable commit
# ---------------------------------------------------------------------------


class TestCatalogBatchTransaction:
    def test_n_updates_single_commit(self, tmp_path):
        cat = FactorCatalog(tmp_path / "b.sqlite")
        for i in range(3):
            cat.register(f"f{i}", author="t", frequency="1d", ast_hash=f"h{i}")
        before = cat._conn.commit_count
        with cat.batch_transaction("gen1") as tx:
            tx.update_watermarks_many(
                [
                    (f"f{i % 3}", "2024-01-01", f"2024-01-{10 + i:02d}", 10 + i)
                    for i in range(15)
                ]
            )
            tx.update_partition_stats_many(
                [
                    {
                        "factor_id": f"f{i % 3}",
                        "partition_key": "year=2024",
                        "rows": 10 + i,
                        "valid_rows": 9,
                        "min_date": "2024-01-01",
                        "max_date": "2024-01-10",
                        "file_bytes": 100,
                    }
                    for i in range(6)
                ]
            )
        after = cat._conn.commit_count
        # 15 watermark upserts + 6 stats upserts → exactly ONE durable commit.
        assert after - before == 1
        # UPSERT semantics: the last write for a given key wins.
        # f0 is updated at i=0,3,6,9,12 → row_count 22.
        assert cat.get_watermark("f0")["row_count"] == 22
        # f1 stats are updated at i=1,4 → rows 11 then 14 (last wins).
        assert cat.get_partition_stats("f1", "year=2024")["rows"] == 14

    def test_exception_rolls_back(self, tmp_path):
        cat = FactorCatalog(tmp_path / "r.sqlite")
        cat.register("f0", author="t", frequency="1d", ast_hash="h0")
        before = cat._conn.commit_count
        with pytest.raises(RuntimeError):
            with cat.batch_transaction("gen1") as tx:
                tx.update_watermarks_many([("f0", "2024-01-01", "2024-01-10", 5)])
                raise RuntimeError("boom")
        after = cat._conn.commit_count
        assert after == before  # no commit on exception
        assert cat.get_watermark("f0") is None

    def test_register_many(self, tmp_path):
        cat = FactorCatalog(tmp_path / "reg.sqlite")
        before = cat._conn.commit_count
        with cat.batch_transaction("gen1") as tx:
            tx.register_many(
                [
                    {"factor_id": "a1", "author": "t", "frequency": "1d", "ast_hash": "ha"},
                    {"factor_id": "a2", "author": "t", "frequency": "5m", "ast_hash": "hb"},
                ]
            )
        assert cat._conn.commit_count - before == 1
        assert cat.get_factor_info("a1")["frequency"] == "1d"
        assert cat.get_factor_info("a2")["ast_hash"] == "hb"


# ---------------------------------------------------------------------------
# (e) PERF-052/083: delta mode — immutable fragments, zero historical rewrite
# ---------------------------------------------------------------------------


class TestDeltaMode:
    def test_two_incremental_writes_delta_fragments(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path, delta_mode=True)
        mat.materialize("d", _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), ast_hash="d1")
        mat.materialize("d", _series([("2024-01-17", "A", 3.0), ("2024-01-18", "A", 4.0)]), ast_hash="d1")

        part_dir = tmp_path / "factors" / "d" / "year=2024"
        manifest = DeltaManifest.load(part_dir)
        assert manifest is not None
        assert len(manifest.deltas) == 2
        assert (part_dir / "delta").exists()
        assert len(list((part_dir / "delta").glob("gen_*.parquet"))) == 2

        merged = read_delta_partition(part_dir)
        assert len(merged) == 4
        assert sorted(merged["value"].tolist()) == [1.0, 2.0, 3.0, 4.0]

        # Ordinary incremental writes must NOT rewrite historical bytes.
        snap = mat.write_amplification.snapshot()
        assert snap.historical_rewrite_bytes == 0
        assert snap.delta_file_count == 2

    def test_upsert_semantics_keep_latest(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path, delta_mode=True)
        mat.materialize("d", _series([("2024-01-17", "A", 3.0)]), ast_hash="d1")
        mat.materialize("d", _series([("2024-01-17", "A", 99.0)]), ast_hash="d1")
        merged = read_delta_partition(tmp_path / "factors" / "d" / "year=2024")
        assert len(merged) == 1
        assert float(merged["value"].iloc[0]) == pytest.approx(99.0)

    def test_legacy_monolithic_readable(self, tmp_path):
        # A factor written without delta mode must remain readable by the reader.
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize("legacy", _series([("2024-01-19", "A", 5.0)]), ast_hash="d2")
        df = read_delta_partition(tmp_path / "factors" / "legacy" / "year=2024")
        assert float(df["value"].iloc[0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# (f) PERF-084: compaction → one sorted base equal to merged result
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_compact_merges_base_and_deltas(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path, delta_mode=True)
        mat.materialize("d", _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]), ast_hash="d1")
        mat.materialize("d", _series([("2024-01-17", "A", 3.0), ("2024-01-18", "A", 4.0)]), ast_hash="d1")
        mat.materialize("d", _series([("2024-01-17", "A", 33.0)]), ast_hash="d1")

        part_dir = tmp_path / "factors" / "d" / "year=2024"
        manifest = DeltaManifest.load(part_dir)
        assert should_compact(manifest, partition_dir=part_dir, max_delta_count=3)

        merged_before = read_delta_partition(part_dir)
        result = compact_partition(part_dir)
        assert result["compacted"] is True

        merged_after = read_delta_partition(part_dir)
        # Compacted base equals merged result, sorted by [asset, datetime].
        assert len(merged_after) == len(merged_before) == 4
        assert merged_after["value"].tolist() == merged_before["value"].tolist()
        assert merged_after["asset"].tolist() == merged_before["asset"].tolist()
        assert merged_after["datetime"].tolist() == merged_before["datetime"].tolist()
        assert (part_dir / "base").exists()
        assert len(list((part_dir / "base").glob("*.parquet"))) == 1
        # Deltas removed after manifest flip.
        assert len(list((part_dir / "delta").glob("*.parquet"))) == 0

    def test_compact_after_overwrite_is_exact(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path, delta_mode=True)
        mat.materialize("d", _series([("2024-01-15", "A", 1.0)]), ast_hash="d1")
        mat.materialize("d", _series([("2024-01-15", "A", 77.0)]), ast_hash="d1")
        part_dir = tmp_path / "factors" / "d" / "year=2024"
        compact_partition(part_dir)
        merged = read_delta_partition(part_dir)
        assert len(merged) == 1
        assert float(merged["value"].iloc[0]) == pytest.approx(77.0)


# ---------------------------------------------------------------------------
# (g) PERF-044: WriteAmplificationTracker accounting
# ---------------------------------------------------------------------------


class TestWriteAmplificationTracker:
    def test_accounting(self):
        tracker = WriteAmplificationTracker(delta_mode=True)
        tracker.record_partition_write(
            logical_changed_bytes=100,
            physical_new_write_bytes=50,
            historical_rewrite_bytes=0,
            delta_file_count=1,
        )
        tracker.record_partition_write(
            logical_changed_bytes=200,
            physical_new_write_bytes=80,
            historical_rewrite_bytes=0,
            delta_file_count=1,
        )
        snap = tracker.snapshot()
        assert snap.write_count == 2
        assert snap.logical_changed_bytes == 300
        assert snap.physical_new_write_bytes == 130
        assert snap.historical_rewrite_bytes == 0
        assert snap.delta_file_count == 2
        assert snap.write_amplification == pytest.approx(130 / 300)

    def test_default_mode_counts_rewrite(self):
        tracker = WriteAmplificationTracker(delta_mode=False)
        tracker.record_partition_write(
            logical_changed_bytes=100,
            physical_new_write_bytes=1000,
            historical_rewrite_bytes=1000,
        )
        snap = tracker.snapshot()
        assert snap.historical_rewrite_bytes == 1000
        assert snap.rewrite_amplification == pytest.approx(10.0)
