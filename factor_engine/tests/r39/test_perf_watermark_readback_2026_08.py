# -*- coding: utf-8 -*-
"""R39-P1-PERF-051: avoid post-write ``get_watermark`` round-trip.

The materializer now returns the pending watermark the transaction just wrote
instead of issuing a second ``catalog.get_watermark`` query after the commit.
The normal write path performs ZERO ``get_watermark`` calls; only genuine
read-back scenarios (``watermark_deferred`` / legacy-partition range merge /
staging-only range merge) still read.  A module-level counter
``post_write_watermark_readback_count`` (plus ``get_``/``reset_`` helpers)
tracks genuine read-backs.

Run serially (small RAM)::
    cd /home/shw/quant_projects && PYTHONPATH=factor_engine .venv/bin/python \
        -m pytest factor_engine/tests/r39/test_perf_watermark_readback_2026_08.py -q
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import storage.catalog as _cat_module
import storage.materialize.materializer as _mat_module
from storage.materialize import ParquetMaterializer
from storage.materialize.materializer import (
    get_post_write_watermark_readback_count,
    reset_post_write_watermark_readback_count,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _series(rows: list[tuple[str, str, float]]) -> pd.Series:
    """rows: [(date_str, asset, value)] → MultiIndex(timestamp, instrument) Series."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), a) for d, a, _ in rows], names=["timestamp", "instrument"]
    )
    return pd.Series([v for _, _, v in rows], index=idx)


def _assert_semantic_equal(actual: dict, expected: dict) -> None:
    """Compare every watermark key except ``last_updated`` (a wall-clock)."""
    for key in ("factor_id", "start_date", "end_date", "row_count"):
        assert actual[key] == expected[key], f"watermark key {key!r} mismatch"


@pytest.fixture(autouse=True)
def _reset_readback_counter():
    reset_post_write_watermark_readback_count()
    yield
    reset_post_write_watermark_readback_count()


# ---------------------------------------------------------------------------
# PERF-051: normal write path — no post-write read-back
# ---------------------------------------------------------------------------


class TestNormalPathNoReadback:
    def test_normal_materialize_returns_watermark_equal_to_catalog(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        summary = mat.materialize(
            "f",
            _series([("2024-01-15", "A", 1.0), ("2024-01-16", "A", 2.0)]),
            ast_hash="h1",
        )
        wm = summary["watermark"]
        assert wm is not None
        db = mat.catalog.get_watermark("f")
        assert db is not None
        # The returned watermark carries the SAME numeric/date semantics as a
        # read-back (last_updated is a wall-clock, excluded).
        _assert_semantic_equal(wm, db)
        assert wm["row_count"] == db["row_count"] == 2
        # No post-write DB query on the normal path.
        assert get_post_write_watermark_readback_count() == 0
        assert mat.post_write_watermark_readback_count == 0

    def test_normal_path_never_calls_get_watermark(self, tmp_path, monkeypatch):
        mat = ParquetMaterializer(lake_root=tmp_path)

        def _boom(*args, **kwargs):
            raise AssertionError(
                "get_watermark must not be called on the normal write path"
            )

        monkeypatch.setattr(mat.catalog, "get_watermark", _boom)
        summary = mat.materialize(
            "nf", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1"
        )
        assert summary["watermark"] is not None
        assert get_post_write_watermark_readback_count() == 0

    def test_returned_watermark_exactly_equals_get_watermark(self, tmp_path, monkeypatch):
        """Freeze the clock in both modules so the returned ``last_updated`` is
        byte-identical to the committed row — the strongest equality form."""
        class _FrozenDateTime:
            FROZEN = _dt.datetime(2026, 8, 11, 12, 0, 0, tzinfo=_dt.timezone.utc)

            @classmethod
            def now(cls, tz=None):
                return cls.FROZEN

        monkeypatch.setattr(_cat_module, "datetime", _FrozenDateTime)
        monkeypatch.setattr(_mat_module, "datetime", _FrozenDateTime)

        mat = ParquetMaterializer(lake_root=tmp_path)
        summary = mat.materialize(
            "ex", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1"
        )
        assert summary["watermark"] == mat.catalog.get_watermark("ex")
        assert get_post_write_watermark_readback_count() == 0


# ---------------------------------------------------------------------------
# PERF-051: read-back is preserved where it is genuinely required
# ---------------------------------------------------------------------------


class TestReadbackPreserved:
    def test_deferred_path_reads_back_committed_watermark(self, tmp_path, monkeypatch):
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize(
            "df", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1"
        )
        calls = {"n": 0}
        real = mat.catalog.get_watermark

        def _counting(factor_id):
            calls["n"] += 1
            return real(factor_id)

        monkeypatch.setattr(mat.catalog, "get_watermark", _counting)
        summary = mat.materialize(
            "df", _series([("2024-06-30", "A", 2.0)]), ast_hash="h1", defer_watermark=True
        )
        # Genuine read-back: the deferred transaction did not commit the watermark.
        assert calls["n"] >= 1
        assert get_post_write_watermark_readback_count() >= 1
        assert mat.post_write_watermark_readback_count >= 1
        # The returned watermark is the OLD committed value, not the pending one.
        assert summary["watermark"]["end_date"].startswith("2024-01-15")
        # Pending carries the merged new range for the later publish commit.
        assert summary["pending_watermark"]["end_date"].startswith("2024-06-30")

    def test_commit_deferred_materialization_returns_pending_without_readback(
        self, tmp_path, monkeypatch
    ):
        mat = ParquetMaterializer(lake_root=tmp_path)
        summary = mat.materialize(
            "cd", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1", defer_watermark=True
        )
        assert summary["watermark_deferred"] is True
        assert summary["watermark"] is None  # fresh factor, nothing committed yet

        calls = {"n": 0}
        real = mat.catalog.get_watermark

        def _counting(factor_id):
            calls["n"] += 1
            return real(factor_id)

        monkeypatch.setattr(mat.catalog, "get_watermark", _counting)
        merged = mat.commit_deferred_materialization(summary)
        assert merged["dual_write_committed"] is True
        # The deferred-commit path knows the pending value it just wrote.
        assert calls["n"] == 0
        db = mat.catalog.get_watermark("cd")
        _assert_semantic_equal(merged["watermark"], db)


# ---------------------------------------------------------------------------
# PERF-051: range / batch semantics are unchanged
# ---------------------------------------------------------------------------


class TestRangeSemantics:
    def test_multi_partition_watermark_range_and_rows(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        summary = mat.materialize(
            "mp",
            _series(
                [
                    ("2024-01-15", "A", 1.0),
                    ("2024-01-16", "A", 2.0),
                    ("2025-03-01", "A", 3.0),
                ]
            ),
            ast_hash="h1",
        )
        db = mat.catalog.get_watermark("mp")
        assert summary["watermark"]["start_date"].startswith("2024-01-15")
        assert summary["watermark"]["end_date"].startswith("2025-03-01")
        assert summary["watermark"]["row_count"] == db["row_count"] == 3
        assert get_post_write_watermark_readback_count() == 0

    def test_multi_factor_batch_watermarks(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        s1 = mat.materialize("f1", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1")
        s2 = mat.materialize(
            "f2",
            _series([("2024-02-01", "B", 5.0), ("2024-02-02", "B", 6.0)]),
            ast_hash="h1",
        )
        assert s1["watermark"]["row_count"] == 1
        assert s2["watermark"]["row_count"] == 2
        assert s1["watermark"]["start_date"].startswith("2024-01-15")
        assert s2["watermark"]["end_date"].startswith("2024-02-02")
        _assert_semantic_equal(s1["watermark"], mat.catalog.get_watermark("f1"))
        _assert_semantic_equal(s2["watermark"], mat.catalog.get_watermark("f2"))
        assert get_post_write_watermark_readback_count() == 0

    def test_incremental_write_extends_range_returned_matches_catalog(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize("inc", _series([("2024-01-15", "A", 1.0)]), ast_hash="h1")
        summary = mat.materialize(
            "inc", _series([("2024-06-30", "A", 2.0)]), ast_hash="h1"
        )
        db = mat.catalog.get_watermark("inc")
        # The stats-derived range must keep the old start and extend the end.
        assert summary["watermark"]["start_date"].startswith("2024-01-15")
        assert summary["watermark"]["end_date"].startswith("2024-06-30")
        assert summary["watermark"]["row_count"] == db["row_count"] == 2
        _assert_semantic_equal(summary["watermark"], db)
        assert get_post_write_watermark_readback_count() == 0

    def test_materialize_block_also_avoids_readback(self, tmp_path, monkeypatch):
        import pyarrow as pa

        mat = ParquetMaterializer(lake_root=tmp_path)
        real = mat.catalog.get_watermark

        def _boom(*args, **kwargs):
            raise AssertionError("get_watermark must not be called on the normal path")

        monkeypatch.setattr(mat.catalog, "get_watermark", _boom)
        table = pa.table(
            {
                "datetime": pa.array(pd.to_datetime(["2024-01-15", "2024-01-16"])),
                "asset": pa.array(["A", "A"]),
                "value": pa.array([1.0, 2.0]),
            }
        )
        summary = mat.materialize_block("blk", table, ast_hash="h1")
        db = real("blk")
        _assert_semantic_equal(summary["watermark"], db)
        assert get_post_write_watermark_readback_count() == 0
