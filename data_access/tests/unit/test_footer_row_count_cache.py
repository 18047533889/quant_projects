from pathlib import Path
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.read import stats


@pytest.fixture(autouse=True)
def clear_cache():
    stats._parquet_footer_rows.cache_clear()
    yield
    stats._parquet_footer_rows.cache_clear()


def test_repeated_estimates_reuse_only_metadata_count(tmp_path, monkeypatch):
    path = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'x': [1, 2, 3]}), path)
    original = pq.read_metadata
    calls = []
    def read(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(stats.pq, 'read_metadata', read)
    assert stats.estimate_parquet_rows([path]) == 3
    assert stats.estimate_parquet_rows([path]) == 3
    assert len(calls) == 1
    assert stats._parquet_footer_rows.cache_info().maxsize == 8192


def test_rewrite_with_restored_mtime_invalidates_count(tmp_path):
    path = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'x': [1]}), path)
    before = path.stat()
    assert stats.estimate_parquet_rows([path]) == 1
    pq.write_table(pa.table({'x': list(range(30))}), path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert stats.estimate_parquet_rows([path]) == 30


def test_generation_change_during_footer_read_is_not_cached(tmp_path, monkeypatch):
    path = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'x': [1]}), path)
    original = pq.read_metadata
    def changing_read(filename):
        result = original(filename)
        pq.write_table(pa.table({'x': [1, 2, 3, 4]}), filename)
        return result
    monkeypatch.setattr(stats.pq, 'read_metadata', changing_read)
    with pytest.raises(RuntimeError, match='changed during'):
        stats.estimate_parquet_rows([path])
    assert stats._parquet_footer_rows.cache_info().currsize == 0
    monkeypatch.setattr(stats.pq, 'read_metadata', original)
    assert stats.estimate_parquet_rows([path]) == 4


def test_missing_and_corrupt_files_are_not_healthy_zero_counts(tmp_path):
    missing = tmp_path / 'missing.parquet'
    with pytest.raises(FileNotFoundError):
        stats.estimate_parquet_rows([missing])
    corrupt = tmp_path / 'corrupt.parquet'
    corrupt.write_bytes(b'not parquet')
    with pytest.raises(Exception):
        stats.estimate_parquet_rows([corrupt])
    assert stats._parquet_footer_rows.cache_info().currsize == 0
