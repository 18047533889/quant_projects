"""store.read_uri / store.read：统一读入口单测。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _no_production(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)


def _make_store(tmp_path):
    (tmp_path / "datasets.yaml").write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "data.parquet"
  time_column: t
  instrument_column: s
""",
        encoding="utf-8",
    )
    return DataAccessStore(registry=load_registry(tmp_path / "datasets.yaml"),
                           engine=DuckDBEngine(threads=2, enable_object_cache=False))


def test_read_registered_returns_handle(tmp_path):
    pq.write_table(pa.table({"t": ["2024-01-02"], "s": ["AAPL"], "v": [1.0]}),
                   str(tmp_path / "data.parquet"))
    store = _make_store(tmp_path)
    handle = store.read("ds", columns=["t", "s", "v"], filters={"s": ["AAPL"]})
    assert handle.to_arrow().num_rows == 1


def test_read_uri_parquet_dev(tmp_path, monkeypatch):
    pq.write_table(pa.table({"t": ["2024-01-02"], "s": ["AAPL"], "v": [1.0]}),
                   str(tmp_path / "data.parquet"))
    monkeypatch.setenv("DATA_ACCESS_READ_URI_ROOTS", str(tmp_path))
    store = _make_store(tmp_path)
    handle = store.read_uri(str(tmp_path / "data.parquet"),
                            columns=["t", "s", "v"],
                            time_column="t", instrument_column="s")
    assert handle.to_arrow().num_rows == 1


def test_read_uri_arrow_engine(tmp_path, monkeypatch):
    table = pa.table({"t": ["2024-01-02"], "s": ["AAPL"], "v": [1.0]})
    writer = pa_ipc.new_file(str(tmp_path / "data.arrow"), table.schema)
    writer.write_table(table)
    writer.close()
    monkeypatch.setenv("DATA_ACCESS_READ_URI_ROOTS", str(tmp_path))
    store = _make_store(tmp_path)
    handle = store.read_uri(str(tmp_path / "data.arrow"),
                            columns=["t", "s", "v"],
                            time_column="t", instrument_column="s",
                            filters={"s": ["AAPL"]})
    assert handle.to_arrow().num_rows == 1


def test_read_uri_blocked_in_production(tmp_path, monkeypatch):
    pq.write_table(pa.table({"a": [1]}), str(tmp_path / "data.parquet"))
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    store = _make_store(tmp_path)
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        store.read_uri(str(tmp_path / "data.parquet"))


def test_read_uri_rejected_outside_whitelist(tmp_path):
    # 用注册根（tmp_path）之外的独立临时目录，确保路径越界
    import tempfile

    outside = Path(tempfile.mkdtemp())
    pq.write_table(pa.table({"a": [1]}), str(outside / "data.parquet"))
    store = _make_store(tmp_path)
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        store.read_uri(str(outside / "data.parquet"))


def test_read_result_stream(tmp_path):
    pq.write_table(pa.table({"t": ["2024-01-02", "2024-01-03"], "s": ["AAPL", "MSFT"]}),
                   str(tmp_path / "data.parquet"))
    store = _make_store(tmp_path)
    handle = store.read("ds", result="stream")
    batches = list(handle.stream())
    assert sum(b.num_rows for b in batches) == 2
