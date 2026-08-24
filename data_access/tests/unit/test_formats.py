"""read/formats.py：格式适配层单测。"""
from __future__ import annotations

import pytest

from data_access.read.formats import (
    CSVAdapter,
    FormatSpec,
    TSVAdapter,
    format_adapter_for_dataset,
    get_format_adapter,
    normalize_format_name,
)


def test_normalize_format_name():
    assert normalize_format_name("parquet") == "parquet"
    assert normalize_format_name("pq") == "parquet"
    assert normalize_format_name("csv") == "csv"
    assert normalize_format_name("json") == "jsonl"
    assert normalize_format_name("ipc") == "arrow"
    with pytest.raises(Exception):
        normalize_format_name("xlsx")


def test_format_spec_parse_string_and_dict():
    assert FormatSpec.from_yaml("csv").type == "csv"
    spec = FormatSpec.from_yaml(
        {"type": "csv", "delimiter": "|", "header": True, "encoding": "utf-8"}
    )
    assert spec.type == "csv"
    assert spec.delimiter == "|"
    assert spec.header is True
    assert spec.encoding == "utf-8"


def test_parquet_adapter_builds_read_parquet():
    ad = get_format_adapter("parquet")
    sql = ad.build_from_clause("?", hive_partitioning=True, union_by_name=True)
    assert sql == "read_parquet(?, hive_partitioning=true, union_by_name=true)"
    assert ad.uses_duckdb is True


def test_csv_adapter_options():
    ad = get_format_adapter(FormatSpec.from_yaml({"type": "csv", "delimiter": "\t"}))
    sql = ad.build_from_clause("?", hive_partitioning=False, union_by_name=False)
    # 分隔符是字面 TAB 字符（SQL 字符串里 DuckDB 不解释 \t 转义）
    assert sql == "read_csv(?, delim='\t')"
    ad2 = get_format_adapter(FormatSpec.from_yaml({"type": "csv", "compression": "gzip"}))
    assert "compression='gzip'" in ad2.build_from_clause(
        "?", hive_partitioning=False, union_by_name=False
    )


def test_tsv_adapter():
    ad = get_format_adapter("tsv")
    sql = ad.build_from_clause("?", hive_partitioning=False, union_by_name=False)
    assert sql.startswith("read_csv(?")
    assert "delim=" in sql
    assert "\t" in sql


def test_jsonl_adapter():
    ad = get_format_adapter("jsonl")
    assert ad.build_from_clause(
        "?", hive_partitioning=False, union_by_name=False
    ) == "read_json_auto(?)"


def test_arrow_feather_not_duckdb():
    for fmt in ("arrow", "feather"):
        ad = get_format_adapter(fmt)
        assert ad.uses_duckdb is False
        with pytest.raises(Exception):
            ad.build_from_clause("?", False, False)


def test_default_glob():
    from data_access.read.formats import default_glob_for_format

    assert default_glob_for_format("csv") == "**/*.csv"
    assert default_glob_for_format("parquet") == "**/*.parquet"


def _make_ds():
    from data_access.registry.loader import load_registry
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    (d / "datasets.yaml").write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  format: csv
  root: {d}
  glob: "data.csv"
  time_column: t
  instrument_column: s
"""
    )
    return load_registry(d / "datasets.yaml").get("ds")


def test_format_adapter_for_dataset():
    ds = _make_ds()
    ad = format_adapter_for_dataset(ds)
    assert isinstance(ad, CSVAdapter)
    assert ds.format == "csv"
