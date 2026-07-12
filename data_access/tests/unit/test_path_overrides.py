"""可选 read_root / write_root / write_dir 路径覆盖。"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def path_store(tmp_path, monkeypatch):
    src = tmp_path / "src"
    alt_src = tmp_path / "alt_src"
    stg = tmp_path / "stg"
    custom_out = tmp_path / "custom_out"
    for p in (src, alt_src, stg, custom_out):
        p.mkdir()

    pd.DataFrame(
        {"TradeDate": pd.to_datetime(["2024-01-02"]), "Symbol": ["AAA"], "Close": [1.0]}
    ).to_parquet(src / "a.parquet")
    pd.DataFrame(
        {"TradeDate": pd.to_datetime(["2024-01-02"]), "Symbol": ["BBB"], "Close": [9.0]}
    ).to_parquet(alt_src / "b.parquet")

    yaml_text = dedent(
        f"""
        prices:
          kind: static
          access_mode: published
          layout: plain
          root: {src}
          glob: "**/*.parquet"
          time_column: TradeDate
          instrument_column: Symbol
          hive_partitioning: false
          union_by_name: true
          schema:
            TradeDate: timestamp
            Symbol: string
            Close: double

        results_stg:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: {stg}/factors/{{factor_id}}
          glob_template: "year=*/*.parquet"
          partition_columns: [year]
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
        """
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("QUANT_DATASETS_YAML", str(cfg))
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "test_ns")
    monkeypatch.setenv("DATA_ACCESS_EXTRA_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    reset_store()
    store = DataAccessStore(load_registry(cfg), DuckDBEngine())
    yield store, alt_src, custom_out, stg
    reset_store()


def test_read_root_static_override(path_store):
    store, alt_src, _custom, _stg = path_store
    df = store.read_frame(
        "prices",
        columns=["Symbol", "Close"],
        read_root=str(alt_src),
    )
    assert list(df["Symbol"]) == ["BBB"]
    assert float(df["Close"].iloc[0]) == 9.0


def test_write_dir_absolute_override(path_store):
    store, _alt, custom_out, _stg = path_store
    target = custom_out / "my_factor"
    tbl = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["AAA"],
            "value": [1.5],
            "year": [2024],
        }
    )
    result = store.write_arrow(
        "results_stg",
        tbl,
        factor_id="ignored_for_path",
        write_dir=str(target),
        mode="overwrite",
        partition_by=["year"],
    )
    assert Path(result["path"]) == target.resolve() or str(result["path"]).endswith("my_factor")
    assert any(target.rglob("*.parquet"))


def test_write_root_keeps_factor_id_suffix(path_store):
    store, _alt, custom_out, _stg = path_store
    write_root = custom_out / "alt_factors"
    write_root.mkdir(exist_ok=True)
    tbl = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["AAA"],
            "value": [2.0],
            "year": [2024],
        }
    )
    result = store.write_arrow(
        "results_stg",
        tbl,
        factor_id="feat_v1",
        write_root=str(write_root),
        mode="overwrite",
        partition_by=["year"],
    )
    assert "feat_v1" in result["path"]
    assert str(write_root) in result["path"] or "alt_factors" in result["path"]
    assert any((write_root / "feat_v1").rglob("*.parquet"))


def test_compute_and_write_with_write_dir(path_store):
    store, _alt, custom_out, _stg = path_store
    out = custom_out / "computed"
    result = store.compute_and_write(
        """
        SELECT TradeDate AS datetime, Symbol AS asset, Close AS value,
               CAST(strftime(TradeDate, '%Y') AS INTEGER) AS year
        FROM {{prices}}
        """,
        read_datasets=["prices"],
        write_dataset="results_stg",
        factor_id="from_compute",
        write_dir=str(out),
        mode="overwrite",
        partition_by=["year"],
    )
    assert result["rows_computed"] >= 1
    assert any(out.rglob("*.parquet"))


def test_read_root_blocked_in_production(path_store, monkeypatch):
    store, alt_src, _c, _s = path_store
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    with pytest.raises(ValidationError, match="read_root"):
        store.read_frame("prices", columns=["Close"], read_root=str(alt_src))


def test_parametric_read_root_rewrite(path_store):
    """write 到默认 staging，再用 read_root 从别处读同结构。"""
    store, _alt, custom_out, stg = path_store
    tbl = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["ZZZ"],
            "value": [3.3],
            "year": [2024],
        }
    )
    # 先写到自定义根
    alt_root = custom_out / "read_factors"
    alt_root.mkdir(exist_ok=True)
    store.write_arrow(
        "results_stg",
        tbl,
        factor_id="rr1",
        write_root=str(alt_root),
        partition_by=["year"],
    )
    # 默认路径下没有，但 read_root 能读到
    df = store.read_frame(
        "results_stg",
        factor_id="rr1",
        columns=["asset", "value"],
        read_root=str(alt_root),
    )
    assert list(df["asset"]) == ["ZZZ"]


def test_write_dir_still_requires_factor_id(path_store):
    store, _alt, custom_out, _stg = path_store
    target = custom_out / "missing_fid"
    tbl = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["AAA"],
            "value": [1.0],
            "year": [2024],
        }
    )
    with pytest.raises(ValidationError, match="factor_id"):
        store.write_arrow(
            "results_stg",
            tbl,
            write_dir=str(target),
            mode="overwrite",
            partition_by=["year"],
        )


def test_partition_append_files_only_new(path_store):
    store, _alt, custom_out, _stg = path_store
    target = custom_out / "append_parts"
    tbl1 = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-02"]),
            "asset": ["AAA"],
            "value": [1.0],
            "year": [2024],
        }
    )
    r1 = store.write_arrow(
        "results_stg",
        tbl1,
        factor_id="ap1",
        write_dir=str(target),
        mode="overwrite",
        partition_by=["year"],
    )
    n1 = len(r1["files"])
    assert n1 >= 1
    tbl2 = pa.table(
        {
            "datetime": pd.to_datetime(["2024-01-03"]),
            "asset": ["BBB"],
            "value": [2.0],
            "year": [2024],
        }
    )
    r2 = store.write_arrow(
        "results_stg",
        tbl2,
        factor_id="ap1",
        write_dir=str(target),
        mode="append",
        partition_by=["year"],
    )
    # 本轮返回的 files 不应包含上一轮全部历史文件
    assert len(r2["files"]) == n1 or len(r2["files"]) >= 1
    assert set(r2["files"]).isdisjoint(set(r1["files"]))
    assert len(list(target.rglob("*.parquet"))) == n1 + len(r2["files"])


def test_dataset_env_root_read_override(path_store, monkeypatch):
    store, alt_src, _c, _s = path_store
    monkeypatch.setenv("DATA_ACCESS_READ_ROOT_PRICES", str(alt_src))
    df = store.read_frame("prices", columns=["Symbol", "Close"])
    assert list(df["Symbol"]) == ["BBB"]
