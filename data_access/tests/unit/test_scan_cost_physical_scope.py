from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.scan_cost import estimate_scan_cost
from data_access.store import DataAccessStore


def _store(tmp_path):
    first = tmp_path / "2024-01-02.parquet"
    second = tmp_path / "2024-01-03.parquet"
    pq.write_table(pa.table({"t": ["2024-01-02"], "s": ["A"], "v": [1.0]}), first)
    pq.write_table(
        pa.table({"t": ["2024-01-03"] * 20, "s": ["B"] * 20, "v": range(20)}),
        second,
    )
    config = tmp_path / "datasets.yaml"
    config.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  schema:
    t: string
    s: string
    v: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(load_registry(config), DuckDBEngine(enable_object_cache=False)), first


def test_frozen_physical_scope_wins_over_independent_dataset_discovery(tmp_path):
    store, first = _store(tmp_path)
    full = estimate_scan_cost(store, "ds", columns=["v"])
    scoped = estimate_scan_cost(store, "ds", columns=["v"], physical_scope=[str(first)])
    assert full.estimated_rows == 21
    assert scoped.cost_basis == "physical_scope"
    assert scoped.selected_files == scoped.file_count == 1
    assert scoped.estimated_rows == 1
    assert scoped.projection_bytes < full.projection_bytes


def test_store_memory_estimate_passes_frozen_paths_to_cost(monkeypatch, tmp_path):
    store, first = _store(tmp_path)
    seen = {}

    def fake_cost(_store, _dataset, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(projection_bytes=123)

    monkeypatch.setattr("data_access.read.scan_cost.estimate_scan_cost", fake_cost)
    ds = store.get_dataset("ds")
    got = store._estimate_read_memory(
        ds,
        columns=["v"],
        src_snapshot=SimpleNamespace(total_bytes=first.stat().st_size),
        paths=[str(first)],
        time_range=("2024-01-02", "2024-01-02"),
        instrument_filter=["A"],
    )
    assert got == 123
    assert seen["physical_scope"] == [str(first)]


def test_registered_minute_datasets_prune_adj_to_requested_daily_scope(
    monkeypatch, tmp_path
):
    for directory in ("StockMinuteBar", "StockMinuteBarAdj"):
        root = tmp_path / directory
        root.mkdir()
        for day in (1, 2, 3, 6, 7, 8, 9, 10):
            pq.write_table(
                pa.table({"placeholder": [day]}),
                root / f"2025-01-{day:02d}.parquet",
            )

    monkeypatch.setenv("ASHARE_PARQUET_ROOT", str(tmp_path))
    registry = load_registry()
    store = DataAccessStore(
        registry, DuckDBEngine(enable_object_cache=False)
    )

    for name in ("ashare_stock_minute", "ashare_stock_minute_adj"):
        partition = registry.get(name).partitioning.time
        assert partition.source == "filename"
        assert partition.frequency == "daily"
        assert partition.pattern == "{date}.parquet"

    planned_paths = store._prepare_dataset_read(
        registry.get("ashare_stock_minute_adj"),
        time_range=("2025-01-02", "2025-01-08"),
        params={},
    )
    paths = store._expand_glob_paths(
        planned_paths, dataset="ashare_stock_minute_adj"
    )
    assert [path.rsplit("/", 1)[-1] for path in paths] == [
        f"2025-01-{day:02d}.parquet" for day in (2, 3, 6, 7, 8)
    ]


def test_frozen_local_scope_omits_unmatched_daily_patterns(tmp_path):
    existing = tmp_path / "2025-01-03.parquet"
    pq.write_table(pa.table({"v": [1]}), existing)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store, _ = _store(store_root)
    frozen = store._expand_glob_paths(
        [
            str(tmp_path / "**" / "2025-01-03.parquet"),
            str(tmp_path / "**" / "2025-01-04.parquet"),
        ]
    )
    assert frozen == [str(existing)]
