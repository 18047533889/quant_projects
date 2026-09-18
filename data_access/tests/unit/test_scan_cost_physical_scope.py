from types import SimpleNamespace
import datetime as dt

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.read.scan_cost import estimate_scan_cost
from data_access.read.stats import estimate_parquet_scope_cost_filtered
from data_access.store import DataAccessStore
from data_access.runtime.prepared_read import (
    DeadlineContext,
    current_deadline,
    reset_deadline_context,
)


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
    assert scoped.scan_projection_bytes == scoped.projection_bytes
    assert scoped.resident_memory_bytes is None
    assert scoped.resident_memory_basis == "unavailable_full_materialization_peak"
    assert scoped.resident_memory_is_hard_bound is False
    assert (
        scoped.cost_by_dimension["scan_projection_bytes"]
        == float(scoped.projection_bytes)
    )


def test_scope_projection_uses_footer_for_variable_width_without_registry_schema(tmp_path):
    path = tmp_path / "strings.parquet"
    pq.write_table(pa.table({"s": ["x", "variable-width-value"], "v": [1, 2]}), path)
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    store, _ = _store(registry_root)
    ds = store.get_dataset("ds")
    object.__setattr__(ds, "schema", {})
    cost = estimate_scan_cost(store, "ds", columns=["s"], physical_scope=[str(path)])
    metadata = pq.read_metadata(path)
    expected = sum(
        metadata.row_group(i).column(0).total_uncompressed_size
        for i in range(metadata.num_row_groups)
    )
    assert cost.estimated_rows == 2
    assert cost.projection_bytes == expected
    assert cost.scan_projection_bytes == expected
    # Variable-width Arrow/Pandas objects and decoder scratch can exceed the
    # footer payload; never promote this work total into a residency guarantee.
    assert cost.resident_memory_bytes is None
    assert cost.resident_memory_is_hard_bound is False
    assert cost.total_columns == 2


def test_fixed_width_projection_is_not_dictionary_encoding_size(tmp_path):
    path = tmp_path / "constant.parquet"
    pq.write_table(pa.table({"v": pa.array([7] * 1000, type=pa.int64())}), path)
    registry_root = tmp_path / "registry-fixed"
    registry_root.mkdir()
    store, _ = _store(registry_root)
    cost = estimate_scan_cost(store, "ds", columns=["v"], physical_scope=[str(path)])
    assert cost.estimated_rows == 1000
    assert cost.projection_bytes >= 1000 * 8


def test_footer_predicates_prune_only_disjoint_row_groups(tmp_path):
    path = tmp_path / "mixed.parquet"
    table = pa.table({
        "TradeDate": pa.array(
            [dt.date(2026, 4, 19)] * 2 + [dt.date(2026, 4, 20)] * 2
            + [dt.date(2026, 4, 25)] * 2,
            type=pa.date32(),
        ),
        "Symbol": ["000001.SZ"] * 6,
        "ShareholderRank": [1, 2, 1, 2, 1, 2],
        "v": pa.array(range(6), type=pa.int64()),
    })
    pq.write_table(table, path, row_group_size=2)
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="TradeDate",
        time_range=("2026-04-20", "2026-04-24"),
        instrument_column="Symbol", instrument_filter=("000001.SZ",),
        filters={"ShareholderRank": 1},
    )
    rows, projection, _, projected, rowgroups, files, compressed = got
    assert (rows, projected, rowgroups, files) == (2, 1, 1, 1)
    assert projection >= 2 * 8
    assert compressed > 0


def test_footer_time_predicate_is_closed_at_both_endpoints(tmp_path):
    path = tmp_path / "endpoints.parquet"
    pq.write_table(
        pa.table({
            "TradeDate": pa.array(
                [dt.date(2026, 4, 20), dt.date(2026, 4, 24)], type=pa.date32()
            ),
            "v": [1, 2],
        }),
        path,
        row_group_size=1,
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="TradeDate",
        time_range=("2026-04-20", "2026-04-24"),
    )
    assert got[0] == 2
    assert got[4] == 2


def test_missing_footer_statistics_retain_every_row_group(tmp_path):
    path = tmp_path / "no-stats.parquet"
    pq.write_table(
        pa.table({
            "TradeDate": pa.array(
                [dt.date(2020, 1, 1), dt.date(2030, 1, 1)], type=pa.date32()
            ),
            "v": [1, 2],
        }),
        path, row_group_size=1, write_statistics=False,
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="TradeDate",
        time_range=("2026-04-20", "2026-04-24"),
    )
    assert got[0] == 2
    assert got[4] == 2


def test_incompatible_timezone_predicate_retains_row_groups(tmp_path):
    path = tmp_path / "timezone.parquet"
    pq.write_table(
        pa.table({
            "ts": pa.array(
                [dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "v": [1],
        }),
        path,
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="ts",
        time_range=("2026-04-20T00:00:00", "2026-04-24T00:00:00"),
    )
    assert got[0] == 1
    assert got[4] == 1


def test_timestamp_date_end_uses_execution_next_day_exclusive_bound(tmp_path):
    path = tmp_path / "timestamp-day.parquet"
    pq.write_table(
        pa.table({
            "ts": pa.array([
                dt.datetime(2026, 4, 24, 12, 0),
                dt.datetime(2026, 4, 24, 23, 59, 59, 999999),
                dt.datetime(2026, 4, 25, 0, 0),
            ], type=pa.timestamp("us")),
            "v": [1, 2, 3],
        }),
        path,
        row_group_size=1,
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="ts",
        time_range=("2026-04-24", "2026-04-24"),
        time_column_is_timestamp=True,
    )
    assert got[0] == 2
    assert got[4] == 2


def test_python_date_start_matches_naive_timestamp_midnight_cast(tmp_path):
    path = tmp_path / "timestamp-date-start.parquet"
    pq.write_table(
        pa.table({
            "ts": pa.array([
                dt.datetime(2026, 4, 23, 23, 59, 59),
                dt.datetime(2026, 4, 24, 0, 0),
            ], type=pa.timestamp("us")),
            "v": [1, 2],
        }),
        path,
        row_group_size=1,
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="ts",
        time_range=(dt.date(2026, 4, 24), dt.date(2026, 4, 24)),
        time_column_is_timestamp=True,
    )
    assert got[0] == 1
    assert got[4] == 1


def test_inverted_footer_min_max_is_unknown_and_retained(tmp_path, monkeypatch):
    path = tmp_path / "inverted.parquet"
    pq.write_table(pa.table({"t": [5], "v": [1]}), path)
    monkeypatch.setattr(
        "data_access.read.stats._parquet_footer_row_group_summary",
        lambda *_args: (
            ("t", "v"),
            ((1, 8, (8, 8), ((True, 10, 1), (True, 1, 1))),),
        ),
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], filters={"t": 99}
    )
    assert got[0] == 1
    assert got[4] == 1


def test_string_filter_does_not_assume_binary_footer_order(tmp_path, monkeypatch):
    path = tmp_path / "binary-stats.parquet"
    pq.write_table(pa.table({"blob": [b"b"], "v": [1]}), path)
    monkeypatch.setattr(
        "data_access.read.stats._parquet_footer_row_group_summary",
        lambda *_args: (
            ("blob", "v"),
            ((1, 8, (1, 8), ((True, b"b", b"c"), (True, 1, 1))),),
        ),
    )
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], filters={"blob": "z"}
    )
    assert got[0] == 1
    assert got[4] == 1


def test_footer_identity_change_fails_closed(tmp_path, monkeypatch):
    from data_access.read import stats as stats_module

    path = tmp_path / "changed.parquet"
    pq.write_table(pa.table({"v": [1]}), path)
    real_identity = stats_module._footer_file_identity(path)
    changed_identity = (*real_identity[:-1], real_identity[-1] + 1)
    calls = iter((changed_identity,))
    monkeypatch.setattr(
        stats_module, "_footer_file_identity", lambda _path: next(calls)
    )
    stats_module._parquet_footer_row_group_summary.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="changed during cost estimation"):
            stats_module._parquet_footer_row_group_summary(
                str(path.absolute()), real_identity
            )
    finally:
        stats_module._parquet_footer_row_group_summary.cache_clear()


def test_string_time_statistics_are_not_used_for_lexical_pruning(tmp_path):
    path = tmp_path / "string-time.parquet"
    pq.write_table(pa.table({"t": ["2020-01-01"], "v": [1]}), path)
    got = estimate_parquet_scope_cost_filtered(
        [path], ["v"], time_column="t",
        time_range=("2026-04-20", "2026-04-24"),
    )
    assert got[0] == 1
    assert got[4] == 1


def test_store_cost_marks_footer_proven_empty_without_erasing_full_scope(tmp_path):
    registry_root = tmp_path / "registry-empty"
    registry_root.mkdir()
    store, _ = _store(registry_root)
    first = tmp_path / "date32.parquet"
    pq.write_table(
        pa.table({
            "t": pa.array([dt.date(2024, 1, 2)], type=pa.date32()),
            "s": ["A"],
            "v": [1.0],
        }),
        first,
    )
    cost = estimate_scan_cost(
        store, "ds", columns=["v"], physical_scope=[str(first)],
        time_range=("2030-01-01", "2030-01-02"),
    )
    assert cost.cost_basis == "physical_scope"
    assert cost.empty_result_proven is True
    assert cost.rowgroup_pruning_basis == "parquet_footer_min_max_closed_predicates"
    assert cost.estimated_rows == 0
    assert cost.selected_files == 0
    assert cost.selected_rowgroups == 0
    assert cost.selected_bytes == 0
    assert cost.projection_bytes == 0
    assert cost.file_count == 1
    assert cost.total_bytes == first.stat().st_size


def test_exact_empty_physical_scope_is_explicitly_proven_empty(tmp_path):
    store, _ = _store(tmp_path)
    cost = estimate_scan_cost(
        store, "ds", columns=["v"], physical_scope=[],
        time_range=("2026-04-20", "2026-04-24"),
    )
    assert cost.cost_basis == "physical_scope"
    assert cost.empty_result_proven is True
    assert cost.file_count == 0
    assert cost.total_bytes == 0
    assert cost.selected_files == 0
    assert cost.selected_rowgroups == 0
    assert cost.selected_bytes == 0
    assert cost.estimated_rows == 0
    assert cost.projection_bytes == 0


def test_store_cost_uses_execution_scope_and_carries_filters(monkeypatch, tmp_path):
    store, first = _store(tmp_path)
    seen = {}
    real = estimate_scan_cost

    def capture(_store, dataset, **kwargs):
        seen.update(kwargs)
        return real(_store, dataset, **kwargs)

    monkeypatch.setattr("data_access.read.scan_cost.estimate_scan_cost", capture)
    cost = store.estimate_scan_cost(
        "ds", columns=["v"], time_range=("2024-01-02", "2024-01-02"),
        filters={"venue": "SSE"},
    )
    assert seen["filters"] == {"venue": "SSE"}
    # Plain layout keeps both physical objects. String time statistics are not
    # proof of SQL temporal ordering, so both row groups remain in the cost.
    assert seen["physical_scope"] == [
        str(first), str(tmp_path / "2024-01-03.parquet")
    ]
    assert cost.file_count == 2
    assert cost.selected_files == 2
    assert cost.estimated_rows == 21


def test_store_cost_preserves_shorter_parent_deadline(monkeypatch, tmp_path):
    store, _ = _store(tmp_path)
    parent = DeadlineContext.start(max_elapsed_ms=100, source="parent")
    token = parent.enter()
    seen = {}

    def fake_cost(*_args, **_kwargs):
        seen["deadline"] = current_deadline()
        return SimpleNamespace(estimated_rows=1)

    monkeypatch.setattr("data_access.read.scan_cost.estimate_scan_cost", fake_cost)
    try:
        store.estimate_scan_cost("ds", columns=["v"])
    finally:
        reset_deadline_context(token)

    assert seen["deadline"] is not parent
    assert seen["deadline"].deadline_at == parent.deadline_at


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
