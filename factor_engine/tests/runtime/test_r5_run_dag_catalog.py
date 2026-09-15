from __future__ import annotations

import sqlite3
import tracemalloc

import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.run_dag_catalog import (
    CatalogIdentityError,
    RunDAGCatalog,
)


def _plan(window: int = 5, column: str = "close") -> PlanNode:
    source = PlanNode("column", attrs={"name": column})
    return PlanNode("ts_mean", inputs=(source,), attrs={"window": window})


@pytest.mark.parametrize("size,pages", [(513, [512, 1]), (1025, [512, 512, 1])])
def test_cross_wave_shared_identity_and_bounded_root_pages(tmp_path, size, pages):
    path = tmp_path / "catalog.sqlite3"
    plan = _plan()
    scope = {"dataset": "prices", "snapshot_id": "s1", "universe": "cn"}
    with RunDAGCatalog(
        path, run_id="run-a", context_identity={"purpose": "research"}, create=True
    ) as catalog:
        catalog.add_roots((i, f"f{i}", plan, scope) for i in range(size))
        assert catalog.counts() == {"roots": size, "nodes": 2, "live_nodes": 2}
        summary = catalog.summary()
        assert summary["root_count"] == size
        assert summary["unique_nodes"] == 2
        assert summary["shared_nodes"] == 2
        assert summary["max_node_consumers"] == size
        assert summary["serialized_plan_bytes"] > 0

        observed = []
        cursor = -1
        while True:
            page = catalog.page_roots(after_ordinal=cursor, limit=512)
            if not page:
                break
            observed.append(len(page))
            cursor = page[-1].ordinal
        assert observed == pages

        nodes = catalog.page_subgraph(0)
        assert {node.consumer_count for node in nodes} == {size}
        for ordinal in range(size - 1):
            assert catalog.consume_root(ordinal) == ()
        before_last = catalog.page_subgraph(size - 1)
        assert {node.remaining_consumers for node in before_last} == {1}
        assert len(catalog.consume_root(size - 1)) == 2
        assert catalog.counts()["live_nodes"] == 0


def test_exact_parameters_and_data_scope_are_part_of_node_identity(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    with RunDAGCatalog(
        path, run_id="run-a", context_identity={"purpose": "research"}, create=True
    ) as catalog:
        first = catalog.add_root(
            0, "a", _plan(5), data_scope={"dataset": "prices", "snapshot": "s1"}
        )
        same = catalog.add_root(
            1, "b", _plan(5), data_scope={"dataset": "prices", "snapshot": "s1"}
        )
        other_param = catalog.add_root(
            2, "c", _plan(10), data_scope={"dataset": "prices", "snapshot": "s1"}
        )
        other_scope = catalog.add_root(
            3, "d", _plan(5), data_scope={"dataset": "prices", "snapshot": "s2"}
        )
        assert first == same
        assert other_param != first
        assert other_scope != first


def test_one_hundred_thousand_roots_are_streamed_and_paged(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    plan = PlanNode("column", attrs={"name": "close"})
    scope = {"dataset": "prices", "snapshot": "s1"}
    tracemalloc.start()
    try:
        with RunDAGCatalog(
            path, run_id="run-big", context_identity={"purpose": "research"}, create=True
        ) as catalog:
            assert catalog.add_roots(
                (i, f"f{i}", plan, scope) for i in range(100_000)
            ) == 100_000
            current, peak = tracemalloc.get_traced_memory()
            assert peak - current < 16 * 1024 * 1024
            assert catalog.counts() == {
                "roots": 100_000,
                "nodes": 1,
                "live_nodes": 1,
            }
            assert len(catalog.page_roots(limit=512)) == 512
            assert len(catalog.page_roots(after_ordinal=99_998, limit=512)) == 1
    finally:
        tracemalloc.stop()

    tables = {
        row[0]
        for row in sqlite3.connect(path).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert tables == {"catalog_meta", "nodes", "node_edges", "roots", "root_nodes"}


def test_transaction_failure_close_integrity_and_identity_isolation(tmp_path):
    path = tmp_path / "catalog.sqlite3"
    catalog = RunDAGCatalog(
        path, run_id="run-a", context_identity={"purpose": "research"}, create=True
    )
    catalog.add_root(0, "a", _plan(), data_scope={"dataset": "prices"})
    with pytest.raises(sqlite3.IntegrityError):
        catalog.add_root(0, "duplicate", _plan(10), data_scope={"dataset": "prices"})
    assert catalog.counts()["roots"] == 1
    assert catalog.integrity_check() == "ok"
    catalog.close()
    catalog.close()
    with pytest.raises(RuntimeError, match="closed"):
        catalog.counts()

    with RunDAGCatalog(
        path, run_id="run-a", context_identity={"purpose": "research"}
    ) as reopened:
        assert reopened.integrity_check() == "ok"
        assert reopened.counts()["roots"] == 1
    with pytest.raises(CatalogIdentityError):
        RunDAGCatalog(path, run_id="run-b", context_identity={"purpose": "research"})
    with pytest.raises(CatalogIdentityError):
        RunDAGCatalog(path, run_id="run-a", context_identity={"purpose": "production"})
