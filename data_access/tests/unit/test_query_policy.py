"""dataset query_policy 单元测试。"""

from __future__ import annotations

import pytest

from data_access.exceptions import ValidationError
from data_access.query_budget import (
    DatasetQueryPolicy,
    QueryBudget,
    merge_dataset_policy,
    parse_dataset_query_policy,
    resolve_query_budget,
)
from data_access.registry import load_registry


def test_parse_dataset_query_policy():
    policy = parse_dataset_query_policy(
        {"require_explicit_columns": True, "max_rows": 1_000},
        context="ds",
    )
    assert policy.require_explicit_columns is True
    assert policy.max_rows == 1_000


def test_parse_rejects_invalid_query_policy():
    with pytest.raises(ValidationError, match="mapping"):
        parse_dataset_query_policy([], context="ds")


def test_merge_dataset_policy_tightens_limits():
    base = QueryBudget(max_rows=10_000, require_columns=False)
    policy = DatasetQueryPolicy(require_explicit_columns=True, max_rows=1_000)
    merged = merge_dataset_policy(base, policy)
    assert merged.require_columns is True
    assert merged.max_rows == 1_000


def test_registry_loads_query_policy(tmp_path):
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        """
        tick_data:
          kind: static
          access_mode: published
          layout: plain
          root: /tmp/tick
          time_column: ts
          instrument_column: sym
          query_policy:
            require_explicit_columns: true
            max_rows: 5000
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    reg = load_registry(yaml_path)
    ds = reg.get("tick_data")
    assert ds.query_policy.require_explicit_columns is True
    assert ds.query_policy.max_rows == 5000


def test_store_rejects_select_star_for_restricted_dataset(tmp_path, monkeypatch):
    from data_access.engine import DuckDBEngine
    from data_access.store import DataAccessStore

    root = tmp_path / "tick"
    root.mkdir()
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        f"""
        tick_data:
          kind: static
          access_mode: published
          layout: plain
          root: {root}
          time_column: ts
          instrument_column: sym
          query_policy:
            require_explicit_columns: true
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    reg = load_registry(yaml_path)
    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    with pytest.raises(ValidationError, match="columns"):
        store.read_arrow("tick_data")
