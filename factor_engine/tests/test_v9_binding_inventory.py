import json
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("operator", [None, object(), SimpleNamespace(metadata={})])
def test_empty_graph_cannot_be_reported_as_resolved(operator):
    from factor_engine.scripts.v8_binding_dependencies import binding_dependencies
    record = binding_dependencies(operator)
    assert record["coverage"] == "PARTIAL"
    assert "no-callable-entrypoints" in record["unresolved"]
    assert record["mathematical_certification"] == "NOT_CLAIMED"


def test_sql_declaration_is_observed_but_not_execution_certified():
    from factor_engine.backend.sql_pushdown.sql_registry import SqlCapableOperator
    from factor_engine.scripts.v8_binding_dependencies import binding_dependencies
    record = binding_dependencies(SqlCapableOperator("ts_sum"))
    assert "_build_sql_specs" in record["payload"]["entrypoints"]
    assert record["coverage"] == "PARTIAL"
    assert "sql-declaration-only-runtime-emitter-not-traced" in record["unresolved"]
    assert record["mathematical_certification"] == "NOT_CLAIMED"


def test_inventory_observes_research_winner_not_none(monkeypatch, tmp_path):
    from factor_engine import cleaned_operators
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.scripts import v8_binding_dependencies as audit
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    ensure_cleaned_loaded()
    monkeypatch.setattr(cleaned_operators, "load_all", lambda: None)
    monkeypatch.setattr(OperatorRegistry, "list_canonical", lambda: ["research_probe"])
    monkeypatch.setattr(OperatorRegistry, "resolve_canonical", lambda name: name)
    monkeypatch.setattr(OperatorRegistry, "backends_for", lambda name: ["pandas_numpy"])
    def get(name, backend, *, mode="production"):
        assert mode == "any", "research-only inventory was hidden by production admission"
        return SimpleNamespace(calculate=lambda x: x + 1)
    monkeypatch.setattr(OperatorRegistry, "get", get)
    path = tmp_path / "bindings.jsonl"
    monkeypatch.setattr(sys, "argv", ["audit", "--canonical", "research_probe", "--output", str(path)])
    audit.main()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["canonical"] == "research_probe"
    assert rows[0]["payload"]["entrypoints"]
    assert rows[0]["mathematical_certification"] == "NOT_CLAIMED"


def test_advertised_missing_binding_cannot_be_certified(monkeypatch, tmp_path):
    from factor_engine import cleaned_operators
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.scripts import v8_binding_dependencies as audit
    monkeypatch.setattr(cleaned_operators, "load_all", lambda: None)
    monkeypatch.setattr(OperatorRegistry, "list_canonical", lambda: ["missing_probe"])
    monkeypatch.setattr(OperatorRegistry, "backends_for", lambda name: ["polars"])
    monkeypatch.setattr(OperatorRegistry, "get", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--all-canonicals", "--output", str(tmp_path / "missing.jsonl")])
    with pytest.raises(RuntimeError, match="advertised binding unavailable"):
        audit.main()
