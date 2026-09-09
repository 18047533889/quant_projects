from types import SimpleNamespace

from factor_engine.scripts.v8_binding_dependencies import binding_dependencies


def helper_a(x):
    return x + 1


def helper_b(x):
    return x + 2


ACTIVE_HELPER = helper_a
UNUSED_METADATA = "unrelated"


def wrapper(x):
    return ACTIVE_HELPER(x)


def factory(scale):
    return lambda x: scale * x


def test_referenced_transitive_helper_changes_fingerprint(monkeypatch):
    operator = SimpleNamespace(calculate=wrapper)
    before = binding_dependencies(operator)
    monkeypatch.setitem(globals(), "ACTIVE_HELPER", helper_b)
    after = binding_dependencies(operator)
    assert before["dependency_hash"] != after["dependency_hash"]


def test_unused_global_does_not_invalidate_binding(monkeypatch):
    operator = SimpleNamespace(calculate=wrapper)
    before = binding_dependencies(operator)
    monkeypatch.setitem(globals(), "UNUSED_METADATA", "edited documentation")
    assert before["dependency_hash"] == binding_dependencies(operator)["dependency_hash"]


def test_factory_captured_parameter_affects_fingerprint():
    first = binding_dependencies(SimpleNamespace(calculate=factory(2)))
    second = binding_dependencies(SimpleNamespace(calculate=factory(3)))
    assert first["dependency_hash"] != second["dependency_hash"]
    assert first["mathematical_certification"] == "NOT_CLAIMED"


def test_budget_exhaustion_is_explicit_not_certified():
    record = binding_dependencies(SimpleNamespace(calculate=wrapper), max_nodes=0)
    assert record["coverage"] == "PARTIAL"
    assert "node-budget-exhausted" in record["unresolved"]


def test_recursive_dataclass_metadata_is_bounded():
    from dataclasses import dataclass

    @dataclass
    class Metadata:
        child: object = None

    metadata = Metadata()
    metadata.child = metadata
    record = binding_dependencies(SimpleNamespace(metadata=metadata))
    assert record["coverage"] == "PARTIAL"
    assert "cyclic-dataclass" in record["unresolved"]


def test_large_metadata_container_is_not_expanded():
    record = binding_dependencies(SimpleNamespace(metadata=list(range(10000))), max_values=32)
    assert record["coverage"] == "PARTIAL"
    assert "value-or-depth-budget-exhausted" in record["unresolved"]
    assert len(str(record)) < 1000
