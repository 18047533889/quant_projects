# -*- coding: utf-8 -*-
"""R2-P0-017 physical implementation identity and inventory gates."""
from __future__ import annotations

from backend.contracts import ExecutionKind, PhysicalImplementationSpec
from backend.operator_capability import enumerate_physical_inventory


def _spec(**overrides):
    values = {
        "canonical": "ts_demo",
        "backend": "polars",
        "execution_kind": ExecutionKind.POLARS_NATIVE_EXPR,
        "implementation_source_hash": "source-sha256",
        "emitter_identity": "polars.expr:v1",
        "parameter_domain_hash": "params-sha256",
        "semantic_contract_hash": "contract-sha256",
    }
    values.update(overrides)
    return PhysicalImplementationSpec(**values)


def test_physical_id_binds_complete_identity_and_rejects_blank_names():
    spec = _spec()
    assert spec.is_production_eligible()
    assert spec.physical_implementation_id is not None
    assert spec.physical_implementation_id == _spec().physical_implementation_id
    assert spec.physical_implementation_id != _spec(kernel_identity="kernel:v2").physical_implementation_id

    for incomplete in (
        _spec(implementation_source_hash=""),
        _spec(emitter_identity="", kernel_identity=""),
        _spec(parameter_domain_hash=""),
        _spec(semantic_contract_hash=""),
    ):
        assert incomplete.physical_implementation_id is None

    for invalid_name in (_spec(canonical=" "), _spec(backend="")):
        assert not invalid_name.is_production_eligible()
        assert invalid_name.physical_implementation_id is None


def test_inventory_enumerates_selectable_paths_and_fails_closed_without_specs():
    class Explicit:
        _physical_spec = _spec()

    class Missing:
        pass

    class Registry:
        _catalog = {"ts_demo": {}, "ts_missing": {}, "ts_sql": {}}
        _operators = {
            "ts_demo": {"polars": Explicit()},
            "ts_missing": {"pandas_numpy": Missing()},
            "ts_sql": {"sql": Missing()},
        }

        @classmethod
        def list_canonical(cls):
            return list(cls._operators)

        @classmethod
        def backends_for(cls, canonical):
            return list(cls._operators[canonical])

        @classmethod
        def get(cls, canonical, backend):
            return cls._operators[canonical].get(backend)

    rows = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend: True,
    )
    assert [(row.canonical, row.backend) for row in rows] == [
        ("ts_demo", "polars"),
        ("ts_missing", "pandas_numpy"),
        ("ts_sql", "duckdb_sql"),
        ("ts_sql", "clickhouse_sql"),
    ]
    admitted = {row.canonical: row.admission.admitted for row in rows}
    assert admitted["ts_demo"] is True
    assert admitted["ts_missing"] is False
    sql_rows = [row for row in rows if row.canonical == "ts_sql"]
    assert all(not row.admission.admitted for row in sql_rows)
    assert all("missing explicit PhysicalImplementationSpec" in row.admission.reasons for row in sql_rows)


def test_inventory_rejects_identity_mismatch_and_any_admission_authority_denial():
    class Operator:
        _physical_spec = _spec(canonical="wrong", backend="pandas_numpy")

    class Registry:
        _catalog = {"ts_demo": {}}

        @staticmethod
        def list_canonical():
            return ["ts_demo"]

        @staticmethod
        def backends_for(canonical):
            return ["pandas_numpy"]

        @staticmethod
        def get(canonical, backend):
            return Operator()

    row = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: False,
        evidence_admission=lambda canonical, backend: True,
    )[0]
    assert not row.admission.admitted
    assert not row.admission.policy_allows_production
    assert "canonical identity mismatch" in row.admission.reasons
