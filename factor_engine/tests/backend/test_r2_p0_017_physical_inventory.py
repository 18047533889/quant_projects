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
        assert not incomplete.is_production_eligible()
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
        evidence_admission=lambda canonical, backend, implementation_id: True,
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
        evidence_admission=lambda canonical, backend, implementation_id: True,
    )[0]
    assert not row.admission.admitted
    assert not row.admission.policy_allows_production
    assert "canonical identity mismatch" in row.admission.reasons


def test_inventory_requires_evidence_bound_to_exact_physical_id():
    class Operator:
        _physical_spec = _spec(backend="pandas_numpy")

    class Registry:
        _catalog = {"ts_demo": {}}

        @staticmethod
        def list_canonical(): return ["ts_demo"]
        @staticmethod
        def backends_for(canonical): return ["pandas_numpy"]
        @staticmethod
        def get(canonical, backend, mode="production"): return Operator()

    expected = Operator._physical_spec.physical_implementation_id
    stale = _spec(backend="pandas_numpy", implementation_source_hash="stale").physical_implementation_id
    assert expected != stale

    def evidence(canonical, backend, implementation_id):
        return implementation_id == stale

    row = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=evidence,
    )[0]
    assert row.implementation_id == expected
    assert not row.admission.evidence_production_safe
    assert not row.admission.admitted

    legacy_row = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend: True,
    )[0]
    assert not legacy_row.admission.admitted


def test_inventory_requests_any_mode_so_default_hidden_path_is_visible():
    class Operator:
        _physical_spec = _spec(backend="pandas_numpy")

    class Registry:
        _catalog = {"ts_demo": {}}
        modes = []

        @staticmethod
        def list_canonical(): return ["ts_demo"]
        @staticmethod
        def backends_for(canonical): return ["pandas_numpy"]
        @classmethod
        def get(cls, canonical, backend, mode="production"):
            cls.modes.append(mode)
            return Operator() if mode == "any" else None

    row = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: True,
    )[0]
    assert Registry.modes == ["any"]
    assert row.admission.admitted


def test_sql_inventory_uses_distinct_per_dialect_specs_and_ids():
    duck = _spec(
        canonical="ts_sql",
        backend="duckdb_sql",
        execution_kind=ExecutionKind.DUCKDB_NATIVE_SQL,
        emitter_identity="sql:duckdb:v1",
    )
    click = _spec(
        canonical="ts_sql",
        backend="clickhouse_sql",
        execution_kind=ExecutionKind.NATIVE_EXPR,
        emitter_identity="sql:clickhouse:v1",
    )

    class Operator:
        _physical_specs = {"duckdb_sql": duck, "clickhouse_sql": click}

    class Registry:
        _catalog = {"ts_sql": {}}
        @staticmethod
        def list_canonical(): return ["ts_sql"]
        @staticmethod
        def backends_for(canonical): return ["sql"]
        @staticmethod
        def get(canonical, backend, mode="production"): return Operator()

    rows = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: (
            implementation_id == {"duckdb_sql": duck, "clickhouse_sql": click}[backend].physical_implementation_id
        ),
    )
    assert len(rows) == 2
    assert rows[0].implementation_id != rows[1].implementation_id
    assert all(row.admission.admitted for row in rows)

    Operator._physical_specs = {"duckdb_sql": duck}
    missing = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: True,
    )
    assert missing[0].admission.admitted
    assert not missing[1].admission.admitted
    assert missing[1].spec is None
