# -*- coding: utf-8 -*-
"""R2-P0-017 physical implementation identity and inventory gates."""
from __future__ import annotations

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.backend.operator_capability import enumerate_physical_inventory


def test_live_production_slots_have_bound_native_physical_specs():
    """Audit a bounded set of real production-selectable registry slots."""
    import pytest

    named_slots = (
        ("ts_mean", "pandas_numpy"),
        ("ts_mean", "polars"),
        ("ts_mean", "sql"),
        ("ts_std", "pandas_numpy"),
        ("ts_std", "polars"),
    )
    try:
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.backend.polars_backend_kind import (
            PolarsImplementationKind,
            canonical_polars_kind,
            canonical_polars_is_delegate,
        )

        load_all()
        selectable = [
            (canonical, backend)
            for canonical, backend in named_slots
            if OperatorRegistry.get(canonical, backend, mode="production") is not None
        ]
    except Exception as exc:
        pytest.skip(
            "live production backend inventory unavailable; refusing to infer "
            f"selectability: {type(exc).__name__}: {exc}"
        )

    if not selectable:
        pytest.skip("named registry slots are not currently production-selectable")

    expected_ids = {}
    for canonical, registry_backend in selectable:
        operator = OperatorRegistry.get(canonical, registry_backend, mode="any")
        assert operator is not None, f"selectable registry slot unavailable: {canonical}/{registry_backend}"
        physical_backends = (
            ("duckdb_sql", "clickhouse_sql")
            if registry_backend == "sql"
            else (registry_backend,)
        )
        specs = getattr(operator, "_physical_specs", {})
        for physical_backend in physical_backends:
            if registry_backend == "sql":
                spec = specs.get(physical_backend) if isinstance(specs, dict) else None
            else:
                spec = getattr(operator, "_physical_spec", None)
                if not isinstance(spec, PhysicalImplementationSpec):
                    spec = specs.get(physical_backend) if isinstance(specs, dict) else None
            assert isinstance(spec, PhysicalImplementationSpec), (
                f"selected slot lacks inspectable explicit spec: "
                f"{canonical}/{physical_backend}"
            )
            assert spec.physical_implementation_id is not None
            expected_ids[(canonical, physical_backend)] = spec.physical_implementation_id

    def exact_evidence(canonical, backend, implementation_id):
        return implementation_id == expected_ids.get((canonical, backend))

    class BoundedRegistry:
        _catalog = {canonical: OperatorRegistry._catalog.get(canonical, {}) for canonical, _ in selectable}

        @classmethod
        def list_canonical(cls):
            return sorted({canonical for canonical, _ in selectable})

        @classmethod
        def backends_for(cls, canonical):
            return sorted({backend for name, backend in selectable if name == canonical})

        @classmethod
        def get(cls, canonical, backend, mode="any"):
            return OperatorRegistry.get(canonical, backend, mode=mode)

    rows = enumerate_physical_inventory(
        registry=BoundedRegistry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=exact_evidence,
    )
    by_slot = {(row.canonical, row.backend): row for row in rows}

    for canonical, registry_backend in selectable:
        physical_backends = (
            ("duckdb_sql", "clickhouse_sql")
            if registry_backend == "sql"
            else (registry_backend,)
        )
        for physical_backend in physical_backends:
            row = by_slot[(canonical, physical_backend)]
            assert row.spec is not None
            assert row.spec.canonical == canonical
            assert row.spec.backend == physical_backend
            assert row.spec.is_production_eligible()
            assert row.implementation_id == expected_ids[(canonical, physical_backend)]
            assert row.admission.evidence_production_safe
            assert row.admission.admitted

            if registry_backend == "polars":
                kind = canonical_polars_kind(canonical, production_mode=True)
                assert not canonical_polars_is_delegate(canonical, production_mode=True)
                assert kind is not PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE
                assert row.spec.execution_kind not in {
                    ExecutionKind.DELEGATE_PYTHON,
                    ExecutionKind.DELEGATE_PANDAS,
                    ExecutionKind.POLARS_PANDAS_DELEGATE,
                }


def _spec(**overrides):
    valid_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    values = {
        "canonical": "ts_demo",
        "backend": "polars",
        "execution_kind": ExecutionKind.POLARS_NATIVE_EXPR,
        "implementation_source_hash": valid_hash,
        "emitter_identity": "polars.expr:v1",
        "parameter_domain_hash": valid_hash,
        "semantic_contract_hash": valid_hash,
        "implementation_closure_hash": valid_hash,
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
        _spec(implementation_closure_hash=""),
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


def test_inventory_does_not_retry_default_mode_after_internal_type_error():
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
            if mode == "any":
                raise TypeError("internal registry failure")
            raise AssertionError("default mode must not be queried")

    row = enumerate_physical_inventory(
        registry=Registry,
        production_surface=lambda canonical, catalog: True,
        policy_admission=lambda canonical: True,
        evidence_admission=lambda canonical, backend, implementation_id: True,
    )[0]
    assert Registry.modes == ["any"]
    assert not row.admission.admitted
    assert "selectable registry slot has no implementation" in row.admission.reasons


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
    by_backend = {row.backend: row for row in rows}
    assert set(by_backend) == {"duckdb_sql", "clickhouse_sql"}
    assert by_backend["duckdb_sql"].implementation_id == duck.physical_implementation_id
    assert by_backend["clickhouse_sql"].implementation_id == click.physical_implementation_id
    assert by_backend["duckdb_sql"].implementation_id != by_backend["clickhouse_sql"].implementation_id
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
