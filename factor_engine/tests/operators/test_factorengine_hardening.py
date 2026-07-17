from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_policy import infer_operator_policy
from cleaned_operators.operator_surface import classify_canonical
from planner.logical_plan import PlanNode


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_backend_registration_preserves_governance_metadata() -> None:
    class Meta:
        name = "_catalog_merge_probe"
        description = "probe"
        param_names = ["x"]

    class Op:
        metadata = Meta()

    OperatorRegistry.register(Op(), canonical=Meta.name, backend="pandas_numpy", source="test")
    OperatorRegistry._catalog[Meta.name].update(
        surface="daily", pit_safe=True, scope="elementwise", checkpoint_contract={"version": 1}
    )
    OperatorRegistry.register(Op(), canonical=Meta.name, backend="sql", source="test-sql")
    entry = OperatorRegistry._catalog[Meta.name]
    assert entry["surface"] == "daily"
    assert entry["pit_safe"] is True
    assert entry["scope"] == "elementwise"
    assert entry["checkpoint_contract"] == {"version": 1}
    OperatorRegistry.unregister(Meta.name)


def test_tanh_native_polars_edges_and_pit() -> None:
    pl = pytest.importorskip("polars")
    values = [None, np.nan, -np.inf, -1e6, -1.0, 0.0, 1.0, 1e6, np.inf]
    pdf = pd.DataFrame({"x": values})
    expected = OperatorRegistry.get("tanh", "pandas_numpy").calculate(pdf)
    actual = OperatorRegistry.get("tanh", "polars").calculate(pl.DataFrame({"x": values}))["x"].to_numpy()
    np.testing.assert_allclose(actual, expected["x"].to_numpy(), equal_nan=True)
    assert infer_operator_policy(OperatorRegistry.get("tanh"), canonical="tanh").pit_safe


def test_inverse_polars_does_not_use_pandas_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    pl = pytest.importorskip("polars")
    import cleaned_operators.common._polars_bridge as bridge

    monkeypatch.setattr(bridge, "to_pandas_panel", lambda *_: (_ for _ in ()).throw(AssertionError("bridge")))
    out = OperatorRegistry.get("inverse", "polars").calculate(
        pl.DataFrame({"x": [2.0, 0.0, None, -4.0]})
    )["x"].to_list()
    assert out == [0.5, None, None, -0.25]


def test_no_registered_polars_backend_uses_pandas_bridge() -> None:
    import inspect

    offenders = []
    for canonical, implementations in OperatorRegistry._operators.items():
        operator = implementations.get("polars")
        if operator is None:
            continue
        source = inspect.getsource(operator.__class__)
        if any(token in source for token in ("to_pandas", "bridge_pandas", "bridge_registry", "from_pandas_panel")):
            offenders.append(canonical)
    assert offenders == []


def test_surface_and_alias_hardening() -> None:
    assert classify_canonical("tanh") == "daily"
    assert classify_canonical("ceil") == "extended"
    assert classify_canonical("identity") == "internal"
    assert OperatorRegistry._aliases["reverse"] == "neg"
    assert "industry_size_neutralize" not in OperatorRegistry._aliases
    assert "size_industry_neutralize" not in OperatorRegistry._aliases


def test_legacy_formula_compat_does_not_widen_daily_authoring() -> None:
    from api.dsl_parser import DSLParseError, parse_expr

    with pytest.raises(DSLParseError, match="flex_max"):
        parse_expr("flex_max(col('x'), col('y'))")
    assert parse_expr("flex_max(col('x'), col('y'))", surface="compat") is not None


def test_daily_scope_is_never_unknown() -> None:
    unknown = []
    for canonical in OperatorRegistry.list_canonical():
        if classify_canonical(canonical) != "daily":
            continue
        operator = OperatorRegistry.get(canonical) or OperatorRegistry.get(canonical, "polars")
        if operator is not None and infer_operator_policy(operator, canonical=canonical).scope == "unknown":
            unknown.append(canonical)
    assert unknown == []


def test_daily_numeric_contract_metadata_is_complete() -> None:
    incomplete = []
    for canonical in OperatorRegistry.list_canonical():
        if classify_canonical(canonical) != "daily":
            continue
        operator = OperatorRegistry.get(canonical) or OperatorRegistry.get(canonical, "polars")
        if operator is None:
            continue
        policy = infer_operator_policy(operator, canonical=canonical)
        if not all((policy.domain_policy, policy.overflow_policy, policy.null_policy, policy.version)):
            incomplete.append(canonical)
    assert incomplete == []


def test_production_sql_lowering_is_fail_closed() -> None:
    from planner.sql_lowerer import lower_to_physical_plan

    plan = PlanNode(op="tanh", inputs=[PlanNode(op="column", attrs={"name": "x"})])
    # tanh is implemented, but cannot enter production until real SQL evidence
    # promotes it into the production-safe tier.
    assert lower_to_physical_plan(plan, mode="research").fully_sql
    assert not lower_to_physical_plan(plan, mode="production").fully_sql


def test_tanh_executes_in_real_duckdb() -> None:
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(op="tanh", inputs=[PlanNode(op="column", attrs={"name": "x"})])
    compiled = compile_plan_to_sql(plan, dataset="panel", time_column="ts", instrument_column="inst")
    con = duckdb.connect()
    con.execute("create table panel(ts integer, inst varchar, x double)")
    con.execute("insert into panel values (1, 'A', NULL), (2, 'A', -1e6), (3, 'A', 0), (4, 'A', 1e6)")
    rows = con.execute(compiled.query.replace("{{panel}}", "panel")).fetchall()
    assert [row[2] for row in rows] == [None, -1.0, 0.0, 1.0]


def test_cbrt_signed_edges_triple_parity() -> None:
    pl = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    values = [None, np.nan, -np.inf, -8.0, -1.0, -0.0, 0.0, 1.0, 8.0, np.inf]
    expected = OperatorRegistry.get("cbrt").calculate(pd.DataFrame({"x": values}))["x"].to_numpy()
    actual = OperatorRegistry.get("cbrt", "polars").calculate(pl.DataFrame({"x": values}))["x"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)

    plan = PlanNode(op="cbrt", inputs=[PlanNode(op="column", attrs={"name": "x"})])
    query = compile_plan_to_sql(plan, dataset="panel", time_column="ts", instrument_column="inst").query
    con = duckdb.connect()
    con.execute("create table panel(ts integer, inst varchar, x double)")
    con.executemany("insert into panel values (?, 'A', ?)", list(enumerate(values)))
    sql = np.asarray([row[2] for row in con.execute(query.replace("{{panel}}", "panel")).fetchall()], dtype=float)
    np.testing.assert_allclose(sql, expected, equal_nan=True)


@pytest.mark.parametrize("decimals", [-2, -1, 0, 1, 2, 6])
def test_truncate_decimals_triple_parity(decimals: int) -> None:
    pl = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    values = [1.239, -1.239, 1.235, -1.235, None, np.nan, np.inf, -np.inf]
    expected = OperatorRegistry.get("truncate").calculate(
        pd.DataFrame({"x": values}), decimals=decimals
    )["x"].to_numpy()
    actual = OperatorRegistry.get("truncate", "polars").calculate(
        pl.DataFrame({"x": values}), decimals=decimals
    )["x"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)

    plan = PlanNode(
        op="truncate", inputs=[PlanNode(op="column", attrs={"name": "x"})],
        attrs={"decimals": decimals},
    )
    query = compile_plan_to_sql(plan, dataset="panel", time_column="ts", instrument_column="inst").query
    con = duckdb.connect()
    con.execute("create table panel(ts integer, inst varchar, x double)")
    con.executemany("insert into panel values (?, 'A', ?)", list(enumerate(values)))
    sql = np.asarray([row[2] for row in con.execute(query.replace("{{panel}}", "panel")).fetchall()], dtype=float)
    np.testing.assert_allclose(sql, expected, equal_nan=True)
