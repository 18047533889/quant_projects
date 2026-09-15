from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_backend_registration_preserves_governance_metadata() -> None:
    from factor_engine.cleaned_operators.registry import _BOOTSTRAP_TOKEN

    class Meta:
        name = "_catalog_merge_probe"
        description = "probe"
        param_names = ["x"]

    class Op:
        metadata = Meta()

    OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
    try:
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
    finally:
        if OperatorRegistry.lifecycle() == "building":
            OperatorRegistry.finalize()
        OperatorRegistry.freeze()


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
    import factor_engine.cleaned_operators.common._polars_bridge as bridge

    monkeypatch.setattr(bridge, "to_pandas_panel", lambda *_: (_ for _ in ()).throw(AssertionError("bridge")))
    out = OperatorRegistry.get("inverse", "polars").calculate(
        pl.DataFrame({"x": [2.0, 0.0, None, -4.0]})
    )["x"].to_list()
    np.testing.assert_allclose(out, [0.5, np.nan, np.nan, -0.25], equal_nan=True)


def test_known_polars_delegates_are_classified_honestly() -> None:
    from factor_engine.backend.polars_backend_kind import canonical_polars_kind

    # Runtime branch selection matters: shared classes can contain unused
    # Pandas branches (e.g. native envelopes). Class-source token scans cannot
    # certify execution. Native routes have separate conversion-forbidden tests.
    for canonical in ("group_percentile", "ts_joint_energy_shift", "event_mark_autocorr"):
        kind = canonical_polars_kind(canonical, production_mode=True)
        assert "PANDAS_DELEGATE" in kind.name, (canonical, kind)


@pytest.mark.parametrize("canonical", ["ewm_std", "ewm_var", "ts_skew", "ts_quantile"])
def test_new_native_polars_implementations_match_pandas(canonical: str) -> None:
    pl = pytest.importorskip("polars")
    values = [1.0, 2.0, None, 4.0, 8.0, 16.0, 32.0]
    pdf = pd.DataFrame({"x": values})
    pldf = pl.DataFrame({"x": values})
    kwargs = {"span": 4} if canonical.startswith("ewm_") else {"window": 4}
    if canonical == "ts_quantile":
        kwargs["q"] = 0.35
    expected = OperatorRegistry.get(canonical).calculate(pdf, **kwargs)["x"].to_numpy()
    actual = OperatorRegistry.get(canonical, "polars").calculate(pldf, **kwargs)["x"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_surface_and_alias_hardening() -> None:
    assert classify_canonical("tanh") == "daily"
    assert classify_canonical("ceil") == "daily"
    assert classify_canonical("identity") == "internal"
    assert OperatorRegistry._aliases["reverse"] == "neg"
    assert classify_canonical("size_neutralize") == "daily"
    assert classify_canonical("industry_size_neutralize") == "daily"
    assert OperatorRegistry._aliases.get("size_industry_neutralize") == "industry_size_neutralize"
    assert OperatorRegistry._aliases.get("market_cap_neutralize") == "size_neutralize"
    assert OperatorRegistry._aliases.get("industry_neutralize") == "group_neutralize"


def test_legacy_formula_compat_does_not_widen_daily_authoring() -> None:
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr

    # flex_max / ts_transition_count / group_decay_linear were migrated to the
    # daily surface in the 2026-08 daily production passes; use a still-extended
    # operator (rolling_beta_to_market migration stub) to keep the gate's intent:
    # legacy/extended authoring must not widen the strict daily surface.
    with pytest.raises(DSLParseError, match="rolling_beta_to_market"):
        parse_expr("rolling_beta_to_market(col('x'), col('y'), 3)")
    assert parse_expr("rolling_beta_to_market(col('x'), col('y'), 3)", surface="compat") is not None
    # group_decay_linear is now a daily production target and parses on daily.
    assert parse_expr("group_decay_linear(col('x'), col('y'), 3)") is not None


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


def test_production_sql_lowering_is_fail_closed(monkeypatch) -> None:
    from factor_engine.planner.sql_lowerer import lower_to_physical_plan
    from factor_engine.backend.sql_pushdown import sql_registry

    plan = PlanNode(op="tanh", inputs=[PlanNode(op="column", attrs={"name": "x"})])
    assert lower_to_physical_plan(plan, mode="research").fully_sql
    # Real SQL capability must not manufacture missing production evidence.
    monkeypatch.setattr(sql_registry, "is_sql_production_safe", lambda plan: False)
    assert not lower_to_physical_plan(plan, mode="production").fully_sql


def test_tanh_executes_in_real_duckdb() -> None:
    duckdb = pytest.importorskip("duckdb")
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql

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
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql

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
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql

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
