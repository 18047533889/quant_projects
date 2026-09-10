from __future__ import annotations

import math

import duckdb
import pytest
import numpy as np

from factor_engine.backend.sql_pushdown import emitter
from factor_engine.backend.sql_pushdown.emitter import (
    CompiledSql,
    SqlDialect,
    SqlTemplate,
    _SqlTemplateCache,
    _literal_binding_key,
    _template_cache_key,
)
from factor_engine.backend.sql_pushdown.plan_fixtures import column
from factor_engine.planner.logical_plan import PlanNode


def _compiled(query: str) -> CompiledSql:
    return CompiledSql(
        query=query,
        read_datasets=("source_a",),
        referenced_columns=frozenset({"close"}),
    )


def test_h14_template_bind_rejects_changed_source_axis_and_filter_context(monkeypatch):
    plan = column("close")
    key = _template_cache_key(
        plan,
        SqlDialect.DUCKDB,
        dataset="source_a",
        table=None,
        time_column="date",
        instrument_column="code",
        filt=None,
    )
    cached = _compiled("SELECT close FROM source_a")
    template = SqlTemplate(key, _literal_binding_key(plan), cached)
    sentinel = _compiled("SELECT close FROM source_b")
    monkeypatch.setattr(emitter, "compile_plan_to_sql", lambda *a, **k: sentinel)
    rebound = template.bind(
        plan,
        dataset="source_b",
        time_column="known_at",
        instrument_column="code",
        dialect=SqlDialect.DUCKDB,
    )
    assert rebound is sentinel


@pytest.mark.parametrize("query", ["x" * 1024, "汉" * 40])
def test_h17_oversize_utf8_template_is_returned_but_not_cached(query):
    cache = _SqlTemplateCache(max_entries=4, max_bytes=64)
    cache.put("shape", "binding", _compiled(query))
    assert cache.info()["entries"] == 0
    assert cache.info()["managed_bytes"] == 0


def test_h17_replacement_accounting_remains_bounded():
    cache = _SqlTemplateCache(max_entries=2, max_bytes=256)
    cache.put("shape", "one", _compiled("select 1"))
    before = cache.info()["managed_bytes"]
    cache.put("shape", "two", _compiled("select 22"))
    info = cache.info()
    assert info["entries"] == 1
    assert 0 < info["managed_bytes"] <= info["max_bytes"]
    assert info["managed_bytes"] != before


def test_h15_batch_runs_identity_preflight_before_multi_root_compilation(monkeypatch):
    def reject():
        raise emitter.SqlCompileError("emitter identity unknown")

    monkeypatch.setattr(emitter, "assert_emitter_identity_known", reject)
    with pytest.raises(emitter.SqlCompileError, match="identity unknown"):
        emitter.compile_plans_batch_to_sql(
            {"a": column("close"), "b": column("open")},
            dataset="prices",
            time_column="date",
            instrument_column="code",
        )


def _execute_layer(plan: PlanNode, columns: str, rows: list[tuple]):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE base({columns})")
    marks = ",".join("?" for _ in rows[0])
    con.executemany(f"INSERT INTO base VALUES ({marks})", rows)
    layer = emitter._compile_layer_impl(plan, dialect=SqlDialect.DUCKDB)
    assert layer is not None
    return con.execute(layer.sql + " ORDER BY ts, inst").fetchall()


@pytest.mark.parametrize(
    ("op", "ddof", "expected"),
    [
        ("cs_variance", 0, 2 / 3),
        ("cs_variance", 1, 1.0),
        ("cs_variance", 2, 2.0),
        ("cs_stddev", 0, math.sqrt(2 / 3)),
        ("cs_stddev", 1, 1.0),
        ("cs_stddev", 2, math.sqrt(2.0)),
    ],
)
def test_h09_ddof_real_duckdb(op, ddof, expected):
    plan = PlanNode(op, (column("x"),), {"ddof": ddof})
    out = _execute_layer(
        plan,
        "ts INTEGER, inst VARCHAR, x DOUBLE",
        [(1, "A", 1.0), (1, "B", 2.0), (1, "C", 3.0)],
    )
    assert out[0][2] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [-1, True, 1.5, 2.0, float("inf")])
def test_h09_rejects_invalid_ddof(bad):
    with pytest.raises(Exception, match="non-negative integer"):
        emitter._compile_layer_impl(
            PlanNode("cs_variance", (column("x"),), {"ddof": bad}),
            dialect=SqlDialect.DUCKDB,
        )


def test_h09_accepts_numpy_integer_ddof():
    layer = emitter._compile_layer_impl(
        PlanNode("cs_variance", (column("x"),), {"ddof": np.int64(2)}),
        dialect=SqlDialect.DUCKDB,
    )
    assert layer is not None
    assert "- 2.0" in layer.sql

def test_h16_period_lag_uses_ordinal_and_revision_policy_real_duckdb():
    rows = [(1, "A", 10.0, "2024Q1"), (2, "A", 11.0, "2024Q1"), (3, "A", 20.0, "2024Q2")]
    common = (column("x"), column("pid"))
    first = _execute_layer(
        PlanNode("period_lag", common, {"periods": 1, "revision_policy": "first_available"}),
        "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR", rows,
    )
    latest = _execute_layer(
        PlanNode("period_lag", common, {"periods": 1, "revision_policy": "latest_available"}),
        "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR", rows,
    )
    assert first[-1][2] == 10.0
    assert latest[-1][2] == 11.0


def test_h16_missing_and_late_periods_are_not_compressed_real_duckdb():
    plan = PlanNode("period_lag", (column("x"), column("pid")), {"periods": 1})
    missing = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR",
        [(1, "A", 10.0, "2024Q1"), (2, "A", 30.0, "2024Q3")],
    )
    late = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR",
        [(1, "A", 10.0, "2024Q1"), (2, "A", 30.0, "2024Q3"), (3, "A", 20.0, "2024Q2")],
    )
    assert missing[-1][2] is None
    assert late[-1][2] == 10.0


@pytest.mark.parametrize(
    ("op", "attrs", "expected"),
    [
        ("period_change", {"periods": 1}, 10.0),
        ("period_cagr", {"periods": 1, "periods_per_year": 4}, 15.0),
        ("yoy_by_period", {"periods": 1}, 1.0),
        ("period_average", {"periods": 2}, 15.0),
        ("ttm_from_quarterly", {"periods": 2}, 30.0),
    ],
)
def test_h16_period_derivatives_propagate_revision_policy(op, attrs, expected):
    attrs = {**attrs, "revision_policy": "first_available"}
    plan = PlanNode(op, (column("x"), column("pid")), attrs)
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR",
        [(1, "A", 10.0, "2024Q1"), (2, "A", 20.0, "2024Q2")],
    )
    assert out[-1][2] == pytest.approx(expected)


def test_h16_cumulative_derivatives_propagate_revision_policy():
    quarter = PlanNode(
        "quarter_from_cumulative",
        (column("x"), column("pid"), column("q")),
        {"revision_policy": "first_available"},
    )
    rows = [(1, "A", 10.0, "2024Q1", 1.0), (2, "A", 25.0, "2024Q2", 2.0)]
    out = _execute_layer(quarter, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR, q DOUBLE", rows)
    assert out[-1][2] == 15.0
    ttm = PlanNode(
        "ttm_from_cumulative",
        (column("x"), column("pid"), column("q")),
        {"revision_policy": "first_available"},
    )
    rows4 = rows + [(3, "A", 45.0, "2024Q3", 3.0), (4, "A", 70.0, "2024Q4", 4.0)]
    out4 = _execute_layer(ttm, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR, q DOUBLE", rows4)
    assert out4[-1][2] == 70.0


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("ts_range", [None, None, 20.0]),
        ("ts_midpoint", [None, None, 20.0]),
        ("ts_sum_abs", [None, None, 60.0]),
        ("ts_mean_abs", [None, None, 20.0]),
        ("ts_abs_max", [None, None, 30.0]),
        ("ts_first_value", [None, None, 10.0]),
        ("ts_last_value", [None, None, 30.0]),
    ],
)
def test_h35_all_candidates_enforce_min_periods_real_duckdb(op, expected):
    plan = PlanNode(op, (column("x"),), {"window": 3, "min_periods": 3})
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE",
        [(1, "A", 10.0), (2, "A", 20.0), (3, "A", 30.0)],
    )
    assert [row[2] for row in out] == expected


@pytest.mark.parametrize(("op", "tail"), [("ts_dense_rank", 3), ("ts_percent_rank", 1.0)])
def test_h34_trailing_rank_is_prefix_stable_real_duckdb(op, tail):
    plan = PlanNode(op, (column("x"),), {"window": 3, "min_periods": 1})
    prefix = [(1, "A", 10.0), (2, "A", 20.0), (3, "A", 30.0)]
    before = _execute_layer(plan, "ts INTEGER, inst VARCHAR, x DOUBLE", prefix)
    after = _execute_layer(plan, "ts INTEGER, inst VARCHAR, x DOUBLE", prefix + [(4, "A", 0.0)])
    assert [r[2] for r in before] == [r[2] for r in after[:3]]
    assert before[-1][2] == tail


def test_h24_joint_dag_compiles_once_and_executes_real_duckdb(monkeypatch):
    x = column("close")
    shared = PlanNode("ts_mean", (x,), {"window": 2, "min_periods": 2})
    roots = {
        "sum": PlanNode("add", (shared, shared), {}),
        "square": PlanNode("multiply", (shared, shared), {}),
    }
    monkeypatch.setattr(emitter, "assert_emitter_identity_known", lambda: "test")
    monkeypatch.setattr(emitter, "plan_is_sql_capable", lambda plan: True)
    calls = 0
    original = emitter._compile_layer_impl

    def counted(node, *, dialect):
        nonlocal calls
        if node.op == "ts_mean":
            calls += 1
        return original(node, dialect=dialect)

    monkeypatch.setattr(emitter, "_compile_layer_impl", counted)
    compiled = emitter.compile_plans_batch_to_sql(
        roots, dataset="prices", time_column="date", instrument_column="code"
    )
    assert compiled is not None
    assert calls == 1
    assert "s1 AS" in compiled.query
    con = duckdb.connect()
    con.execute("CREATE TABLE prices(date INTEGER, code VARCHAR, close DOUBLE)")
    con.executemany("INSERT INTO prices VALUES (?,?,?)", [(1, "A", 1.0), (2, "A", 2.0), (3, "A", 3.0)])
    out = con.execute(compiled.query.replace("{{prices}}", "prices")).fetchall()
    assert out == [(1, "A", None, None), (2, "A", 3.0, 2.25), (3, "A", 5.0, 6.25)]
    assert con.execute("EXPLAIN " + compiled.query.replace("{{prices}}", "prices")).fetchall()


def test_h09_nonfinite_is_excluded_before_duckdb_variance():
    plan = PlanNode("cs_variance", (column("x"),), {"ddof": 1})
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE",
        [(1, "A", 1.0), (1, "B", 2.0), (1, "C", float("inf"))],
    )
    assert [row[2] for row in out] == pytest.approx([0.5, 0.5, 0.5])


def test_h16_period_domain_rejects_arbitrary_numeric_and_strict_period_count():
    plan = PlanNode("period_lag", (column("x"), column("pid")), {"periods": 1})
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid BIGINT",
        [(1, "A", 10.0, 2024), (2, "A", 20.0, 20241)],
    )
    assert out[0][2] is None
    assert out[1][2] is None
    with pytest.raises(Exception):
        emitter._compile_layer_impl(
            PlanNode("period_lag", (column("x"), column("pid")), {"periods": 1.5}),
            dialect=SqlDialect.DUCKDB,
        )


def test_h16_explicit_date_and_compact_quarter_encodings_share_ordinal_contract():
    plan = PlanNode("period_lag", (column("x"), column("pid")), {"periods": 1})
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR",
        [(1, "A", 10.0, "2024Q1"), (2, "A", 20.0, "2024-06-30"), (3, "A", 30.0, "20243")],
    )
    assert [row[2] for row in out] == [None, 10.0, 20.0]


def test_h14_template_key_binds_semantics_generation_and_unambiguous_filter_context():
    left = PlanNode("column", attrs={"name": "close"}, semantic_attrs={"unit": "CNY"})
    right = PlanNode("column", attrs={"name": "close"}, semantic_attrs={"unit": "USD"})
    common = dict(
        dialect=SqlDialect.DUCKDB, dataset="source", table=None,
        time_column="date", instrument_column="code", filt=None,
    )
    assert _template_cache_key(left, **common) != _template_cache_key(right, **common)
    filt_a = emitter.SqlPushdownFilter(
        time_column="date", instrument_column="code", instruments=("A,B", "C")
    )
    filt_b = emitter.SqlPushdownFilter(
        time_column="date", instrument_column="code", instruments=("A", "B,C")
    )
    assert _template_cache_key(left, **{**common, "filt": filt_a}) != _template_cache_key(
        left, **{**common, "filt": filt_b}
    )


def test_h15_empty_batch_policy_does_not_require_execution_identity(monkeypatch):
    monkeypatch.setattr(
        emitter, "assert_emitter_identity_known",
        lambda: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert emitter.compile_plans_batch_to_sql(
        {}, dataset="prices", time_column="date", instrument_column="code"
    ) is None


def test_h24_joint_cse_does_not_merge_distinct_semantic_contexts(monkeypatch):
    left_col = PlanNode("column", attrs={"name": "close"}, semantic_attrs={"unit": "CNY"})
    right_col = PlanNode("column", attrs={"name": "close"}, semantic_attrs={"unit": "USD"})
    left = PlanNode("ts_mean", (left_col,), {"window": 2, "min_periods": 2}, {"unit": "CNY"})
    right = PlanNode("ts_mean", (right_col,), {"window": 2, "min_periods": 2}, {"unit": "USD"})
    monkeypatch.setattr(emitter, "assert_emitter_identity_known", lambda: "test")
    monkeypatch.setattr(emitter, "plan_is_sql_capable", lambda plan: True)
    calls = 0
    original = emitter._compile_layer_impl

    def counted(node, *, dialect):
        nonlocal calls
        if node.op == "ts_mean":
            calls += 1
        return original(node, dialect=dialect)

    monkeypatch.setattr(emitter, "_compile_layer_impl", counted)
    result = emitter.compile_plans_batch_to_sql(
        {"cny": left, "usd": right}, dataset="prices",
        time_column="date", instrument_column="code",
    )
    assert result is not None
    assert calls == 2


def test_h34_h35_candidates_remain_outside_production_safe_set():
    from factor_engine.backend.sql_tiers import is_sql_production_safe

    candidates = {
        "ts_dense_rank", "ts_percent_rank", "ts_range", "ts_midpoint",
        "ts_sum_abs", "ts_mean_abs", "ts_abs_max", "ts_first_value", "ts_last_value",
    }
    assert not any(is_sql_production_safe(name) for name in candidates)


@pytest.mark.parametrize(
    ("ddof", "values", "expected"),
    [
        (0, [1.0], 0.0),
        (1, [1.0], None),
        (0, [1.0, 3.0], 1.0),
        (1, [1.0, 3.0], 2.0),
        (2, [1.0, 3.0], None),
    ],
)
def test_h09_ddof_small_sample_boundary(ddof, values, expected):
    rows = [(1, chr(ord("A") + i), value) for i, value in enumerate(values)]
    out = _execute_layer(
        PlanNode("cs_variance", (column("x"),), {"ddof": ddof}),
        "ts INTEGER, inst VARCHAR, x DOUBLE", rows,
    )
    assert out[0][2] == expected


def test_h16_positional_revision_policy_reaches_nested_period_lag():
    from factor_engine.backend.sql_pushdown.plan_fixtures import literal

    plan = PlanNode(
        "period_average",
        (column("x"), column("pid"), literal(2), literal(True), literal("first_available")),
        {},
    )
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE, pid VARCHAR",
        [(1, "A", 10.0, "2024Q1"), (2, "A", 11.0, "2024Q1"), (3, "A", 20.0, "2024Q2")],
    )
    assert out[-1][2] == 15.0


@pytest.mark.parametrize(
    "op",
    ["ts_range", "ts_midpoint", "ts_sum_abs", "ts_mean_abs", "ts_abs_max", "ts_first_value", "ts_last_value"],
)
def test_h35_window_one_and_short_window_twenty(op):
    one = _execute_layer(
        PlanNode(op, (column("x"),), {"window": 1, "min_periods": 1}),
        "ts INTEGER, inst VARCHAR, x DOUBLE", [(1, "A", 2.0)],
    )
    assert one[0][2] is not None
    short = _execute_layer(
        PlanNode(op, (column("x"),), {"window": 20, "min_periods": 20}),
        "ts INTEGER, inst VARCHAR, x DOUBLE", [(1, "A", 2.0), (2, "A", 3.0)],
    )
    assert [row[2] for row in short] == [None, None]


@pytest.mark.parametrize("op", ["ts_dense_rank", "ts_percent_rank"])
def test_h34_rank_ties_missing_and_instrument_isolation(op):
    plan = PlanNode(op, (column("x"),), {"window": 3, "min_periods": 1})
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE",
        [
            (1, "A", 2.0), (2, "A", 2.0), (3, "A", None), (4, "A", 3.0),
            (1, "B", 100.0), (2, "B", 50.0),
        ],
    )
    a = [row[2] for row in out if row[1] == "A"]
    b = [row[2] for row in out if row[1] == "B"]
    if op == "ts_dense_rank":
        assert a == [1, 1, None, 2]
        assert b == [1, 1]
    else:
        assert a == [0.0, 0.0, None, 1.0]
        assert b == [0.0, 0.0]


def test_h24_joint_dag_query_growth_is_bounded_for_the_diamond_fixture(monkeypatch):
    monkeypatch.setattr(emitter, "assert_emitter_identity_known", lambda: "test")
    monkeypatch.setattr(emitter, "plan_is_sql_capable", lambda plan: True)
    x = column("close")
    shared = PlanNode("ts_mean", (x,), {"window": 2, "min_periods": 2})
    roots = {str(i): PlanNode("add", (shared, shared), {}) for i in range(8)}
    compiled = emitter.compile_plans_batch_to_sql(
        roots, dataset="prices", time_column="date", instrument_column="code"
    )
    assert compiled is not None
    assert compiled.query.count("s1 AS") == 1
    assert len(compiled.query.encode("utf-8")) < 8_000


def test_followup_alma_real_duckdb_matches_independent_gaussian_reference():
    window, offset, sigma = 3, 0.85, 6.0
    plan = PlanNode(
        "ALMA", (column("x"),),
        {"window": window, "offset": offset, "sigma": sigma},
    )
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE",
        [(1, "A", 1.0), (2, "A", 2.0), (3, "A", 4.0), (4, "A", 8.0)],
    )
    center = offset * (window - 1)
    scale = window / sigma
    weights = np.array([
        np.exp(-((i - center) ** 2) / (2.0 * scale * scale))
        for i in range(window)
    ])
    weights /= weights.sum()
    expected = [
        None,
        None,
        float(np.dot(np.array([1.0, 2.0, 4.0]), weights)),
        float(np.dot(np.array([2.0, 4.0, 8.0]), weights)),
    ]
    assert out[0][2] is None and out[1][2] is None
    assert [row[2] for row in out[2:]] == pytest.approx(expected[2:])


@pytest.mark.parametrize("roc_mode", ["pct", "log"])
def test_followup_coppock_real_duckdb_matches_independent_reference(roc_mode):
    roc1, roc2, wma_window = 2, 1, 3
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    plan = PlanNode(
        "CoppockCurve", (column("x"),),
        {"roc1": roc1, "roc2": roc2, "wma_window": wma_window,
         "roc_mode": roc_mode},
    )
    out = _execute_layer(
        plan, "ts INTEGER, inst VARCHAR, x DOUBLE",
        [(i + 1, "A", float(value)) for i, value in enumerate(values)],
    )
    total = np.full(values.shape, np.nan)
    for i in range(max(roc1, roc2), len(values)):
        if roc_mode == "pct":
            total[i] = values[i] / values[i - roc1] - 1.0
            total[i] += values[i] / values[i - roc2] - 1.0
        else:
            total[i] = np.log(values[i] / values[i - roc1])
            total[i] += np.log(values[i] / values[i - roc2])
    weights = np.arange(1.0, wma_window + 1.0)
    expected = np.full(values.shape, np.nan)
    for i in range(max(roc1, roc2) + wma_window - 1, len(values)):
        expected[i] = np.dot(total[i - wma_window + 1:i + 1], weights) / weights.sum()
    actual = np.array([np.nan if row[2] is None else row[2] for row in out])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


@pytest.mark.parametrize("op", sorted({
    "ElderRay", "atr_pct", "atr_acceleration", "atr_zscore",
    "atr_percentile", "atr_short_long_ratio", "candle_body_strength",
    "candle_wick_balance", "candle_range_pct", "candle_pattern_count",
    "ofi_abs_imbalance_trend", "ofi_imbalance_persistence",
    "ofi_reversal_rate", "aq1_accrual_stability",
    "aq1_accrual_ratio_dispersion",
}))
def test_followup_explicit_sql_deferrals_compile_to_none(op):
    from factor_engine.backend.sql_pushdown.plan_fixtures import minimal_plan
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
    from factor_engine.backend.sql_pushdown.sql_registry import (
        SQL_CAPABLE_CANONICALS as REGISTRY_SQL_CAPABLE_CANONICALS,
    )
    from factor_engine.backend.sql_tiers import (
        SQL_CAPABLE_CANONICALS,
        SQL_EXPLICIT_UNSUPPORTED_REASONS,
        SQL_IMPLEMENTED_CANONICALS,
        SQL_PARITY_VERIFIED_CANONICALS,
        SQL_PRODUCTION_SAFE_CANONICALS,
    )

    assert SQL_EXPLICIT_UNSUPPORTED_REASONS[op]
    assert op not in SQL_IMPLEMENTED_CANONICALS
    assert op not in SQL_CAPABLE_CANONICALS
    assert op not in REGISTRY_SQL_CAPABLE_CANONICALS
    assert op not in SQL_PARITY_VERIFIED_CANONICALS
    assert op not in SQL_PRODUCTION_SAFE_CANONICALS
    assert compile_plan_to_sql(
        minimal_plan(op), dataset="test_panel", time_column="ts",
        instrument_column="inst",
    ) is None
