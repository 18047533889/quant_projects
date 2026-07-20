from __future__ import annotations

import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

STRICT_PERIOD_CASES = [
    (name, lambda: None)
    for name in (
        "period_change", "period_average", "period_cagr", "quarter_from_cumulative",
        "ttm_from_quarterly", "ttm_from_cumulative", "yoy_by_period",
    )
]


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_surfaces_are_disjoint_and_daily_contains_no_inactive_names() -> None:
    from cleaned_operators import operator_surface as surface

    sets = {
        "daily": set(surface.DAILY_CANONICALS),
        "extended": set(surface.EXTENDED_ONLY_CANONICALS),
        "research": set(surface.RESEARCH_ONLY_CANONICALS),
        "unsafe": set(surface.UNSAFE_CANONICALS),
        "legacy": set(surface.LEGACY_ONLY_CANONICALS),
        "internal": set(surface.INTERNAL_ONLY_CANONICALS),
    }
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not sets[left] & sets[right], (left, right, sets[left] & sets[right])
    assert set(surface.DAILY_CANONICALS) <= set(OperatorRegistry.list_canonical()) | {
        # Static review may include a not-yet-imported implementation only when
        # it is explicitly part of the sealed daily additions.
        "ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr",
    }
    forbidden = {
        "vwap", "ttm", "quarter", "yoy", "avg2", "volatility", "WMA",
        "current_ratio", "downside_beta", "rank_corr", "size_neutralize",
        "acos", "atan2", "sin", "cos", "tan", "ts_moment", "ts_max_buildup",
    }
    assert not forbidden & set(surface.DAILY_CANONICALS)


def test_strict_fiscal_period_operators_are_not_production_denied() -> None:
    from cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS

    strict = {
        "period_lag", "period_average", "period_change", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
        "yoy_by_period",
    }
    assert not strict & PRODUCTION_DENIED_CANONICALS
    assert {"ttm", "quarter", "yoy", "avg2"} <= PRODUCTION_DENIED_CANONICALS


def test_pit_and_scope_metadata_do_not_confuse_domain_risk_with_lookahead() -> None:
    from cleaned_operators.operator_policy import infer_operator_policy

    pit_safe = {
        "cs_count", "cs_mean", "cs_std", "cs_sum", "group_count", "group_max",
        "group_min", "group_sum", "is_infinite", "cbrt", "round", "truncate",
        "ts_quantile", "ts_skew", "ts_kurt",
    }
    for canonical in pit_safe & set(OperatorRegistry.list_canonical()):
        assert infer_operator_policy(canonical).pit_safe, canonical
    for canonical in {
        "period_lag", "period_average", "period_change", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
        "yoy_by_period",
    }:
        assert infer_operator_policy(canonical).scope == "fundamental_period", canonical


def test_generated_manifests_share_the_final_registry_snapshot() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    operator_manifest = json.loads((root / "docs/operator_manifest.json").read_text())
    catalog_payload = json.loads((root / "cleaned_operators/docs/operators_catalog.json").read_text())
    backend_manifest = json.loads((root / "docs/backend_evidence_manifest.json").read_text())

    def _as_map(payload: dict) -> dict:
        ops = payload.get("operators")
        if isinstance(ops, dict):
            return ops
        if isinstance(ops, list):
            out = {}
            for row in ops:
                key = row.get("canonical") or row.get("name")
                if key:
                    out[str(key)] = row
            return out
        raise AssertionError(f"unexpected operators payload: {type(ops)!r}")

    operators = _as_map(operator_manifest)
    catalog = _as_map(catalog_payload)
    evidence = _as_map(backend_manifest)
    assert set(operators) == set(catalog) == set(evidence)
    for name, operator in operators.items():
        row = catalog[name]
        ev = evidence[name]
        assert sorted(operator.get("backends") or []) == sorted(row.get("backends") or []), name
        assert bool(operator.get("pit_safe")) == bool(row.get("pit_safe")), name
        # Layer catalog + backend evidence share the sealed surface label.
        assert row.get("surface") == ev.get("surface"), name
        if "surface" in operator:
            assert operator.get("surface") == row.get("surface"), name


def test_explicit_backend_preference_cannot_bypass_production_evidence() -> None:
    # ts_ewm_std is implemented natively but is not independently production-certified.
    op, backend = OperatorRegistry.get_preferred("ts_ewm_std", prefer="polars", mode="production")
    assert backend == "pandas_numpy"
    assert op is OperatorRegistry.get("ts_ewm_std", "pandas_numpy")
    op, backend = OperatorRegistry.get_preferred(
        "ts_ewm_std", prefer="polars", mode="research", allow_unverified_backend=True
    )
    assert backend == "polars"
    assert op is OperatorRegistry.get("ts_ewm_std", "polars")


def test_clickhouse_does_not_inherit_duckdb_parity() -> None:
    from backend.operator_capability import capability_for
    from backend.sql_tiers import CLICKHOUSE_SQL_PARITY_VERIFIED

    assert CLICKHOUSE_SQL_PARITY_VERIFIED == frozenset()
    assert capability_for("ts_mean", "clickhouse_sql").status != "parity_verified"


def test_recursive_ewm_rejects_segment_restart_without_checkpoint_support() -> None:
    from stateful_contract import StatefulCheckpointRegistry, StatefulContractError

    spec = StatefulCheckpointRegistry.get("ts_ewm_std")
    assert spec is not None and not spec.segmented_execution_supported
    with pytest.raises(StatefulContractError, match="unsupported without a native checkpoint"):
        StatefulCheckpointRegistry.require_for_segment(
            "ts_ewm_std", starts_at_dataset_origin=False, checkpoint=None
        )


def test_rank_polars_is_native_and_matches_pandas_edges() -> None:
    pl = pytest.importorskip("polars")
    pdf = pd.DataFrame({"a": [1.0, 1.0, None], "b": [2.0, 1.0, 3.0], "c": [3.0, None, 3.0]})
    expected = OperatorRegistry.get("rank").calculate(pdf)
    actual = OperatorRegistry.get("rank", "polars").calculate(pl.from_pandas(pdf)).to_pandas()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "canonical,args",
    [
        ("period_lag", (1,)),
        ("period_change", (1,)),
        ("period_average", (2,)),
        ("period_cagr", (1, 4)),
        ("ttm_from_quarterly", (4,)),
        ("yoy_by_period", (4,)),
    ],
)
def test_strict_period_polars_matches_pandas_without_future_leakage(canonical, args) -> None:
    pl = pytest.importorskip("polars")
    values = pd.DataFrame({"A": [1.0, 3.0, 2.0, 4.0, 5.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q3", "2023Q2", "2023Q4", "2024Q1"]})
    expected = OperatorRegistry.get(canonical).calculate(values, periods, *args)
    actual = OperatorRegistry.get(canonical, "polars").calculate(
        pl.from_pandas(values), pl.from_pandas(periods), *args
    ).to_pandas()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "canonical,input_names,attrs",
    [
        ("period_change", ("x", "pid"), {"periods": 1}),
        ("period_average", ("x", "pid"), {"periods": 2}),
        ("period_cagr", ("x", "pid"), {"periods": 1, "periods_per_year": 4}),
        ("quarter_from_cumulative", ("x", "pid", "q"), {}),
        ("ttm_from_quarterly", ("x", "pid"), {"periods": 4}),
        ("ttm_from_cumulative", ("x", "pid", "q"), {}),
        ("yoy_by_period", ("x", "pid"), {"periods": 4}),
    ],
)
def test_strict_period_duckdb_executes_real_sql(canonical, input_names, attrs) -> None:
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql
    from planner.logical_plan import PlanNode

    panel = pd.DataFrame({
        "ts": range(6), "inst": ["A"] * 6,
        "x": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
        "pid": ["2023Q1", "2023Q2", "2023Q2", "2023Q3", "2023Q4", "2024Q1"],
        "q": [1.0, 2.0, 2.0, 3.0, 4.0, 1.0],
    })
    plan = PlanNode(
        canonical,
        [PlanNode("column", attrs={"name": name}) for name in input_names],
        attrs=attrs,
    )
    query = compile_plan_to_sql(plan, dataset="panel", time_column="ts", instrument_column="inst").query
    con = duckdb.connect()
    con.register("panel", panel)
    actual = con.execute(query.replace("{{panel}}", "panel")).df().sort_values("ts")["value"].to_numpy()
    wide = {name: pd.DataFrame({"A": panel[name].to_numpy()}) for name in input_names}
    expected = OperatorRegistry.get(canonical).calculate(
        *(wide[name] for name in input_names), **attrs
    )["A"].to_numpy()
    import numpy as np
    np.testing.assert_allclose(actual, expected, equal_nan=True)
