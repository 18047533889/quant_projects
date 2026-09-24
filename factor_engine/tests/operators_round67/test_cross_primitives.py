# -*- coding: utf-8 -*-
"""R67: ``cross_above`` / ``cross_under`` — semantics, first-bar contract, 3 backends.

The reference implementation below is written **from the specification**
(``a_t > b_t`` and ``a_{t-1} <= b_{t-1}``; row 0 → NaN; any non-finite value at
the current *or* previous row of either input → NaN; otherwise the float event
indicator 1.0/0.0).  It never imports the code under test, so it is an
independent oracle.

Coverage:
* exact crossing hits on a deterministic battery — continuous-equality plateau,
  NaN segment, tie boundary, ±Inf, first bar;
* first-bar / NaN contract is identical to the stateful ``cross_event``
  authority (elementwise equality, NaN included);
* pandas_numpy / polars / DuckDB-SQL three-backend parity;
* DSL wiring: canonical/alias parse through ``parse_expr(surface="all")``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

import polars as pl

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

load_all()

CANONICALS = ("cross_above", "cross_under")


# ---------------------------------------------------------------------------
# Independent reference oracle (spec-derived, never imports the subject code)
# ---------------------------------------------------------------------------


def _reference(av: np.ndarray, bv: np.ndarray, *, above: bool) -> np.ndarray:
    n = len(av)
    out = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        ok = (
            np.isfinite(av[t])
            and np.isfinite(bv[t])
            and np.isfinite(av[t - 1])
            and np.isfinite(bv[t - 1])
        )
        if not ok:
            continue
        if above:
            out[t] = 1.0 if (av[t] > bv[t] and av[t - 1] <= bv[t - 1]) else 0.0
        else:
            out[t] = 1.0 if (av[t] < bv[t] and av[t - 1] >= bv[t - 1]) else 0.0
    return out


def _panel(values: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=idx, dtype=float)


def _cross(av, bv, above: bool):
    a = pd.DataFrame({"S": np.asarray(av, dtype=float)}, index=pd.RangeIndex(len(av)))
    b = pd.DataFrame({"S": np.asarray(bv, dtype=float)}, index=pd.RangeIndex(len(bv)))
    canonical = "cross_above" if above else "cross_under"
    op = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    return np.asarray(op.calculate(a, b).to_numpy(dtype=float)).ravel()


# Deterministic battery: name -> (a, b, above)
CASES = {
    # plain up-cross at t=1, then no further cross
    "simple_up": ([1.0, 2.0, 3.0], [1.5, 1.5, 1.5], True),
    # equality plateau then strict breakout: only t=3 fires (prev == is <=)
    "equality_plateau_up": ([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], True),
    # tie boundary: prev equal fires, current equal does not
    "tie_boundary": ([1.0, 1.0, 1.0], [1.0, 1.0, 2.0], True),
    # NaN in the middle kills that row and the row after it (prev not finite)
    "nan_gap": ([1.0, 3.0, np.nan, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0], True),
    # first bar has no predecessor -> NaN even when it is an obvious state
    "first_bar": ([5.0, 6.0], [1.0, 1.0], True),
    # +Inf / -Inf are invalid operands, not infinity maths
    "inf_values": ([1.0, np.inf, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], True),
    # down-cross mirror
    "simple_down": ([3.0, 2.0, 1.0], [1.5, 1.5, 1.5], False),
    # down-cross from an equality plateau
    "equality_plateau_down": ([1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], False),
    # both series flat: never crosses, but must emit 0.0 (not NaN) after row 0
    "flat": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], True),
    "flat_down": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], False),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_semantics_match_independent_oracle(case):
    a, b, above = CASES[case]
    got = _cross(a, b, above)
    exp = _reference(np.asarray(a, float), np.asarray(b, float), above=above)
    np.testing.assert_array_equal(got, exp)
    # row 0 must be NaN (no predecessor) unless the whole series is empty
    assert np.isnan(got[0])


def test_equality_plateau_hit_is_exact():
    got = _cross([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], True)
    # only the strict-breakout bar fires; the plateau itself never fires
    np.testing.assert_array_equal(np.isnan(got), [True, False, False, False])
    np.testing.assert_array_equal(got[1:], [0.0, 0.0, 1.0])


def test_nan_segment_output_is_nan_and_rebaselines():
    got = _cross([1.0, 3.0, np.nan, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0], True)
    # t=1 up-cross fires; t=2 (NaN current) NaN; t=3 (prev NaN) NaN; t=4 0.0
    np.testing.assert_array_equal(np.isnan(got), [True, False, True, True, False])
    assert got[1] == 1.0
    assert got[4] == 0.0


def test_inf_is_invalid_not_infinity_maths():
    got = _cross([1.0, np.inf, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], True)
    # t=2 must be NaN (prev is +Inf), never a fabricated 1.0 from 3.0 > 2.0
    assert np.isnan(got[2])
    np.testing.assert_array_equal(np.isnan(got), [True, True, True, False])


def test_output_dtype_is_float_event_indicator():
    op = OperatorRegistry.get("cross_above", "pandas_numpy", mode="any")
    out = op.calculate(
        pd.DataFrame({"S": [1.0, 2.0]}), pd.DataFrame({"S": [1.5, 1.5]})
    )
    assert out.to_numpy().dtype == np.float64
    assert set(np.unique(out.to_numpy()[~np.isnan(out.to_numpy())])) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# First-bar / NaN contract must equal the stateful cross_event authority
# ---------------------------------------------------------------------------


def _random_panel(seed: int = 11, n: int = 60, cols: int = 3):
    rng = np.random.default_rng(seed)
    data = {
        f"S{c}": rng.normal(0.0, 1.0, n).cumsum() for c in range(cols)
    }
    df = pd.DataFrame(data, index=pd.date_range("2024-01-02", periods=n))
    # inject NaN / Inf holes to stress the missing-value contract
    df.iloc[5, 0] = np.nan
    df.iloc[9, 1] = np.inf
    df.iloc[20, 0] = -np.inf
    df.iloc[33, 2] = np.nan
    return df


@pytest.mark.parametrize("canonical,direction", [("cross_above", "up"), ("cross_under", "down")])
def test_first_bar_and_nan_contract_matches_cross_event(canonical, direction):
    a = _random_panel(seed=3)
    b = _random_panel(seed=29)
    got = OperatorRegistry.get(canonical, "pandas_numpy", mode="any").calculate(a, b)
    ref = OperatorRegistry.get("cross_event", "pandas_numpy", mode="any").calculate(
        a, b, direction=direction
    )
    np.testing.assert_array_equal(got.to_numpy(), ref.to_numpy())


# ---------------------------------------------------------------------------
# Three-backend parity (pandas_numpy / polars / DuckDB SQL)
# ---------------------------------------------------------------------------


def _parity_source():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp(f"2024-01-{d:02d}"), sym)
            for sym in ("A", "B")
            for d in (2, 3, 4, 5, 6, 7)
        ],
        names=["timestamp", "instrument"],
    )
    # A: cross above its trailing mean at 2024-01-05; NaN hole at 01-03.
    # B: monotone rise then cross under at 01-06.
    close = pd.Series(
        [
            10.0, np.nan, 11.0, 15.0, 12.0, 13.0,   # A
            20.0, 18.0, 22.0, 19.0, 21.0, 15.0,     # B
        ],
        index=idx,
    )
    return InMemorySeriesSource(data={"close": close})


@pytest.fixture(scope="module")
def mem_source():
    return _parity_source()


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_r67:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), close in mem.data["close"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Close": float(close) if pd.notna(close) else None,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, mem_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
    _seed_duckdb(tmp_path / "data", mem_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_r67"})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


_EXPRS = {
    "cross_above": lambda c: make_cleaned_call_factory("cross_above")(
        c, make_cleaned_call_factory("ts_mean")(c, 3)
    ),
    "cross_under": lambda c: make_cleaned_call_factory("cross_under")(
        c, make_cleaned_call_factory("ts_mean")(c, 3)
    ),
}


def _sql_col(name: str):
    return col({"close": "Close"}[name])


@pytest.mark.parametrize("canonical", CANONICALS)
def test_polars_long_matches_pandas(mem_source, canonical):
    expr = _EXPRS[canonical](col("close"))
    pd_out = _series(_run(mem_source, expr, "pandas"))
    pl_run = _run(mem_source, expr, "polars_long")
    assert pl_run.get("used_polars_long_path") is True
    assert not pl_run.get("polars_long_fallback_reason")
    _assert_equal(pd_out, _series(pl_run), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("canonical", CANONICALS)
def test_duckdb_sql_matches_pandas(mem_source, duckdb_source, canonical):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    pd_out = _series(_run(mem_source, _EXPRS[canonical](col("close")), "pandas"))
    sql_run = _run(duckdb_source, _EXPRS[canonical](_sql_col("close")), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    _assert_equal(pd_out, _series(sql_run), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("canonical", CANONICALS)
def test_polars_long_matches_duckdb_sql(mem_source, duckdb_source, canonical):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    pl_out = _series(_run(mem_source, _EXPRS[canonical](col("close")), "polars_long"))
    sql_run = _run(duckdb_source, _EXPRS[canonical](_sql_col("close")), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    _assert_equal(pl_out, _series(sql_run), rtol=1e-12, atol=1e-12)


def _assert_equal(a: pd.Series, b: pd.Series, *, rtol: float, atol: float) -> None:
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Operator-level polars native kernel parity (no engine)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canonical,above", [("cross_above", True), ("cross_under", False)])
def test_polars_native_kernel_matches_pandas_operator(canonical, above):
    a = _random_panel(seed=5)
    b = _random_panel(seed=17)
    pd_out = OperatorRegistry.get(canonical, "pandas_numpy", mode="any").calculate(a, b)

    pl_a = pl.DataFrame({c: a[c].to_numpy() for c in a.columns})
    pl_b = pl.DataFrame({c: b[c].to_numpy() for c in b.columns})
    pl_op = OperatorRegistry.get(canonical, "polars", mode="any")
    pl_out = pl_op.calculate(pl_a, pl_b)
    got = pl_out.select(list(a.columns)).to_numpy()
    np.testing.assert_array_equal(got, pd_out.to_numpy())


# ---------------------------------------------------------------------------
# DSL wiring
# ---------------------------------------------------------------------------


def test_canonicals_registered_on_all_three_backends():
    for canonical in CANONICALS:
        backends = OperatorRegistry.backends_for(canonical)
        for required in ("pandas_numpy", "polars", "sql"):
            assert required in backends, f"{canonical} missing {required} backend"
        assert classify_canonical(canonical) == "daily"


@pytest.mark.parametrize(
    "expr",
    [
        "cross_above(close, ts_mean(close, 20))",
        "cross_under(close, ts_mean(close, 20))",
        "crosses_above(close, volume)",
        "crosses_under(close, volume)",
        "cross_over(close, volume)",
    ],
)
def test_dsl_parse_surface_all(expr):
    parse_expr(expr, surface="all")


def test_alias_resolution():
    assert OperatorRegistry.resolve_canonical("crosses_above") == "cross_above"
    assert OperatorRegistry.resolve_canonical("cross_over") == "cross_above"
    assert OperatorRegistry.resolve_canonical("crosses_under") == "cross_under"
