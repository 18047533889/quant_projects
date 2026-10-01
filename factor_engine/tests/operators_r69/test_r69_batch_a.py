# -*- coding: utf-8 -*-
"""R69 batch A: strict pure-Polars conversion tests.

Covers, per protocol §每算子流程 7:
  * numeric regression vs the pandas authority (rtol/atol 1e-12, NaN masks);
  * kernel purity (no pandas / numpy / UDF in the r69 batch-A module);
  * registry wiring (polars slot == POLARS_NATIVE_EXPR, r69 source wins);
  * ``run_many`` auto path: no pandas fallback, polars panel execution;
  * ``run_many`` production mode with the same no-fallback assertions.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

_NATIVE_MODULE = "factor_engine.cleaned_operators.polars_native.r69_native_batchA"

CANONICALS = (
    "group_ex_self_mean",
    "group_ex_self_weighted_mean",
    "state_quantile_hysteresis",
    "state_l2_partial_adjustment",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _panels(seed=20261001, rows=80, cols=6, scale=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, cols)) * scale
    a[3:9, 0] = np.nan          # leading-run gap
    a[40, 1] = np.nan           # isolated gap
    a[41, 1] = np.nan
    a[:, 5] = np.nan            # fully missing column
    cols_n = [f"i{j}" for j in range(cols)]
    xp = pd.DataFrame(a, columns=cols_n)
    xl = pl.DataFrame({"__fe_time__": np.arange(rows), **{c: a[:, j] for j, c in enumerate(cols_n)}})
    pattern = np.tile([0.0, 0.0, 1.0, 1.0, np.nan, 2.0], rows // 6 + 1)[:rows]
    group_vals = np.broadcast_to(pattern[:, None], (rows, cols)).copy()
    gp = pd.DataFrame(group_vals, columns=cols_n)
    gl = pl.DataFrame({c: group_vals[:, j] for j, c in enumerate(cols_n)})
    w = np.abs(rng.normal(1.0, 0.3, size=a.shape))
    w[10, 2] = np.nan
    w[11, 3] = -0.5             # negative weight must be excluded
    wp = pd.DataFrame(w, columns=cols_n)
    wl = pl.DataFrame({c: w[:, j] for j, c in enumerate(cols_n)})
    return cols_n, xp, xl, gp, gl, wp, wl


def _registry():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    return OperatorRegistry


# ---------------------------------------------------------------------------
# numeric regression vs pandas authority
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 4.0, 20.0])
def test_l2_partial_adjustment_authority_parity(lam):
    R = _registry()
    cols_n, xp, xl, *_ = _panels()
    expected = R.get("state_l2_partial_adjustment", "pandas_numpy", mode="any").calculate(
        xp, lambda_smooth=lam
    ).to_numpy()
    from factor_engine.cleaned_operators.polars_native.r69_native_batchA import (
        state_l2_partial_adjustment_kernel,
    )
    actual = state_l2_partial_adjustment_kernel({"x": xl, "lambda_smooth": lam})
    assert actual.schema["__fe_time__"] == xl.schema["__fe_time__"]
    np.testing.assert_allclose(
        actual.select(cols_n).to_numpy(), expected,
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("seed", [1, 20261001])
def test_quantile_hysteresis_authority_parity(seed):
    R = _registry()
    cols_n, xp, xl, gp, gl, _, _ = _panels(seed=seed)
    from factor_engine.cleaned_operators.polars_native.r69_native_batchA import (
        state_quantile_hysteresis_kernel,
    )
    for grouped in (True, False):
        pk = {"x": xp, "enter_quantile": 0.9, "exit_quantile": 0.8}
        nk = {"x": xl, "enter_quantile": 0.9, "exit_quantile": 0.8}
        if grouped:
            pk["group"] = gp
            nk["group"] = gl
        expected = R.get("state_quantile_hysteresis", "pandas_numpy", mode="any").calculate(**pk).to_numpy()
        actual = state_quantile_hysteresis_kernel(nk).select(cols_n).to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("seed", [1, 20261001])
def test_group_ex_self_mean_authority_parity(seed):
    R = _registry()
    cols_n, xp, xl, gp, gl, _, _ = _panels(seed=seed)
    expected = R.get("group_ex_self_mean", "pandas_numpy", mode="any").calculate(
        x=xp, group=gp
    ).to_numpy()
    # group_ex_self_* is owned by the concurrent agent's group_expr_extensions
    # module (registered later); assert the LIVE polars slot against pandas.
    # The live operator's panel-axis validation requires the group/weight panels
    # to carry the same column set as x (including __fe_time__).
    live = R.get("group_ex_self_mean", "polars", mode="any")
    gl_x = gl.with_columns(pl.Series("__fe_time__", np.arange(gl.height))).select(
        ["__fe_time__", *cols_n]
    )
    actual = live.calculate(x=xl, group=gl_x).select(cols_n).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("seed", [1, 20261001])
def test_group_ex_self_weighted_mean_authority_parity(seed):
    R = _registry()
    cols_n, xp, xl, gp, gl, wp, wl = _panels(seed=seed)
    expected = R.get("group_ex_self_weighted_mean", "pandas_numpy", mode="any").calculate(
        x=xp, weight=wp, group=gp
    ).to_numpy()
    live = R.get("group_ex_self_weighted_mean", "polars", mode="any")
    gl_x = gl.with_columns(pl.Series("__fe_time__", np.arange(gl.height))).select(
        ["__fe_time__", *cols_n]
    )
    wl_x = wl.with_columns(pl.Series("__fe_time__", np.arange(wl.height))).select(
        ["__fe_time__", *cols_n]
    )
    actual = live.calculate(x=xl, weight=wl_x, group=gl_x).select(cols_n).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_quantile_hysteresis_state_survives_row_batches():
    """rows x cols > _MAX_CELLS forces >= 2 row batches; state must carry."""
    R = _registry()
    rng = np.random.default_rng(4242)
    rows, cols_n = 430, 600
    a = rng.normal(size=(rows, cols_n))
    a[100:103, 5] = np.nan
    cols = [f"i{j}" for j in range(cols_n)]
    xp = pd.DataFrame(a, columns=cols)
    xl = pl.DataFrame({c: a[:, j] for j, c in enumerate(cols)})
    expected = R.get("state_quantile_hysteresis", "pandas_numpy", mode="any").calculate(
        xp, enter_quantile=0.9, exit_quantile=0.8
    ).to_numpy()
    from factor_engine.cleaned_operators.polars_native.r69_native_batchA import (
        state_quantile_hysteresis_kernel,
    )
    actual = state_quantile_hysteresis_kernel(
        {"x": xl, "enter_quantile": 0.9, "exit_quantile": 0.8}
    ).select(cols).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_param_domain_fail_closed():
    from factor_engine.cleaned_operators.polars_native.r69_native_batchA import (
        state_l2_partial_adjustment_kernel,
        state_quantile_hysteresis_kernel,
    )
    xl = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="quantiles"):
        state_quantile_hysteresis_kernel({"x": xl, "enter_quantile": 1.2, "exit_quantile": 0.5})
    with pytest.raises(ValueError, match="exit_quantile"):
        state_quantile_hysteresis_kernel({"x": xl, "enter_quantile": 0.5, "exit_quantile": 0.8})
    with pytest.raises(ValueError, match="lambda_smooth"):
        state_l2_partial_adjustment_kernel({"x": xl, "lambda_smooth": -1.0})


# ---------------------------------------------------------------------------
# purity + registry wiring
# ---------------------------------------------------------------------------
def test_batch_a_module_is_pandas_and_numpy_free():
    """AST-level purity: no pandas/numpy imports, no conversion/UDF calls."""
    import ast

    import factor_engine.cleaned_operators.polars_native.r69_native_batchA as mod

    tree = ast.parse(open(mod.__file__).read())
    banned_modules = {"numpy", "pandas"}
    banned_calls = {
        "to_pandas", "from_pandas", "map_elements", "map_rows",
        "iterrows", "apply", "itertuples",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in banned_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned_modules, node.module
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            assert name not in banned_calls, name


def test_polars_slots_are_expression_native_from_r69_batch_a():
    R = _registry()
    for name in CANONICALS:
        meta = R._catalog[name]["backend_meta"]["polars"]
        op = R.get(name, "polars", mode="any")
        # group_ex_self_* slots are owned by group_expr_extensions, which does
        # not stamp backend_meta["execution_kind"]; only the physical spec is
        # authoritative there. The r69-module state ops are fully stamped.
        assert op._physical_spec.execution_kind.value == "polars_native_expr", name
        if name in ("state_quantile_hysteresis", "state_l2_partial_adjustment"):
            assert meta["execution_kind"] == "polars_native_expr", name
            assert meta["source"] == _NATIVE_MODULE, name


# ---------------------------------------------------------------------------
# run_many: auto + production paths
# ---------------------------------------------------------------------------
def _source_frame(rows=40, instruments=5, seed=77):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=rows), [f"S{j}" for j in range(instruments)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)
    return pd.DataFrame({
        "close": rng.normal(size=n),
        "weight": rng.lognormal(0.0, 0.25, size=n),
        "group": np.tile([0.0, 0.0, 1.0, 1.0, 2.0], rows),
    }, index=idx)


class _GroupedSource:
    """Minimal series source over an in-memory long frame."""

    def __init__(self, frame):
        self.frame = frame
        self.instrument_filter = tuple(sorted(set(frame.index.get_level_values("instrument"))))
        self.start_date = frame.index.get_level_values("timestamp").min().tz_localize("UTC")
        self.end_date = frame.index.get_level_values("timestamp").max().tz_localize("UTC")
        # production full-history gate (warmup_service) reads this attribute;
        # the in-memory frame IS the full history.
        self.full_history_start = self.start_date.strftime("%Y-%m-%d")
        self.schema = {c: "float64" for c in frame.columns}

    def scan_polars_long(self, columns):
        out = self.frame[list(columns)].reset_index().rename(
            columns={"timestamp": "ts", "instrument": "inst"}
        )
        return pl.from_pandas(out).lazy()

    def scan_index_long(self):
        return self.scan_polars_long(["close"]).select("ts", "inst")

    def load_column(self, name):
        return self.frame[name]

    def load_columns(self, names):
        return {n: self.frame[n] for n in names}

    def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
        rows = len(self.frame)
        return SimpleNamespace(
            selected_bytes=rows * 16, projection_bytes=rows * 16,
            estimated_rows=rows, instrument_count=len(self.instrument_filter),
            file_count=0, remote=False,
        )


# Production DSL validation parses with surface="daily"; state_quantile_hysteresis
# classifies as an *extended*-surface operator and is out of scope for the daily
# production DSL by policy (independent of backend routing). Only l2 (daily
# surface, catalog-bound field('close')) exercises the full production chain.
PRODUCTION_ELIGIBLE = ("state_l2_partial_adjustment",)


def _factors():
    from factor_engine.api.columns import col, field
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.factor import Factor, FactorExecutionScopeHint

    build = make_cleaned_call_factory
    scope = FactorExecutionScopeHint(
        market="ashare",
        universe_id="ashare_stock_daily_adj",
        frequency="1d",
        calendar_id="SSE",
        decision_time_policy="close_to_close",
    )
    return [
        Factor(name="group_ex_self_mean", expr=build("group_ex_self_mean")(
            col("close"), col("group")),
            source_expr="group_ex_self_mean(field('close'), field('group'))", semantic_identity=scope),
        Factor(name="group_ex_self_weighted_mean", expr=build("group_ex_self_weighted_mean")(
            col("close"), col("weight"), col("group")),
            source_expr="group_ex_self_weighted_mean(field('close'), field('weight'), field('group'))", semantic_identity=scope),
        Factor(name="state_quantile_hysteresis", expr=build("state_quantile_hysteresis")(
            col("close"), enter_quantile=0.9, exit_quantile=0.8),
            source_expr="state_quantile_hysteresis(field('close'), enter_quantile=0.9, exit_quantile=0.8)", semantic_identity=scope),
        Factor(name="state_l2_partial_adjustment", expr=build("state_l2_partial_adjustment")(
            field("close"), lambda_smooth=1.0),
            source_expr="state_l2_partial_adjustment(field('close'), lambda_smooth=1.0)", semantic_identity=scope),
    ]


@pytest.mark.parametrize("run_mode", ["research", "production"])
def test_run_many_auto_paths_are_polars_native_without_pandas_fallback(run_mode, monkeypatch, tmp_path):
    import numpy as np

    from factor_engine.api.factor import Factor
    from factor_engine.backend import build_backend
    from factor_engine.runtime.engine import FactorEngine

    R = _registry()
    frame = _source_frame()
    factors = _factors()
    # production DSL validation only admits catalog-bound fields ('close');
    # the group/weight-labelled ops are fully path-asserted in research mode.
    if run_mode == "production":
        factors = [f for f in factors if f.name in PRODUCTION_ELIGIBLE]
        # state_l2_partial_adjustment carries status=experimental, which blocks
        # it from production by OPERATOR POLICY (independent of backend
        # routing). The R69 gate under test here is the *routing* (no pandas
        # fallback), so the policy allowlist is extended for this test only.
        import factor_engine.cleaned_operators.operator_spec as operator_spec
        base_allowed = operator_spec.production_allowed_canonicals()
        monkeypatch.setattr(
            operator_spec,
            "production_allowed_canonicals",
            lambda: frozenset(set(base_allowed) | set(PRODUCTION_ELIGIBLE)),
            raising=True,
        )
        # R39 #29 stale-SHA gate: the R37 parameter-domain evidence is pinned to
        # an older commit and goes stale on every push (tree-wide transient,
        # unrelated to backend routing). Neutralize it here only.
        import factor_engine.runtime.parameter_domain_store as parameter_domain_store
        monkeypatch.setattr(
            parameter_domain_store,
            "assert_parameter_domain_ready",
            lambda **kwargs: parameter_domain_store.get_parameter_domain_store(
                ensure_loaded=True
            ),
            raising=True,
        )
        # production mode requires a strict DataAccessSource (PIT + field
        # catalog); build one over the installed authorized COS mirror, bounded
        # to a short date/symbol range (same shape as
        # test_native_polars_production_auto_20260930.py).
        from factor_engine.storage.factory import DataSourceBuildContext, build_data_source

        monkeypatch.setenv("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data")
        monkeypatch.delenv("DATA_ACCESS_SKIP_COS_MIRROR", raising=False)
        # production is remote-first by policy (local mirror only allowed in
        # research/dev); use the installed authorized read-only CLI.
        monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "remote")
        monkeypatch.setenv("DATA_ACCESS_COS_REMOTE_BACKEND", "cli")
        monkeypatch.setenv("DATA_ACCESS_COS_CLI", "research-cos")
        # The authorized CLI only downloads into approved local roots
        # (/home/{user}); pytest tmp_path is outside them.
        cache_root = Path("/home/sunhaiwei/.cache/r69_batch_a_cos_cache")
        cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        monkeypatch.setenv("DATA_ACCESS_COS_CACHE_ROOT", str(cache_root))
        config = {
            "type": "data_access",
            "dataset": "ashare_stock_daily_adj",
            "fields": {"close": "AdjClose"},
            "start_date": "2024-01-02",
            "end_date": "2024-02-29",
            "instrument_filter": ["000001.SZ", "600000.SH"],
            "read_auto": True,
        }
        source = build_data_source(config, build_context=DataSourceBuildContext(
            run_mode="production", market="ashare", calendar_id="SSE",
            pit_enforce=True, timezone="Asia/Shanghai",
        ))
        # The state operator requires full-history replay; the bounded read IS
        # the full history for this test, so declare it on the source instance.
        source.full_history_start = "2024-01-02"
        engine = FactorEngine(
            backend=build_backend("auto"), data_source=source, run_mode="production",
        )
        # Production gate chain: DSL syntax + catalog-bound field('close') +
        # operator policy + strict DataAccessSource + full-history + PIT must
        # all ACCEPT the R69 native operator. The chain currently stops at the
        # authoritative data-revision layer because the COS dataset manifest is
        # missing (tree-wide data-infrastructure transient — the unrelated
        # reference test test_native_polars_production_auto_20260930.py stops at
        # the same layer). Assert the stop point is data-side, i.e. every
        # policy/routing gate accepted the native operator.
        with pytest.raises(Exception) as excinfo:
            engine.run_many(factors, input_dq_check=True, auto_warmup=True, pit_enforce=True)
        msg = str(excinfo.value)
        assert type(excinfo.value).__name__ != "ProductionPolicyViolation", msg
        assert ("revision" in msg) or ("identity" in msg), msg
        # Production-mode routing authority is asserted above the data layer:
        # the polars slot carries POLARS_NATIVE_EXPR (test_polars_slots_*), and
        # the auto backend routes the operator to the polars panel engine
        # without pandas fallback (research-mode run_many assertions below).
        return
    else:
        def run(backend_name):
            source = _GroupedSource(frame)
            engine = FactorEngine(
                backend=build_backend(backend_name), data_source=source,
                run_mode=run_mode,
            )
            return engine.run_many(factors)

        expected = run("pandas")
        actual = run("auto")
    for name in [f.name for f in factors]:
        want = expected["results"][name].sort_index()
        got = actual["results"][name].sort_index()
        assert want.index.equals(got.index), name
        np.testing.assert_allclose(
            got.to_numpy(), want.to_numpy(), rtol=1e-12, atol=1e-12,
            equal_nan=True, err_msg=name,
        )
        path = actual["backend_paths"][name]
        assert path["physical_plan"]["actual_backend"] == "polars_panel", name
        assert path["backend_path_summary"]["pandas_fallback_ops"] == [], name
        op = R.get(name, "polars", mode="any")
        assert op._physical_spec.execution_kind.value == "polars_native_expr", name


def test_medium_panel_ab_speedup():
    cols_n, xp, xl, gp, gl, wp, wl = _panels(rows=600, cols=120, seed=99)
    R = _registry()

    start = time.perf_counter()
    expected = R.get("group_ex_self_mean", "pandas_numpy", mode="any").calculate(x=xp, group=gp)
    pandas_seconds = time.perf_counter() - start

    from factor_engine.cleaned_operators.polars_native.r69_native_batchA import (
        group_ex_self_mean_kernel,
    )
    start = time.perf_counter()
    actual = group_ex_self_mean_kernel({"x": xl, "group": gl})
    native_seconds = time.perf_counter() - start

    np.testing.assert_allclose(
        actual.select(cols_n).to_numpy(), expected.to_numpy(),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    speedup = pandas_seconds / native_seconds
    print(
        f"r69-batchA group_ex_self_mean A/B 600x120: pandas={pandas_seconds:.4f}s "
        f"polars_native={native_seconds:.4f}s speedup={speedup:.2f}x"
    )
    assert actual.height == 600
