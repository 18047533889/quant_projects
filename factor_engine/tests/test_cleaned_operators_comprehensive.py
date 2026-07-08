"""cleaned_operators 全面集成测试：API、DSL、IR、Planner、Bridge、Backend、未来算子扩展。"""

from __future__ import annotations

import pandas as pd
import pytest

from api import add, delay, divide, multiply, rank, subtract, ts_mean, zscore
from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.dsl_parser import DSLParseError, parse_expr, parse_factor
from api.factor import Factor
from api.operator_registry import STUB_IR_OPS, build_dsl_allowlist
from backend.cleaned_bridge import (
    ensure_cleaned_loaded,
    list_cleaned_ops_for_backend,
    panel_to_series,
    series_to_panel,
)
from backend.context import ExecutionContext
from backend.factory import build_backend
from backend.pandas_backend import PandasBackend
from cleaned_operators.base import OperatorMetadata, SeriesOperator
from cleaned_operators.registry import OperatorRegistry
from expr.base import ensure_expr
from expr.cleaned_call import CleanedCall
from expr.literal import Literal
from ir.analyzer import Analyzer
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from runtime.engine import FactorEngine
from storage.cache import CacheManager
from tests.helpers import InMemorySeriesSource

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _panel_index(n_days: int = 4, instruments: tuple[str, ...] = ("A", "B")):
    return pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=n_days, freq="D"), instruments],
        names=["timestamp", "instrument"],
    )


@pytest.fixture
def panel_source():
    idx = _panel_index(n_days=4, instruments=("A", "B"))
    close = pd.Series(
        [10.0, 20.0, 11.0, 21.0, 12.0, 18.0, 13.0, 19.0],
        index=idx,
    )
    open_ = pd.Series(
        [9.0, 19.0, 10.0, 20.0, 11.0, 17.0, 12.0, 18.0],
        index=idx,
    )
    group = pd.Series(["G1", "G1", "G2", "G2", "G1", "G2", "G1", "G2"], index=idx)
    high = close + 1.0
    low = close - 1.0
    return InMemorySeriesSource(
        data={"close": close, "open": open_, "g": group, "high": high, "low": low}
    )


@pytest.fixture
def three_inst_source():
    """三标的面板：用于验证时序算子不会跨 instrument 泄漏。"""
    idx = _panel_index(n_days=5, instruments=("A", "B", "C"))
    base = {"A": 100.0, "B": 200.0, "C": 300.0}
    vals = []
    for ts_i in range(5):
        for inst in ("A", "B", "C"):
            vals.append(base[inst] + ts_i)
    return InMemorySeriesSource(data={"close": pd.Series(vals, index=idx)})


@pytest.fixture
def engine(panel_source):
    return FactorEngine(backend=PandasBackend(), data_source=panel_source)


def _run(engine: FactorEngine, expr, *, name: str = "t"):
    return engine.run(Factor(name=name, expr=expr))["result"]


def _ctx(source: InMemorySeriesSource) -> ExecutionContext:
    return ExecutionContext(data_source=source)


# ---------------------------------------------------------------------------
# API 层
# ---------------------------------------------------------------------------


class TestApiLayer:
    def test_stub_ir_ops_empty_after_migration(self):
        assert STUB_IR_OPS == frozenset()

    def test_dsl_allowlist_includes_col_and_core_ops(self):
        allow = build_dsl_allowlist()
        assert "col" in allow
        for name in (
            "rank",
            "ts_mean",
            "delay",
            "SMA",
            "add",
            "subtract",
            "multiply",
            "divide",
            "if_else",
            "where",
            "MACD",
            "RSI",
            "winsorize",
            "neutralize",
        ):
            assert name in allow, f"missing {name!r} in allowlist"

    def test_deduplicated_aliases_resolve(self):
        """重复算子注销后，旧名仍可通过别名解析到保留实现。"""
        from cleaned_operators.registry import OperatorRegistry

        pairs = [
            ("SMA", "ts_mean"),
            ("clamp", "clip"),
            ("deltas", "ts_delta"),
            ("neutralize", "group_neutralize"),
            ("pct_change", "ts_pct"),
            ("returns", "ts_pct"),
            ("if_else", "where"),
            ("cap", "clip"),
            ("group_demean", "group_neutralize"),
            ("decay_linear", "ts_decay_linear"),
            ("m_var", "ts_var"),
            ("percentile", "ts_quantile"),
            ("running_std", "ts_std"),
            ("cum_standardize", "expanding_zscore"),
        ]
        for alias, canon in pairs:
            assert OperatorRegistry.get(alias) is OperatorRegistry.get(canon)
        for removed in ("Sum", "clamp", "industry_neutralize", "ts_decay", "Percentile", "running_std", "returns"):
            assert removed not in OperatorRegistry._operators

    def test_named_ops_in_allowlist(self):
        """常用规范名应在 pandas DSL 白名单内。"""
        allow = build_dsl_allowlist()
        for name in ("ts_argmax", "cs_regression", "rank_corr", "ts_pct", "ts_quantile", "clip", "cap"):
            assert name in allow, f"missing {name!r} in allowlist"

    def test_allowlist_factory_returns_cleaned_call(self):
        allow = build_dsl_allowlist()
        node = allow["rank"](col("x"))
        assert isinstance(node, CleanedCall)
        assert node.op == "rank"

    def test_dynamic_getattr_resolves_alias_and_caches(self):
        import api

        factory = api.ts_delay
        assert callable(factory)
        assert api.ts_delay is factory

    def test_dynamic_getattr_unknown_raises(self):
        import api

        with pytest.raises(AttributeError):
            _ = api.this_operator_does_not_exist_xyz

    def test_parse_factor_wrapper(self):
        f = parse_factor('rank(col("close"))', name="from_dsl", universe="US")
        assert f.name == "from_dsl"
        assert f.universe == "US"
        assert isinstance(f.expr, CleanedCall)


# ---------------------------------------------------------------------------
# Expr 四则 / 字面量
# ---------------------------------------------------------------------------


class TestExprConstruction:
    @pytest.mark.parametrize(
        "builder,op",
        [
            (lambda: col("a") + col("b"), "add"),
            (lambda: col("a") - col("b"), "subtract"),
            (lambda: col("a") * col("b"), "multiply"),
            (lambda: col("a") / col("b"), "divide"),
            (lambda: 1 + col("a"), "add"),
            (lambda: 10 - col("a"), "subtract"),
            (lambda: 2 * col("a"), "multiply"),
            (lambda: 100 / col("a"), "divide"),
        ],
    )
    def test_binop_overloads(self, builder, op):
        expr = builder()
        assert isinstance(expr, CleanedCall)
        assert expr.op == op

    def test_neg_and_ensure_expr(self):
        assert isinstance(-col("x"), CleanedCall)
        lit = ensure_expr(42)
        assert isinstance(lit, Literal)
        assert lit.value == 42

    def test_cleaned_call_with_kwargs(self):
        node = make_cleaned_call_factory("winsorize")(
            col("close"), lower=0.05, upper=0.95
        )
        assert node.kwargs_dict() == {"lower": 0.05, "upper": 0.95}

    def test_cleaned_call_expr_kwargs(self):
        node = make_cleaned_call_factory("ts_mean")(col("close"), window=3)
        assert node.kwargs_dict()["window"] == 3


# ---------------------------------------------------------------------------
# DSL 解析
# ---------------------------------------------------------------------------


class TestDslParsing:
    def test_binop_via_ast_becomes_cleaned_call(self):
        expr = parse_expr('col("close") - col("open")')
        assert isinstance(expr, CleanedCall)
        assert expr.op == "subtract"

    def test_scalar_and_reverse_binop(self):
        assert parse_expr('col("close") * 2').op == "multiply"
        assert parse_expr('2 * col("close")').op == "multiply"
        assert parse_expr('10 - col("close")').op == "subtract"

    def test_unary_neg_via_dsl(self):
        assert parse_expr("-col('close')").op == "neg"

    @pytest.mark.parametrize(
        "text,op",
        [
            ('col("a") < col("b")', "lt"),
            ('col("a") <= 1', "le"),
            ('col("a") == col("b")', "eq"),
            ('col("a") > col("b")', "gt"),
            ('col("a") >= 1', "ge"),
            ('col("a") != col("b")', "ne"),
        ],
    )
    def test_comparison_ops(self, text, op):
        assert parse_expr(text).op == op

    def test_dsl_kwargs(self):
        expr = parse_expr('winsorize(col("close"), lower=0.1, upper=0.9)')
        assert isinstance(expr, CleanedCall)
        assert expr.kwargs_dict()["lower"] == 0.1

    def test_dsl_bool_literals(self):
        expr = parse_expr('if_else(col("close") > 0, True, False)')
        assert isinstance(expr, CleanedCall)

    def test_rejects_chained_comparison(self):
        with pytest.raises(DSLParseError):
            parse_expr('1 < col("x") < 2')

    def test_rejects_unsafe_ast(self):
        with pytest.raises(DSLParseError):
            parse_expr('__import__("os").system("x")')

    def test_rejects_unknown_function(self):
        with pytest.raises(DSLParseError):
            parse_expr('not_a_real_operator(col("x"))')


# ---------------------------------------------------------------------------
# IR 降级
# ---------------------------------------------------------------------------


class TestIrLowering:
    @pytest.mark.parametrize(
        "dsl,canonical",
        [
            ('delay(col("close"), 1)', "ts_delay"),
            ('mean(col("close"), 2)', "ts_mean"),
            ('ts_macd(col("close"))', "MACD"),
            ('ts_bbands(col("close"), 20)', "BollingerBands"),
            ('shift(col("close"), 1)', "ts_delay"),
            ('std(col("close"), 3)', "ts_std"),
        ],
    )
    def test_alias_canonicalized(self, dsl, canonical):
        ir = Analyzer().lower(parse_expr(dsl)).ir
        assert ir.op == canonical

    def test_lookback_max_across_nested_ts(self):
        expr = parse_expr('ts_mean(col("close"), 3) + delay(col("close"), 5)')
        assert Analyzer().lower(expr).lookback == 5

    def test_referenced_columns_union(self):
        expr = parse_expr('rank(col("close") - col("open"))')
        assert Analyzer().lower(expr).referenced_columns == {"close", "open"}

    def test_kwargs_literal_in_ir_attrs(self):
        ir = Analyzer().lower(
            make_cleaned_call_factory("winsorize")(col("close"), lower=0.1, upper=0.9)
        ).ir
        assert ir.attrs["lower"] == 0.1
        assert ir.attrs["upper"] == 0.9

    def test_non_literal_kwarg_in_ir_raises(self):
        with pytest.raises(NotImplementedError):
            Analyzer().lower(
                make_cleaned_call_factory("winsorize")(
                    col("close"), lower=col("open"), upper=0.9
                )
            )


# ---------------------------------------------------------------------------
# Planner / Optimizer
# ---------------------------------------------------------------------------


class TestPlannerOptimizer:
    @pytest.mark.parametrize(
        "factory,expected",
        [
            (lambda: add(Literal(2.0), Literal(3.0)), 5.0),
            (lambda: subtract(Literal(5.0), Literal(2.0)), 3.0),
            (lambda: multiply(Literal(2.0), Literal(4.0)), 8.0),
            (lambda: divide(Literal(8.0), Literal(2.0)), 4.0),
        ],
    )
    def test_literal_fold_all_binops(self, factory, expected):
        plan = Optimizer().optimize(
            Lowerer().to_logical_plan(Analyzer().lower(factory()).ir)
        )
        assert plan.op == "literal"
        assert float(plan.attrs["value"]) == expected

    def test_divide_by_zero_not_folded(self):
        plan = Optimizer().optimize(
            Lowerer().to_logical_plan(
                Analyzer().lower(divide(Literal(1.0), Literal(0.0))).ir
            )
        )
        assert plan.op == "divide"


# ---------------------------------------------------------------------------
# Bridge panel 转换
# ---------------------------------------------------------------------------


class TestCleanedBridge:
    def test_series_panel_roundtrip(self, panel_source):
        ensure_cleaned_loaded()
        ctx = _ctx(panel_source)
        s = panel_source.load_column("close")
        panel = series_to_panel(s, ctx)
        assert list(panel.columns) == ["A", "B"]
        back = panel_to_series(panel, ctx, template=s)
        pd.testing.assert_series_equal(back, s)

    def test_series_to_panel_requires_multiindex(self, panel_source):
        ctx = _ctx(panel_source)
        with pytest.raises(TypeError):
            series_to_panel(pd.Series([1.0, 2.0]), ctx)


# ---------------------------------------------------------------------------
# Backend 注册与错误路径
# ---------------------------------------------------------------------------


class TestBackendRegistration:
    def test_all_cleaned_ops_registered(self):
        backend = PandasBackend()
        registered = set(backend._registry._kernels.keys())
        cleaned = set(list_cleaned_ops_for_backend(set()))
        assert cleaned <= registered
        assert {"column", "literal"} <= registered

    def test_unregistered_op_raises(self, panel_source):
        backend = PandasBackend()
        ctx = _ctx(panel_source)
        from planner.logical_plan import PlanNode

        bad = PlanNode(op="___not_registered_op___", inputs=[], attrs={})
        with pytest.raises(NotImplementedError):
            backend.execute(bad, ctx)


# ---------------------------------------------------------------------------
# 执行：按类别 smoke + 边界
# ---------------------------------------------------------------------------


class TestExecutionCore:
    def test_binop_code_and_dsl_equivalent(self, engine):
        a = _run(engine, col("close") - col("open"))
        b = _run(engine, parse_expr('col("close") - col("open")'))
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_ts_mean_rank_momentum(self, engine):
        result = _run(engine, rank(ts_mean(col("close"), 2)))
        assert result.loc[(pd.Timestamp("2024-01-03"), "A")] == 0.5
        assert result.loc[(pd.Timestamp("2024-01-03"), "B")] == 1.0

    def test_delay_shifts_panel(self, engine):
        result = _run(engine, delay(col("close"), 1))
        close = engine.data_source.load_column("close")
        for inst in ("A", "B"):
            sub = close.xs(inst, level="instrument").sort_index()
            got = result.xs(inst, level="instrument").sort_index()
            assert pd.isna(got.iloc[0])
            pd.testing.assert_series_equal(
                got.iloc[1:].reset_index(drop=True),
                sub.iloc[:-1].reset_index(drop=True),
                check_names=False,
            )

    def test_sma_matches_ts_mean_window(self, engine):
        sma = _run(engine, parse_expr('SMA(col("close"), 2)'))
        mean = _run(engine, ts_mean(col("close"), 2))
        pd.testing.assert_series_equal(sma, mean, check_names=False, rtol=1e-9)

    def test_zscore_cross_sectional_per_timestamp(self, engine):
        result = _run(engine, zscore(col("close")))
        for ts in result.index.get_level_values("timestamp").unique():
            slice_ = result.xs(ts, level="timestamp")
            if slice_.notna().sum() >= 2:
                assert abs(float(slice_.mean())) < 1e-9

    def test_instrument_isolation_ts_mean(self, three_inst_source):
        """宽表 panel 下各 instrument 列独立 rolling，不互相污染。"""
        eng = FactorEngine(backend=PandasBackend(), data_source=three_inst_source)
        result = _run(eng, ts_mean(col("close"), 2))
        for inst, base in [("A", 100.0), ("B", 200.0), ("C", 300.0)]:
            sub = result.xs(inst, level="instrument").sort_index()
            # window=2：第二行起为 (base+i + base+i+1)/2
            assert sub.iloc[1] == pytest.approx(base + 0.5)
        # A 与 B 的 rolling 均值应相差 100（基数差），而非混成同一序列
        a1 = result.xs("A", level="instrument").sort_index().iloc[1]
        b1 = result.xs("B", level="instrument").sort_index().iloc[1]
        assert b1 - a1 == pytest.approx(100.0)


class TestExecutionByCategory:
    """按 cleaned 业务类别抽样：math / ts / cs / signal / price_volume 等。"""

    @pytest.mark.parametrize(
        "dsl",
        [
            'abs(col("close"))',
            'log(col("close"))',
            'sign(col("close"))',
            'power(col("close"), 2)',
            'ts_std(col("close"), 3)',
            'ts_delta(col("close"), 1)',
            'ts_sum(col("close"), 3)',
            'decay_linear(col("close"), 3)',
            'scale(col("close"))',
            'winsorize(col("close"), lower=0.05, upper=0.95)',
            'neutralize(col("close"), col("g"))',
            'if_else(col("close") > 15, col("close"), 0)',
            'where(col("close") > 15, col("close"), 0)',
            'RSI(col("close"), 14)',
            'ATR(col("high"), col("low"), col("close"), 14)',
            'ts_corr(col("close"), col("high"), 3)',
            'group_rank(col("close"), col("g"))',
            'rank(ts_mean(col("close"), 2) - delay(col("close"), 1))',
        ],
    )
    def test_category_smoke(self, engine, dsl):
        result = _run(engine, parse_expr(dsl))
        assert isinstance(result, pd.Series)
        assert result.index.names == ["timestamp", "instrument"]
        assert result.notna().any()


class TestExecutionEngineFeatures:
    def test_run_many_cse_shared_subtree(self, panel_source):
        eng = FactorEngine(backend=PandasBackend(), data_source=panel_source)
        sub = ts_mean(col("close"), 2)
        f1 = Factor(name="a", expr=sub)
        f2 = Factor(name="b", expr=rank(sub))
        out = eng.run_many([f1, f2])
        assert len(out["dag"].shared_nodes) >= 1
        pd.testing.assert_series_equal(
            out["results"]["a"], eng.run(f1)["result"], check_names=False
        )

    def test_cache_returns_same_values(self, panel_source):
        cache = CacheManager()
        eng = FactorEngine(
            backend=PandasBackend(), data_source=panel_source, cache=cache
        )
        factor = Factor(name="f", expr=rank(col("close")))
        r1 = eng.run(factor)["result"]
        r2 = eng.run(factor)["result"]
        pd.testing.assert_series_equal(r1, r2, check_names=False)

    def test_polars_backend_matches_pandas(self, panel_source):
        """PolarsBackend 走 auto 优先 polars，结果与 PandasBackend 对齐。"""
        expr = rank(ts_mean(col("close"), 2))
        eng_pd = FactorEngine(
            backend=build_backend("pandas"), data_source=panel_source
        )
        eng_pl = FactorEngine(
            backend=build_backend("polars"), data_source=panel_source
        )
        pd.testing.assert_series_equal(
            eng_pd.run(Factor(name="t", expr=expr))["result"],
            eng_pl.run(Factor(name="t", expr=expr))["result"],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# 未来算子 / 扩展注册（测试框架）
# ---------------------------------------------------------------------------


class TestFutureOperatorExtension:
    """模拟未来新增算子：注册 → 白名单 → IR → 执行 全链路。"""

    @pytest.fixture
    def future_double_op(self):
        ensure_cleaned_loaded()

        class FutureDoubleOp(SeriesOperator):
            metadata = OperatorMetadata(
                name="future_double",
                category="math",
                description="panel x * 2",
                param_names=["x"],
            )

            def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
                return x * 2.0

        op = FutureDoubleOp()
        OperatorRegistry.register(
            op, canonical="future_double", backend="pandas_numpy", source="test"
        )
        OperatorRegistry.register_alias("FUTURE_DOUBLE", "future_double")
        yield "future_double"
        OperatorRegistry._operators.pop("future_double", None)
        OperatorRegistry._catalog.pop("future_double", None)
        OperatorRegistry._aliases.pop("FUTURE_DOUBLE", None)

    def test_future_op_in_allowlist_after_register(self, future_double_op):
        allow = build_dsl_allowlist()
        assert future_double_op in allow
        assert "FUTURE_DOUBLE" in allow

    def test_future_op_execute_via_engine(self, future_double_op, panel_source):
        eng = FactorEngine(backend=PandasBackend(), data_source=panel_source)
        expr = make_cleaned_call_factory(future_double_op)(col("close"))
        result = _run(eng, expr)
        expected = panel_source.load_column("close") * 2
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_future_op_dsl_parse_and_ir(self, future_double_op):
        allow = build_dsl_allowlist()
        expr = allow[future_double_op](col("close"))
        ir = Analyzer().lower(expr).ir
        assert ir.op == future_double_op


class TestFutureStubOperators:
    """未来 stub 算子：仅 catalog、无 runtime 时不应进入 DSL/backend。"""

    def test_catalog_only_op_not_in_allowlist(self):
        ensure_cleaned_loaded()
        name = "_test_catalog_only_op_xyz"
        OperatorRegistry.register_catalog_only(
            name,
            status="doc_only",
            business_category="fundamental",
            description="future stub",
        )
        try:
            allow = build_dsl_allowlist()
            assert name not in allow
        finally:
            OperatorRegistry._catalog.pop(name, None)

    def test_unimplemented_canonical_not_in_backend(self):
        ensure_cleaned_loaded()
        name = "_test_unimplemented_op_xyz"
        OperatorRegistry.register_catalog_only(name, status="planned")
        try:
            assert name not in list_cleaned_ops_for_backend(set())
        finally:
            OperatorRegistry._catalog.pop(name, None)


class TestRealWorldFactorPatterns:
    """常见因子模板：动量、反转、波动、价量。"""

    @pytest.mark.parametrize(
        "name,dsl",
        [
            ("mom_rank", 'rank(ts_mean(col("close"), 2) / delay(col("close"), 1) - 1)'),
            ("rev_zscore", "zscore(col('close') - ts_mean(col('close'), 10))"),
            ("vol_rank", "rank(ts_std(col('close'), 2))"),
            ("range_pos", "rank((col('close') - col('low')) / (col('high') - col('low')))"),
        ],
    )
    def test_factor_pattern_runs(self, engine, name, dsl):
        result = _run(engine, parse_expr(dsl), name=name)
        assert result.notna().any()


class TestEndToEndCompilePlan:
    def test_compile_plan_op_chain(self, engine):
        out = engine.run(Factor(name="mom", expr=rank(ts_mean(col("close"), 2))))
        plan = out["plan"]
        assert plan.op == "rank"
        assert plan.inputs[0].op == "ts_mean"
        assert plan.inputs[0].inputs[0].op == "column"
        assert out["analysis"].referenced_columns == {"close"}
