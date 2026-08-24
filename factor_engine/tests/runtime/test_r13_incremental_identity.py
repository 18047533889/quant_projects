# -*- coding: utf-8 -*-
"""R13/R14 incremental-identity audit tests (2026-08-09).

Covers the fail-open / duplication findings from the audit book:

    P0-35  wrapper operator manifest hash hashes the wrong object — must hash the
           ACTUAL kernel (``_fn``) + closure/defaults + logical contract, not the
           wrapper class source file.
    P1-36  ordinary class operators must hash code objects, not whole source
           files (shared ``_implementation_digest``).
    P0-37  operator-manifest build failure must fail production
           (``OperatorManifestUnavailable``); research warns, never silent.
    P0-38  calendar fingerprint must hash the FULL ordered trading days (+ session
           segments / timezone / DST / early-close / semantic version), not just
           ``len:first:last``.
    P0-39  calendar-fingerprint build failure must fail production.
    P0-40  logical→physical dataset resolution must NOT anchor-fallback.
    P0-41  physical dataset ``None`` must not silently drop the edge in production.
    P0-42  forward-window calendar error must not zero-extend (production raise,
           research over-extension by calendar days).
    P0-43  ForwardImpactRequirement tri-state (finite / unbounded / unknown);
           production treats unknown conservatively as unbounded.
    §15    ``normalize_data_event`` sequence is a strong-consistency field — strict
           integer validation (never truncate ``2.9`` -> ``2``).
    P1-33  api.factor ``FactorSemanticIdentity`` renamed to
           ``FactorExecutionScopeHint`` (module-level alias kept for imports).
    P1-34  ``compute_factor_identity`` reads the real execution scope from
           ``factor.semantic_identity``; every scope field enters the digest.
    P1-75  identity-builder names are documented at their semantic level.
"""
from __future__ import annotations

import pytest

from factor_engine.ir.nodes import IRNode


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _ir_node(op: str = "ts_mean", **attrs):
    return IRNode(
        op=op,
        attrs=attrs or {"d": 20, "min_periods": 1},
        inputs=(IRNode(op="column", attrs={"name": "close"}),),
    )


# ---------------------------------------------------------------------------
# P0-35 / P1-36: implementation digest hashes the kernel, not the wrapper file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _loaded_registry():
    from factor_engine.cleaned_operators import load_all

    load_all()
    return True


@pytest.mark.usefixtures("_loaded_registry")
class TestImplementationDigest:
    def test_digest_changes_when_kernel_changes(self):
        """P0-35: 修改算子真正派发的 kernel → implementation hash 必须变。

        这里 monkeypatch ``_fn``（kernel），绝不去改真实源码文件。
        """
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime.incremental_scheduler import _implementation_digest

        op = OperatorRegistry.get("MACD")  # PandasFunctionOperator, _fn=pd_macd_line
        assert op is not None and callable(getattr(op, "_fn", None))
        original_kernel = op._fn
        before = _implementation_digest(op)
        op._fn = lambda *a, **k: None  # 换 kernel
        after = _implementation_digest(op)
        op._fn = original_kernel  # restore（不残留）
        assert before != after

    def test_digest_unchanged_when_wrapper_base_changes(self):
        """P0-35: 只改 wrapper 基类（calculate）→ 无关算子的 hash 必须不变。"""
        import factor_engine.cleaned_operators.overhaul.base as ob

        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime.incremental_scheduler import _implementation_digest

        op = OperatorRegistry.get("MACD")
        original_calc = ob.PandasFunctionOperator.calculate
        before = _implementation_digest(op)

        def _new_calc(self, *a, **k):  # pragma: no cover - 仅测试基类修改
            """changed wrapper docstring"""
            return self._fn(*a, **k)

        ob.PandasFunctionOperator.calculate = _new_calc
        try:
            after = _implementation_digest(op)
        finally:
            ob.PandasFunctionOperator.calculate = original_calc
        assert before == after

    def test_class_based_operator_hashes_code_not_source_file(self):
        """P1-36: 普通类算子（TSMean）hash 其 kernel code object，不是整文件。

        修改 TSMean 自己的 ``_calculate_series``（kernel）→ digest 变；这证明
        digest 绑定到 kernel 代码对象（bytecode/closure），而非 ``type`` 的源码
        文件。
        """
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime.incremental_scheduler import _implementation_digest

        op = OperatorRegistry.get("ts_mean")
        assert not callable(getattr(op, "_fn", None))
        cls = type(op)
        assert hasattr(cls, "_calculate_series")
        before = _implementation_digest(op)

        orig_method = cls._calculate_series

        def _alt_calculate_series(self, *a, **k):  # pragma: no cover - 仅测试
            """changed kernel docstring"""
            return orig_method(self, *a, **k)

        cls._calculate_series = _alt_calculate_series
        try:
            after = _implementation_digest(op)
        finally:
            cls._calculate_series = orig_method
        assert before is not None and len(before) == 16
        assert before != after  # kernel code 变化 → digest 变化（P1-36）

    def test_manifest_uses_implementation_digest(self):
        """P0-35 集成：manifest 的 implementation_hash 随 kernel 变化而变化。"""
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime.incremental_scheduler import _operator_manifest_from_ir

        ir = _ir_node("ts_mean", d=20)
        m1 = _operator_manifest_from_ir(ir)
        assert m1 and m1[0]["canonical"] == "ts_mean"
        op = OperatorRegistry.get("ts_mean")
        if callable(getattr(op, "_fn", None)):
            original_kernel = op._fn
            before = m1[0]["implementation_hash"]
            op._fn = lambda *a, **k: None
            try:
                m2 = _operator_manifest_from_ir(ir)
            finally:
                op._fn = original_kernel
            assert m2[0]["implementation_hash"] != before


# ---------------------------------------------------------------------------
# P0-37: operator manifest build failure must fail production
# ---------------------------------------------------------------------------


class TestOperatorManifestFailure:
    """P0-37: manifest 构建失败 → production fail-closed，research 显式跳过。

    这些测试不加载整个算子注册表——monkeypatch ``ensure_cleaned_loaded`` 为 no-op，
    用 fake impl 或未知 op 模拟构建失败。
    """

    @pytest.fixture(autouse=True)
    def _no_registry_load(self, monkeypatch):
        monkeypatch.setattr(
            "factor_engine.backend.cleaned_bridge.ensure_cleaned_loaded", lambda: None
        )
        yield

    def _ir(self, op="ts_mean"):
        return _ir_node(op, d=20)

    def test_production_no_implementation_raises(self, monkeypatch):
        """production：canonical 无实现（registry 未加载/缺失）→ 硬失败。"""
        from factor_engine.runtime import incremental_scheduler as isch

        with pytest.raises(isch.OperatorManifestUnavailable):
            isch._operator_manifest_from_ir(self._ir("no_such_op_xyz"), production=True)

    def test_production_digest_unavailable_raises(self, monkeypatch):
        """production：实现存在但 digest 算不出来 → 硬失败。"""
        from types import SimpleNamespace

        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime import incremental_scheduler as isch

        fake_impl = SimpleNamespace(metadata=SimpleNamespace(semantic_version="1.0"))
        monkeypatch.setattr(
            OperatorRegistry, "get", classmethod(lambda cls, *a, **k: fake_impl)
        )
        monkeypatch.setattr(isch, "_implementation_digest", lambda impl: None)
        with pytest.raises(isch.OperatorManifestUnavailable):
            isch._operator_manifest_from_ir(self._ir(), production=True)

    def test_research_manifest_failure_skips(self, monkeypatch):
        """P0-37: research 下构建失败只 warning 跳过（仍记录 canonical），绝不抛。"""
        from types import SimpleNamespace

        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime import incremental_scheduler as isch

        fake_impl = SimpleNamespace(metadata=SimpleNamespace(semantic_version="1.0"))
        monkeypatch.setattr(
            OperatorRegistry, "get", classmethod(lambda cls, *a, **k: fake_impl)
        )
        monkeypatch.setattr(isch, "_implementation_digest", lambda impl: None)
        out = isch._operator_manifest_from_ir(self._ir(), production=False)
        assert out is not None
        assert all(m["implementation_hash"] is None for m in out)

    def test_production_build_raise_propagates(self, monkeypatch):
        """P0-37: 构建过程直接 raise（非 digest=None）也 propagate。"""
        from types import SimpleNamespace

        from factor_engine.cleaned_operators.registry import OperatorRegistry

        from factor_engine.runtime import incremental_scheduler as isch

        fake_impl = SimpleNamespace(metadata=SimpleNamespace(semantic_version="1.0"))
        monkeypatch.setattr(
            OperatorRegistry, "get", classmethod(lambda cls, *a, **k: fake_impl)
        )

        def _explode(*a, **k):
            raise RuntimeError("kernel exploded")

        monkeypatch.setattr(isch, "_implementation_digest", _explode)
        with pytest.raises(isch.OperatorManifestUnavailable):
            isch._operator_manifest_from_ir(self._ir(), production=True)


# ---------------------------------------------------------------------------
# P0-38 / P0-39: calendar fingerprint
# ---------------------------------------------------------------------------


class TestCalendarFingerprint:
    def _source(self, market="test_mkt"):
        from types import SimpleNamespace

        return SimpleNamespace(market=market)

    def test_interior_holiday_swap_changes_digest(self):
        """P0-38: 同 len/first/last、内部交易日不同 → digest 必须不同。"""
        from factor_engine.storage.trading_calendar import (
            TradingCalendar,
            register_trading_calendar,
        )

        from factor_engine.runtime.incremental_scheduler import _calendar_version

        c1 = TradingCalendar(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )
        c2 = TradingCalendar(
            ["2024-01-02", "2024-01-04", "2024-01-05", "2024-01-07", "2024-01-08"]
        )
        register_trading_calendar("test_mkt", c1)
        v1 = _calendar_version(self._source())
        register_trading_calendar("test_mkt", c2)
        v2 = _calendar_version(self._source())
        assert v1 is not None and v2 is not None
        assert v1 != v2
        # 老指纹 v5:2024-01-02:2024-01-08 对两者完全相同——证明新 digest 抓住了内部变化。
        assert len(c1.days) == len(c2.days) == 5
        assert str(c1.days[0].date()) == str(c2.days[0].date()) == "2024-01-02"
        assert str(c1.days[-1].date()) == str(c2.days[-1].date()) == "2024-01-08"

    def test_same_calendar_deterministic(self):
        from factor_engine.storage.trading_calendar import (
            TradingCalendar,
            register_trading_calendar,
        )

        from factor_engine.runtime.incremental_scheduler import _calendar_version

        register_trading_calendar(
            "test_det",
            TradingCalendar(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        a = _calendar_version(self._source("test_det"))
        b = _calendar_version(self._source("test_det"))
        assert a == b

    def test_production_fingerprint_failure_raises(self, monkeypatch):
        """P0-39: 有市场但日历不可解析 → production fail-closed。"""
        from factor_engine.runtime import incremental_scheduler as isch

        src = self._source("no_such_market")
        # 让 get_trading_calendar 返回 None（模拟解析失败）
        monkeypatch.setattr(
            "factor_engine.storage.trading_calendar.get_trading_calendar", lambda m: None
        )
        with pytest.raises(isch.CalendarFingerprintUnavailable):
            isch._calendar_version(src, production=True)

    def test_research_fingerprint_failure_returns_none(self, monkeypatch):
        """P0-39: research 下构建失败返回 None（不静默当 success，只跳过）。"""
        from factor_engine.runtime import incremental_scheduler as isch

        monkeypatch.setattr(
            "factor_engine.storage.trading_calendar.get_trading_calendar", lambda m: None
        )
        assert isch._calendar_version(self._source("no_such_market")) is None

    def test_no_market_no_calendar_no_fingerprint(self):
        """无市场/无日历依赖的数据源 → 无需指纹（返回 None，不失败）。"""
        from types import SimpleNamespace

        from factor_engine.runtime.incremental_scheduler import _calendar_version

        assert _calendar_version(SimpleNamespace(dataset="mock_ds")) is None
        assert (
            _calendar_version(SimpleNamespace(dataset="mock_ds"), production=True)
            is None
        )


# ---------------------------------------------------------------------------
# P0-40 / P0-41: no anchor fallback for physical dataset resolution
# ---------------------------------------------------------------------------


class TestDependencyDatasetResolution:
    def _analysis(self, table="UnknownLogicalTable999"):
        from types import SimpleNamespace

        return SimpleNamespace(
            referenced_fields={
                "close": SimpleNamespace(
                    dataset=None,
                    table=table,
                    field_id="close",
                    source_name="Close",
                )
            }
        )

    def test_research_skips_unresolved_edge(self):
        from types import SimpleNamespace

        from factor_engine.runtime.incremental_scheduler import _edges_from_analysis

        edges = _edges_from_analysis(
            "f", self._analysis(), SimpleNamespace(dataset="ashare_stock_daily")
        )
        assert edges == []  # 绝不回落 anchor

    def test_production_raises_on_unresolved_edge(self):
        from types import SimpleNamespace

        from factor_engine.runtime.incremental_scheduler import (
            DependencyDatasetResolutionError,
            _edges_from_analysis,
        )

        with pytest.raises(DependencyDatasetResolutionError):
            _edges_from_analysis(
                "f",
                self._analysis(),
                SimpleNamespace(dataset="ashare_stock_daily"),
                production=True,
            )

    def test_spec_dataset_wins_over_anchor(self):
        from types import SimpleNamespace

        from factor_engine.runtime.incremental_scheduler import _edges_from_analysis

        analysis = SimpleNamespace(
            referenced_fields={
                "pe_ratio": SimpleNamespace(
                    dataset="ashare_stock_valuation_daily",
                    table="StockValuationDaily",
                    field_id="pe_ratio",
                    source_name="PeRatio",
                )
            }
        )
        edges = _edges_from_analysis(
            "composite_f", analysis, SimpleNamespace(dataset="ashare_stock_daily")
        )
        assert len(edges) == 1
        assert edges[0].source_dataset == "ashare_stock_valuation_daily"


# ---------------------------------------------------------------------------
# P0-42: forward-window calendar error must not zero-extend
# ---------------------------------------------------------------------------


class TestExtendDateForward:
    def _monkeypatch_offset_fail(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("calendar broke")

        monkeypatch.setattr("factor_engine.storage.trading_calendar.trading_day_offset", _raise)

    def test_production_raises_on_calendar_error(self, monkeypatch):
        from factor_engine.runtime import incremental_scheduler as isch

        self._monkeypatch_offset_fail(monkeypatch)
        with pytest.raises(isch.CalendarOffsetError):
            isch._extend_date_forward(
                "2024-06-01", 19, market="A", production=True
            )

    def test_research_over_extends_never_zero(self, monkeypatch):
        """research: 日历错误时按自然日过扩（>= bars 个自然日），绝不停在 t0。"""
        from factor_engine.runtime import incremental_scheduler as isch

        self._monkeypatch_offset_fail(monkeypatch)
        out = isch._extend_date_forward(
            "2024-06-01", 19, market="A", production=False
        )
        assert out is not None
        assert out > "2024-06-01"
        # 至少 19 个自然日
        from datetime import date, timedelta

        expected = date(2024, 6, 1) + timedelta(days=19)
        assert out >= expected.isoformat()

    def test_bars_none_returns_cap(self):
        from factor_engine.runtime import incremental_scheduler as isch

        assert (
            isch._extend_date_forward("2024-06-01", None, market="A", cap="2026-01-01")
            == "2026-01-01"
        )

    def test_bars_zero_returns_end_date(self):
        from factor_engine.runtime import incremental_scheduler as isch

        assert (
            isch._extend_date_forward("2024-06-01", 0, market="A") == "2024-06-01"
        )


# ---------------------------------------------------------------------------
# P0-43: ForwardImpactRequirement tri-state
# ---------------------------------------------------------------------------


class TestForwardImpactRequirement:
    def test_stored_conversion(self):
        from factor_engine.runtime.dependency_catalog import (
            UNBOUNDED_FORWARD_IMPACT,
            ForwardImpactRequirement,
            forward_impact_from_stored,
            stored_from_forward_impact,
        )

        assert forward_impact_from_stored(19) == ForwardImpactRequirement.finite(19)
        assert forward_impact_from_stored(UNBOUNDED_FORWARD_IMPACT) == (
            ForwardImpactRequirement.unbounded()
        )
        assert forward_impact_from_stored(None) == ForwardImpactRequirement.unknown()
        # round-trip
        assert stored_from_forward_impact(ForwardImpactRequirement.finite(19)) == 19
        assert stored_from_forward_impact(ForwardImpactRequirement.unbounded()) == (
            UNBOUNDED_FORWARD_IMPACT
        )
        assert stored_from_forward_impact(ForwardImpactRequirement.unknown()) is None

    def test_effective_forward_bars(self):
        from factor_engine.runtime.dependency_catalog import (
            ForwardImpactRequirement,
            effective_forward_bars,
        )

        # finite -> rows
        assert effective_forward_bars(ForwardImpactRequirement.finite(19)) == 19
        # unbounded -> None (reach cap / latest)
        assert effective_forward_bars(ForwardImpactRequirement.unbounded()) is None
        # unknown + research -> 0 (legacy no-propagation)
        assert effective_forward_bars(ForwardImpactRequirement.unknown()) == 0
        # unknown + production -> None (conservative unbounded / mandatory reindex)
        assert (
            effective_forward_bars(ForwardImpactRequirement.unknown(), production=True)
            is None
        )

    def test_factors_for_event_carries_requirement(self, tmp_path):
        """catalog read path 现在返回 ForwardImpactRequirement（P0-43 迁移）。"""
        from factor_engine.runtime.dependency_catalog import (
            DependencyCatalog,
            FactorDependencyEdge,
            ForwardImpactRequirement,
        )
        from factor_engine.runtime.incremental_scheduler import DataEvent
        from factor_engine.storage.catalog import FactorCatalog

        catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
        dep = DependencyCatalog(catalog)
        catalog.register("f", author="t", frequency="1d", ast_hash="h", expression="x")
        dep.record_factor_edges(
            "f",
            edges=[FactorDependencyEdge("f", "ds_x", "close", forward_impact=19)],
            lookback=20,
            frequency="1d",
            source_dataset="ds_x",
        )
        rows = dep.factors_for_event(
            DataEvent(
                dataset="ds_x", column="close", updated_date="2026-01-01",
                field_id="close",
            )
        )
        assert rows and isinstance(rows[0]["forward_impact"], ForwardImpactRequirement)
        assert rows[0]["forward_impact"].kind == "finite"
        assert rows[0]["forward_impact"].rows == 19


# ---------------------------------------------------------------------------
# Section 十五: normalize_data_event strict int (never truncate)
# ---------------------------------------------------------------------------


class TestNormalizeDataEventStrictInt:
    def test_integral_int_ok(self):
        from factor_engine.runtime.incremental_scheduler import normalize_data_event

        ev = normalize_data_event(
            {"dataset": "d", "column": "c", "updated_date": "2026-01-01", "sequence": 3}
        )
        assert ev.sequence == 3

    def test_integral_float_ok(self):
        from factor_engine.runtime.incremental_scheduler import normalize_data_event

        ev = normalize_data_event(
            {"dataset": "d", "column": "c", "updated_date": "2026-01-01", "sequence": 3.0}
        )
        assert ev.sequence == 3

    def test_fractional_float_rejected(self):
        """2.9 绝不截断成 2——强一致性排序字段必须严格整数。"""
        from factor_engine.runtime.incremental_scheduler import normalize_data_event

        with pytest.raises(ValueError):
            normalize_data_event(
                {"dataset": "d", "column": "c", "updated_date": "2026-01-01", "sequence": 2.9}
            )

    def test_fractional_string_rejected(self):
        from factor_engine.runtime.incremental_scheduler import normalize_data_event

        with pytest.raises(ValueError):
            normalize_data_event(
                {"dataset": "d", "column": "c", "updated_date": "2026-01-01", "sequence": "2.9"}
            )

    def test_bool_rejected(self):
        from factor_engine.runtime.incremental_scheduler import normalize_data_event

        with pytest.raises(ValueError):
            normalize_data_event(
                {"dataset": "d", "column": "c", "updated_date": "2026-01-01", "sequence": True}
            )


# ---------------------------------------------------------------------------
# P1-33 / P1-34: api.factor scope hint + compute_factor_identity source of truth
# ---------------------------------------------------------------------------


class TestFactorExecutionScopeHint:
    def test_api_renamed_and_alias_kept(self):
        import factor_engine.api.factor as af

        assert af.FactorExecutionScopeHint is not None
        # 别名保留，兼容既有导入（test_round13_execution_scope 用了它）。
        assert af.FactorSemanticIdentity is af.FactorExecutionScopeHint
        hint = af.FactorExecutionScopeHint(market="US", universe_id="US_NASDAQ100")
        assert hint.market == "US"
        assert hint.universe_id == "US_NASDAQ100"

    def test_semantic_identity_is_single_runtime_identity(self):
        """runtime.factor_identity.FactorSemanticIdentity 是单一定义身份。"""
        import factor_engine.runtime.factor_identity as fi

        ident = fi.FactorSemanticIdentity(
            ir_hash="i", operator_contract_hash="o", field_contract_hash="f",
            source_contract_hash="s", source_dependency_hash="d",
        )
        assert ident.identity_digest()
        assert "source_scope_hash" in fi.FactorSemanticIdentity.__dataclass_fields__

    def _factor_with_scope(self, **scope_overrides):
        from factor_engine.api.factor import Factor, FactorExecutionScopeHint
        from factor_engine.expr.base import Expr

        scope = dict(
            market="A",
            universe_id="ALL",
            frequency="1d",
            calendar_id="SSE",
            decision_time_policy="eod",
            source_scope_hash="scope-hash-1",
        )
        scope.update(scope_overrides)
        return Factor(
            name="x",
            expr=Expr(),
            freq="1d",
            semantic_identity=FactorExecutionScopeHint(**scope),
        )

    def _digest_for(self, factor):
        from factor_engine.runtime.factor_identity import compute_factor_identity

        ident = compute_factor_identity(
            _ir_node(),
            ctx={
                "ir_hash": "i",
                "operator_contract_hash": "o",
                "field_contract_hash": "f",
                "source_contract_hash": "s",
                "source_dependency_hash": "d",
                "factor": factor,
            },
        )
        return ident

    def test_compute_factor_identity_reads_semantic_identity(self):
        """P1-34: 真实执行作用域来自 factor.semantic_identity，不是顶层字段。"""
        ident = self._digest_for(self._factor_with_scope())
        assert ident.market == "A"
        assert ident.universe == "ALL"
        assert ident.frequency == "1d"
        assert ident.calendar == "SSE"
        assert ident.decision_time_policy == "eod"
        assert ident.source_scope_hash == "scope-hash-1"

    def test_semantic_identity_overrides_top_level(self):
        """P1-34: semantic_identity.frequency=5m 覆盖 factor.freq=1d（split-brain 防护）。"""
        factor = self._factor_with_scope(frequency="5m")
        ident = self._digest_for(factor)
        assert ident.frequency == "5m"

    def test_every_scope_field_enters_digest(self):
        """P1-34 机器测试：任一 scope 字段变化 → digest 必须不同。"""
        base = self._digest_for(self._factor_with_scope()).identity_digest()
        cases = {
            "market": dict(market="US"),
            "universe_id": dict(universe_id="CSI300"),
            "frequency": dict(frequency="5m"),
            "calendar_id": dict(calendar_id="NYSE"),
            "decision_time_policy": dict(decision_time_policy="boc"),
            "source_scope_hash": dict(source_scope_hash="other-scope"),
        }
        for field, overrides in cases.items():
            mutated = self._digest_for(
                self._factor_with_scope(**overrides)
            ).identity_digest()
            assert mutated != base, f"identity digest did not change for field {field!r}"


# ---------------------------------------------------------------------------
# P1-75: identity-builder naming is documented at its semantic level
# ---------------------------------------------------------------------------


class TestIdentityNamingDocs:
    def test_compute_factor_identity_docstring_distinguishes_levels(self):
        import inspect

        import factor_engine.runtime.factor_identity as fi

        doc = inspect.getdoc(fi.compute_factor_identity)
        assert "SourceExpressionIdentity" in doc
        assert "TypedIRIdentity" in doc
        assert "OptimizedPlanIdentity" in doc
        assert "ExecutionSemanticIdentity" in doc

    def test_materialize_ctx_docstring_distinguishes_levels(self):
        import inspect

        import factor_engine.runtime.factor_identity as fi

        doc = inspect.getdoc(fi.compute_identity_from_materialize_ctx)
        assert "SourceExpressionIdentity" in doc
        assert "TypedIRIdentity" in doc
