"""Phase 2-4 tests：FactorEngineAdapter + evaluate_many + identity factory。

- FE import 成功（quant_projects 在路径上）时跑 validate/canonicalize 小样例；
- FE 不可 import 时对应用例 pytest.skip（不依赖真实数据）；
- 纯本地 identity 测试：f / (-f) / ((f))*1 的 signal id / canonical_ast_hash 规则。
"""

from __future__ import annotations

import sys

import pytest

from alphaprobe.contracts import ExperimentContext
from alphaprobe.identity import FactorIdentityFactory, from_formula

# ---------------------------------------------------------------------------
# FE 可用性探测（不抛异常；失败只 skip FE 相关用例）
# ---------------------------------------------------------------------------


def _fe_importable() -> bool:
    try:
        import alphaprobe.fe_bridge.paths as _p

        _p.ensure_factor_engine_importable()
        import factor_engine.api  # noqa: F401

        return True
    except Exception:
        return False


FE_AVAILABLE = _fe_importable()


def _make_context() -> ExperimentContext:
    return ExperimentContext(
        run_id="test",
        round_id="r1",
        campaign_id="c1",
        data_snapshot_id="test",
        universe_snapshot_id="test",
    )


def _adapter():
    from alphaprobe.integration.factor_engine_adapter import FactorEngineAdapter

    return FactorEngineAdapter()


# ---------------------------------------------------------------------------
# FactorEngineAdapter.validate / canonicalize / inspect（FE 可用时）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FE_AVAILABLE, reason="factor_engine not importable")
class TestFactorEngineAdapter:
    def test_validate_ok(self):
        ok, msg = _adapter().validate("rank(ts_mean(close, 20))", surface="daily")
        assert ok is True
        assert msg == "OK"

    def test_validate_rejects_bad_syntax(self):
        ok, msg = _adapter().validate("rank((", surface="daily")
        assert ok is False
        assert "rank" in str(msg) or "syntax" in str(msg).lower() or "paren" in str(msg).lower()

    def test_validate_rejects_empty(self):
        ok, msg = _adapter().validate("", surface="daily")
        assert ok is False

    def test_canonicalize_fields(self):
        c = _adapter().canonicalize("rank(ts_mean(close, 20))")
        assert c.canonical_formula
        assert c.canonical_ast_hash
        assert c.signal_equivalence_id
        assert c.parameter_family_id
        assert len(c.canonical_ast_hash) == 32

    def test_canonicalize_mul_one_same_ast_hash(self):
        a = _adapter()
        c1 = a.canonicalize("rank(close)")
        c2 = a.canonicalize("(rank(close) * 1)")
        assert c1.canonical_ast_hash == c2.canonical_ast_hash

    def test_inspect_fields(self):
        ins = _adapter().inspect(_adapter().canonicalize("rank(ts_mean(close, 20))"))
        assert "close" in ins.field_set
        assert "rank" in ins.operator_set
        assert ins.complexity >= 3
        assert ins.lookback >= 19


# ---------------------------------------------------------------------------
# FE 不可用时的降级（本地括号平衡 + dedup）
# ---------------------------------------------------------------------------


class TestFactorEngineAdapterDegraded:
    def _force_no_fe(self, monkeypatch):
        import alphaprobe.integration.factor_engine_adapter as m

        monkeypatch.setattr(m, "_fe", lambda: None)
        return m.FactorEngineAdapter()

    def test_validate_paren_balance_fallback(self, monkeypatch):
        a = self._force_no_fe(monkeypatch)
        ok, msg = a.validate("rank(ts_mean(close, 20))", surface="daily")
        assert ok is True
        assert "fallback" in msg
        ok, msg = a.validate("rank((", surface="daily")
        assert ok is False
        assert "unbalanced" in msg

    def test_canonicalize_dedup_fallback(self, monkeypatch):
        a = self._force_no_fe(monkeypatch)
        c = a.canonicalize("((rank(close)))")
        assert c.canonical_formula == "rank(close)"
        assert c.fe_canonical == "rank(close)"

    def test_inspect_local(self, monkeypatch):
        a = self._force_no_fe(monkeypatch)
        ins = a.inspect(a.canonicalize("rank(ts_mean(close, 20))"))
        assert "close" in ins.field_set
        assert ins.lookback >= 19

    def test_run_many_no_executor_error_batch(self, monkeypatch):
        a = self._force_no_fe(monkeypatch)
        batch = a.run_many(["rank(close)"], _make_context())
        assert batch.factor_ids == []
        assert "error" in batch.coverage_summary


# ---------------------------------------------------------------------------
# run_many：executor 注入路径（不依赖真实数据）
# ---------------------------------------------------------------------------


class TestRunManyWithExecutor:
    def test_run_many_injects_executor(self, monkeypatch):
        import alphaprobe.integration.factor_engine_adapter as m

        calls: list[list[str]] = []

        def fake_executor(formulas: list[str]) -> dict:
            calls.append(list(formulas))
            return {"h1": "series-a", "h2": "series-b"}

        a = m.FactorEngineAdapter()
        a._executor = fake_executor
        batch = a.run_many(["rank(close)", "rank(volume)"], _make_context())
        assert len(batch.factor_ids) == 2
        assert batch.values_ref["h1"] == "series-a"
        assert batch.coverage_summary.get("n") == 2
        assert len(calls) == 1

    def test_run_many_error_batch_on_exception(self):
        from alphaprobe.integration.factor_engine_adapter import FactorEngineAdapter

        def boom(_formulas):
            raise RuntimeError("data missing")

        a = FactorEngineAdapter()
        a._executor = boom
        batch = a.run_many(["rank(close)"], _make_context())
        assert batch.factor_ids == []
        assert "run_many failed" in batch.coverage_summary["error"]


# ---------------------------------------------------------------------------
# identity factory：f / -f / ((f))*1 规则（纯本地，不依赖 FE）
# ---------------------------------------------------------------------------


class TestIdentityFactory:
    def test_from_formula_basic(self):
        ident = from_formula("rank(ts_mean(close, 20))")
        assert ident.factor_id
        assert ident.canonical_formula
        assert ident.orientation == 1
        assert len(ident.canonical_ast_hash) == 32

    def test_orientation_preserved(self):
        pos = from_formula("rank(close)", orientation=1)
        neg = from_formula("rank(close)", orientation=-1)
        assert pos.orientation == 1
        assert neg.orientation == -1
        assert pos.factor_id == neg.factor_id

    def test_signal_equivalence_f_and_neg_f(self):
        f = from_formula("rank(close)")
        nf = from_formula("(-(rank(close)))")
        assert f.signal_equivalence_id == nf.signal_equivalence_id

    def test_canonical_ast_hash_differs_f_and_neg_f(self):
        f = from_formula("rank(close)")
        nf = from_formula("(-(rank(close)))")
        assert f.canonical_ast_hash != nf.canonical_ast_hash

    def test_signal_equivalence_f_and_mul_one(self):
        # dedup.canonicalize_dsl 对 "(rank(close)*1)" 化简为 "rank(close)"，
        # 因此 signal id 相同；"((rank(close)))*1" 顶层乘 1 不化简 → 不同。
        f = from_formula("rank(close)")
        m1 = from_formula("(rank(close) * 1)")
        assert f.signal_equivalence_id == m1.signal_equivalence_id
        assert f.canonical_ast_hash == m1.canonical_ast_hash

    def test_parameter_family_window_invariant(self):
        f19 = from_formula("ts_mean(close, 19)")
        f21 = from_formula("ts_mean(close, 21)")
        assert f19.parameter_family_id == f21.parameter_family_id

    def test_factory_reuses_canonical(self):
        from alphaprobe.integration.factor_engine_adapter import FactorEngineAdapter

        canonical = FactorEngineAdapter().canonicalize("rank(close)")
        ident = FactorIdentityFactory().from_canonical(canonical)
        direct = from_formula("rank(close)")
        assert ident.canonical_ast_hash == direct.canonical_ast_hash

    def test_empty_formula_raises(self):
        with pytest.raises(ValueError):
            from_formula("  ")
