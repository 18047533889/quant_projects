"""Compile isolation must not masquerade as factor execution."""
from types import SimpleNamespace
import pytest
import smoke_catalog as smoke


def test_bad_compile_does_not_remove_valid_siblings_or_claim_execution():
    factors = [SimpleNamespace(name=n) for n in ("good1", "bad", "good2")]
    rows = {f.name: {} for f in factors}
    calls = []
    class Engine:
        def compile(self, factor):
            calls.append(factor.name)
            if factor.name == "bad":
                raise ValueError("invalid window")
    ready = smoke.preflight_factors(
        Engine(), factors, rows, timeout_seconds=2, deadline_mode="legacy-cooperative")
    assert [f.name for f in ready] == ["good1", "good2"]
    assert calls == ["good1", "bad", "good2"]
    assert rows["bad"]["status"] == "COMPILE_FAILED"
    assert rows["bad"]["error_type"] == "ValueError"
    assert rows["bad"]["error"] == "invalid window"
    assert all("status" not in rows[n] for n in ("good1", "good2"))
    assert all(rows[n]["compile_status"] == "COMPILED" for n in ("good1", "good2"))


def test_compile_timeout_has_finite_distinct_terminal():
    factor = SimpleNamespace(name="slow")
    rows = {"slow": {}}
    class Engine:
        def compile(self, factor):
            raise smoke.FactorTimeout("compile timed out")
    assert smoke.preflight_factors(
        Engine(), [factor], rows, timeout_seconds=2, deadline_mode="legacy-cooperative") == []
    assert rows["slow"]["status"] == "COMPILE_TIMEOUT"


def test_compile_interrupt_is_not_swallowed():
    class Engine:
        def compile(self, factor):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        smoke.preflight_factors(
            Engine(), [SimpleNamespace(name="a")], {"a": {}}, timeout_seconds=2,
            deadline_mode="legacy-cooperative")


def test_real_preflight_rejects_invalid_without_any_data_reads():
    from compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    parser, engine = build_runtime()
    factors = [Factor(name=n, expr=parser.parse(dsl), source_expr=dsl, surface="compat_research")
               for n, dsl in (("valid", "ts_mean(close, 5)"), ("invalid", "ts_mean(close, 0)"))]
    rows = {f.name: {} for f in factors}
    ready = smoke.preflight_factors(
        engine, factors, rows, timeout_seconds=15, deadline_mode="legacy-cooperative")
    assert [f.name for f in ready] == ["valid"]
    assert rows["invalid"]["status"] == "COMPILE_FAILED"
    assert "status" not in rows["valid"]
