# -*- coding: utf-8
"""R10 #7 regression tests: ``assert_no_production_pandas_fallbacks`` — the hard
gate that already existed in production_policy — must actually be INVOKED on the
engine ``run()`` path after backend execute (previously it was only logged).

Two angles:
* ``test_research_run_invokes_gate_no_raise`` — end-to-end research run: the gate
  is invoked on the path (proves the wiring) and research is unaffected.
* Production semantics of the gate itself: with ``run_mode=production`` and a
  recorded pandas fallback it raises ``ProductionPolicyViolation``; with
  ``production_fallback_policy='warn'`` it allows the result through.

NOTE on pre-existing blockers (not caused by this change): a full production
``run()`` currently cannot complete in this tree — the operator catalog reports
core operators (ts_mean/rank) as ``research`` tier and the production Analyzer
rejects raw columns without a registered FieldSpec.  Both are separate,
pre-existing issues (operator tier / typed-field certification), so the
production behavior of the gate is tested directly rather than through a full
production run.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import ts_mean
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.production_policy import ProductionPolicyViolation
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


@pytest.fixture
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    open_ = pd.Series([9.0, 10.0, 19.0, 20.0, 9.5, 11.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "open": open_})


def _factor():
    return Factor(
        name="f1",
        expr=ts_mean(col("close"), 3),
        source_expr="ts_mean(col('close'), 3)",
    )


def _ctx_with_fallback(production_fallback_policy: str = "error"):
    """Build an ExecutionContext carrying a recorded pandas fallback."""
    from factor_engine.backend.context import ExecutionContext

    # data_source is required by the dataclass; a trivial stand-in is fine for
    # the gate (it only reads run_mode / runtime_stats / production_fallback_policy).
    class _Sink:
        pass

    ctx = ExecutionContext(
        data_source=_Sink(),
        run_mode="production",
        production_fallback_policy=production_fallback_policy,
        runtime_stats={
            "production_pandas_fallbacks": [
                {"op": "ts_mean", "backend": "pandas", "reason": "simulated"}
            ]
        },
    )
    return ctx


def test_research_run_invokes_gate_no_raise(_loaded, source, monkeypatch):
    # ``run()`` imports the gate from runtime.production_policy at call time, so
    # patching the gate there proves the engine path actually invokes it.
    import factor_engine.runtime.production_policy as pp

    called: list[str] = []
    orig = pp.assert_no_production_pandas_fallbacks

    def _recording(ctx, *, mode=None, context=None):
        # the gate is invoked from BOTH engine.run() (mode=) and
        # polars_long_backend post-execute (context= only) — accept either.
        orig(ctx, mode=mode, context=context or "execute")
        called.append(str(context))

    monkeypatch.setattr(pp, "assert_no_production_pandas_fallbacks", _recording)
    engine = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = engine.run(_factor())
    assert called, "gate must be invoked on the run() path"
    assert "result" in out


def test_gate_raises_on_production_fallback():
    from factor_engine.runtime.production_policy import assert_no_production_pandas_fallbacks

    with pytest.raises(ProductionPolicyViolation):
        assert_no_production_pandas_fallbacks(_ctx_with_fallback())


def test_gate_allows_warn_policy():
    from factor_engine.runtime.production_policy import assert_no_production_pandas_fallbacks

    # warn policy: no raise, but the fallback is still surfaced on the ctx.
    assert_no_production_pandas_fallbacks(
        _ctx_with_fallback(production_fallback_policy="warn")
    )


def test_gate_research_mode_ignores_fallback():
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.runtime.production_policy import assert_no_production_pandas_fallbacks

    class _Sink:
        pass

    ctx = ExecutionContext(
        data_source=_Sink(),
        run_mode="research",
        runtime_stats={"production_pandas_fallbacks": [{"op": "ts_mean"}]},
    )
    assert_no_production_pandas_fallbacks(ctx)  # research: no raise
