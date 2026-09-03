"""P0#6 + P0#7: run_many enforcement/canonicalization + CSE/fusion/fallback telemetry.

Covers:
- ``_canonicalize_batch_factors`` collapses structural duplicates and reports
  requested/unique/duplicates via runmany telemetry; unique-name rule preserved.
- ``telemetry.execution_telemetry`` counters increment on CSE hit, fusion
  admission, and forced fallback; ``snapshot`` / ``reset`` work.
- run() (single-op) counters its event (agent-facing batch caller visibility).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.planner.cse import apply_cse
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.batch_service import _canonicalize_batch_factors
from factor_engine.telemetry import execution_telemetry as et
from factor_engine.planner.native_fusion import _emit_fusion_telemetry
from tests.helpers import InMemorySeriesSource


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """Isolate counters between tests."""
    et.reset()
    yield
    et.reset()


def _sub_expr():
    return ts_mean(col("close"), 2)


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {"close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx)}


def _engine() -> FactorEngine:
    src = InMemorySeriesSource(data=_data())
    return FactorEngine(backend=PandasBackend(), data_source=src)


# ---------------------------------------------------------------------------
# P0#7: telemetry snapshot / reset / counters.
# ---------------------------------------------------------------------------


def test_counter_inc_and_snapshot():
    et.record("cse", "candidates", 3)
    et.record("cse", "shared_nodes", 1)
    snap = et.snapshot()
    assert snap["cse.candidates"] == 3
    assert snap["cse.shared_nodes"] == 1


def test_reset_clears_snapshot():
    et.record("fusion", "planned")
    assert et.get("fusion", "planned") == 1
    et.reset()
    assert et.get("fusion", "planned", 99) == 99
    assert et.snapshot() == {}


# ---------------------------------------------------------------------------
# P0#6: canonicalization.
# ---------------------------------------------------------------------------

def test_canonicalize_collapses_structural_duplicates():
    f1 = Factor(name="a", expr=_sub_expr())
    f2 = Factor(name="b", expr=_sub_expr())  # same expr, different name
    eng = _engine()
    out = _canonicalize_batch_factors(eng, [f1, f2], context="test")
    # distinct names => both kept (name is the unique identity).
    assert {f.name for f in out} == {"a", "b"}
    assert et.get("runmany", "requested") == 2
    assert et.get("runmany", "unique") == 2


def test_canonicalize_collapses_exact_name_expr_duplicate():
    f1 = Factor(name="a", expr=_sub_expr())
    f2 = Factor(name="a", expr=_sub_expr())  # exact duplicate (same name+expr)
    eng = _engine()
    out = _canonicalize_batch_factors(eng, [f1, f2], context="test")
    assert len(out) == 1
    assert et.get("runmany", "requested") == 2
    assert et.get("runmany", "unique") == 1
    assert et.get("runmany", "duplicates") == 1


def test_canonicalize_collapses_canonical_equal_but_textually_distinct():
    # §17: formula canonicalization at run_many entry. ``add(x, 0)`` simplifies
    # to ``x``, so same-name factors that only differ in this algebraic wrapper
    # must dedup to ONE root (old str(expr) key kept both).
    from factor_engine.api import add

    def _wrapped():
        return add(_sub_expr(), 0)  # ts_mean(close,2) + 0 == ts_mean(close,2)

    f1 = Factor(name="a", expr=_sub_expr())
    f2 = Factor(name="a", expr=_wrapped())
    eng = _engine()
    out = _canonicalize_batch_factors(eng, [f1, f2], context="test")
    assert len(out) == 1
    assert et.get("runmany", "requested") == 2
    assert et.get("runmany", "unique") == 1
    assert et.get("runmany", "duplicates") == 1



def test_canonicalize_empty_is_identity():
    eng = _engine()
    assert _canonicalize_batch_factors(eng, [], context="test") == []


# ---------------------------------------------------------------------------
# P0#7: CSE counters fire on structural reuse.
# ---------------------------------------------------------------------------

def test_cse_telemetry_fires_on_reuse():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.lowerer import Lowerer
    from factor_engine.planner.optimizer import Optimizer
    from factor_engine.ir.analyzer import Analyzer

    et.reset()
    expr = ts_mean(col("close"), 2)
    ir = Analyzer().lower(expr).ir
    plan = Optimizer().optimize(Lowerer().to_logical_plan(ir))
    roots, shared = apply_cse([plan, plan])
    assert len(shared) >= 1
    snap = et.snapshot()
    assert snap.get("cse.candidates", 0) >= 1
    assert snap.get("cse.shared_nodes", 0) >= 1
    assert snap.get("cse.reuse_edges", 0) >= 1


def test_engine_run_many_survives_and_emits_runmany_telemetry():
    et.reset()
    eng = _engine()
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=rank(ts_mean(col("close"), 2)))
    out = eng.run_many([f1, f2])
    assert "a" in out["results"] and "b" in out["results"]
    assert et.get("runmany", "factors") == 2


def test_run_single_op_counts_visibility_event():
    et.reset()
    eng = _engine()
    f = Factor(name="a", expr=ts_mean(col("close"), 2))
    r = eng.run(f)["result"]
    assert r is not None
    assert et.get("runmany", "single_run_events", 0) == 1


def test_fusion_telemetry_emit_handles_counters():
    et.reset()
    _emit_fusion_telemetry(
        {
            "native_fusion_planned": 2,
            "native_fusion_executed": 1,
            "native_fusion_binary_split": 0,
            "native_fusion_fallback": 1,
            "native_fusion_roots_total": 4,
        }
    )
    snap = et.snapshot()
    assert snap["fusion.planned"] == 2
    assert snap["fusion.executed"] == 1
    assert snap["fusion.fallback"] == 1
    assert snap["fusion.fallback_events"] == 1
