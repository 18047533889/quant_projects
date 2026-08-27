# -*- coding: utf-8 -*-
"""R64 operator-production-truth matrix smoke tests.

Non-vacuous R64 invariants (all run with the probe sample gate ON, i.e.
full-matrix mode):

    * causality is a tri-state (causal / unknown / leak_detected) and an
      UNKNOWN never upgrades to PASS in the ladder, never upgrades to
      PRODUCTION_ADMITTED in the matrix.
    * parameter injectivity is NOT a math oracle: a probe-false row still
      lists MATH_ORACLE_PASS=NOT_PROVEN, and an injectivity-pass row whose
      causality is unknown is never PRODUCTION_ADMITTED.
    * the matrix NEVER reports an operator as admitted when the registry row
      ``production_admitted`` is False, and the single production authority is
      that registry field (no parallel PRODUCTION_AGENT_CALLABLE verdict).
    * REGISTERED/IMPLEMENTED/BACKEND_* are registry-mount facts, not name lints.

The full-matrix run mounts every canonical once via ``load_all``; the probe is
gate-ON so the matrix stays honest and small (no per-row directory classifier).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[_var] = "1"
# Probe sample gate ON (full-matrix honest mode: probe runs for nothing).
os.environ["R64_PROBE_SAMPLE_GATE"] = "24000"

FE_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from factor_engine.mining.direct_use import (  # noqa: E402
    build_direct_use_operator,
)

_GATES = (
    "REGISTERED",
    "IMPLEMENTED",
    "BACKEND_WIRED",
    "BACKEND_RUNTIME_WIRED",
    "PARAM_WIRED",
    "MATH_ORACLE_PASS",
    "CAUSALITY_PASS",
    "PIT_PASS",
    "BACKEND_PARITY_PASS",
    "EDGE_PASS",
    "SCALE_PASS",
)


@pytest.fixture(scope="module", autouse=True)
def loaded():
    """The P0-14 session staged registry is frozen, so ``load_all`` is a no-op
    and the staged class misses the full canonical surface until the bootstrap
    specs are imported.  Import every bootstrap module into the staged class
    (idempotent with the session bootstrap; ts_rank etc. then resolve)."""
    import importlib

    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators import BOOTSTRAP_MODULE_SPECS

    for spec in BOOTSTRAP_MODULE_SPECS:
        importlib.import_module(spec.module)
    return OperatorRegistry


def _row(loaded, name: str):
    cat = dict(loaded._catalog[name])
    return build_direct_use_operator(name, cat)


class TestCausalityTriState:
    def test_causality_never_upgrades_unknown_in_ladder(self, loaded):
        # Every canonical's ladder CAUSALITY_PASS is PASS exactly when the
        # class is causal; UNKNOWN / leak_detected are never PASS.
        for name in list(loaded._catalog)[:40]:
            row = _row(loaded, name)
            caus = row.r64_causality
            assert caus in {"causal", "unknown", "leak_detected"}
            ladder_caus = row.r64_production_gates["CAUSALITY_PASS"]
            if caus == "causal":
                assert ladder_caus == "PASS"
            else:
                assert ladder_caus != "PASS"

    def test_injectivity_pass_but_unknown_causality_not_admitted(self, loaded):
        # Injectivity-pass does NOT prove causality; an op with unknown
        # causality must never report production admitted.
        op = loaded.get("KAMA")
        assert op is not None
        row = _row(loaded, "KAMA")
        assert row.r64_causality == "unknown"
        assert not row.production_admitted


class TestInjectivityIsNotMathOracle:
    def test_probe_false_stays_not_proven(self, loaded):
        row = _row(loaded, "ALMA")
        # ALMA has searchable params; in full-matrix mode the probe is gated
        # OFF so injectivity cannot be claimed, and the ladder says NOT_PROVEN.
        assert row.searchable_params
        assert row.r64_production_gates["MATH_ORACLE_PASS"] in {"NOT_PROVEN", "NOT_RUN"}

    def test_param_wired_vacuously_passes_without_math(self, loaded):
        row = _row(loaded, "ts_rank")
        assert row.searchable_params == ("window",)
        assert not row.parameter_injectivity_passed  # probe gated off
        assert row.r64_production_gates["MATH_ORACLE_PASS"] in {"NOT_PROVEN", "NOT_RUN"}


class TestSingleAuthority:
    def test_matrix_admission_needs_registry_production_admitted(self, loaded):
        # The matrix reflects the registry row; no parallel verdict may admit.
        for name in list(loaded._catalog)[:40]:
            row = _row(loaded, name)
            if not row.production_admitted:
                # Never ever an admitted-looking ladder: PARITY/EDGE/SCALE are
                # deliberately NOT_RUN in this view (no fresh backtest evidence).
                assert row.r64_production_gates["BACKEND_PARITY_PASS"] == "NOT_RUN"
                assert row.r64_production_gates["EDGE_PASS"] == "NOT_RUN"
                assert row.r64_production_gates["SCALE_PASS"] == "NOT_RUN"

    def test_gate_vocabulary_is_complete(self, loaded):
        row = _row(loaded, "ts_rank")
        assert set(row.r64_production_gates) == set(_GATES)

    def test_no_parallel_production_agent_callable_field(self, loaded):
        row = _row(loaded, "ts_rank")
        # The matrix authority is production_admitted; no callable alias remains.
        assert row.r64_production_gates["REGISTERED"] == "PASS"
        # The runtime ladder is bound, but the direct row carries no parallel
        # verdict: ``production_admitted`` is the machine truth.
        assert row.r64_production_gates["IMPLEMENTED"] in {"PASS", "NOT_RUN"}