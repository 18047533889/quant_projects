# -*- coding: utf-8 -*-
"""Tests for PERF-2 vector coverage generator + telemetry counters (GO §6.1/§6.3)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk
from factor_engine.cleaned_operators.intraday import perf_vec_coverage as pvc
from factor_engine.cleaned_operators.intraday import perf_vec_telemetry as pvt
from factor_engine.cleaned_operators.intraday import _core


@pytest.fixture()
def _reset_telemetry():
    """Reset counter mutable state used by other tests (independent of bind)."""
    # No-op guard: keep the global counters monotonic so a single process never
    # asserts a specific absolute value; each test asserts *deltas* instead.
    yield


def test_bind_whitelist_runs_and_fail_closed() -> None:
    ids = list(pvk.bind_whitelist())
    bound = int(pvk.count_bound())
    # fail-closed: a lost bind must raise, not silently pass the report.
    assert bound == len(ids), "count_bound must equal len(bind_whitelist())"


def test_coverage_schema_and_bound_unbound_disjoint() -> None:
    cov = pvc.build_vector_coverage()
    assert cov["schema"] == "intraday_vector_coverage/v1"
    assert cov["bound_count"] == int(pvk.count_bound())
    assert set(cov) >= {"version", "generated_at", "bound_count", "operators", "summary"}

    ops = cov["operators"]
    assert ops, "coverage must enumerate at least the whitelisted operators"
    assert cov["summary"]["total"] == len(ops)
    assert cov["summary"]["vectorized"] + cov["summary"]["scalar_only"] == len(ops)

    bound_req = {"canonical", "scalar_impl", "vector_impl", "vector_bound",
                 "vector_equivalence", "fallback_reason"}
    for o in ops:
        assert bound_req <= set(o)
        if o["vector_bound"]:
            assert o["vector_impl"], "vectorized op must name a vector impl"
            assert o["fallback_reason"] is None
            assert o["vector_equivalence"].startswith("harness-proven")
        else:
            assert o["vector_impl"] is None
            assert o["fallback_reason"], "scalar-only op must state a fallback_reason"
            assert o["vector_equivalence"] == "unproven"

    bound_canons = {o["canonical"] for o in ops if o["vector_bound"]}
    scalar_canons = {o["canonical"] for o in ops if not o["vector_bound"]}
    assert bound_canons.isdisjoint(scalar_canons), "bound/unbound sets must be disjoint"


def test_coverage_reports_current_whitelist() -> None:
    cov = pvc.build_vector_coverage()
    bound_ids = list(pvk.bind_whitelist())
    # The report's vectorized operators should cover the live whitelist canonicals.
    bound_canons = {o["canonical"] for o in cov["operators"] if o["vector_bound"]}
    for bid in bound_ids:
        _mod, op = pvc._parse_bound_id(bid)
        canon = op.lstrip("intra_")
        # canonical may be either the raw operator name or without the intra_ prefix
        assert (op in bound_canons) or (op[6:] in bound_canons) or (
            op.replace("_", "") in {c.replace("_", "") for c in bound_canons}
        ), f"whitelist id {bid} not surfaced as a vectorized operator"


def test_markdown_render_has_stats() -> None:
    cov = pvc.build_vector_coverage()
    md = pvc.render_intraday_markdown(cov)
    assert "# Intraday Vector Coverage" in md
    assert "vector_bound" in md
    assert f'total={cov["summary"]["total"]}' in md
    assert f'vectorized={cov["summary"]["vectorized"]}' in md
    assert md.count("|") > 8  # header + separator + at least one row of cells


def test_script_writes_json_and_md(tmp_path: Path) -> None:
    out_json = tmp_path / "intraday_vector_coverage.json"
    repo_root = str(Path(__file__).parents[1])
    prog = (
        "import sys, pathlib; "
        f"sys.path.insert(0, {repo_root!r}); "
        "sys.path.insert(0, str(pathlib.Path('.').resolve() / 'scripts')); "
        "from generate_intraday_vector_coverage import main; "
        f"sys.exit(main(['--output', r'{out_json}']))"
    )
    # Invoke via subprocess so the script's own sys.path bootstrap is exercised.
    proc = subprocess.run(
        [sys.executable, "-c", prog],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_json.is_file(), "json artifact not written"
    md = out_json.parent / f"{out_json.stem.upper()}.md"
    assert md.is_file(), "markdown artifact not written"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["schema"] == "intraday_vector_coverage/v1"
    assert data["summary"]["total"] > 0


def test_telemetry_counters_increment_from_scalar_fallback() -> None:
    pvt.set_vector_kernels_bound(0)
    # A fresh (unbound) kernel must always take the scalar path -> counter bumps.
    import numpy as np
    import pandas as pd

    idx = pd.DatetimeIndex(["2026-01-05 09:30", "2026-01-05 09:31", "2026-01-06 09:30"])
    fr = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    before = pvt.get_telemetry_snapshot()["intraday_scalar_fallback_count"]
    _core.daily_agg(fr, lambda v, t: float(np.mean(v)), min_finite=1)
    after = pvt.get_telemetry_snapshot()["intraday_scalar_fallback_count"]
    # The counter bumps once per scalar-fallback call to `daily_agg`.
    assert after == before + 1


def test_telemetry_snapshot_keys_and_mutators() -> None:
    snapshot = pvt.get_telemetry_snapshot()
    assert set(snapshot) == {
        "intraday_vector_kernels_bound",
        "intraday_vector_bind_failures",
        "intraday_scalar_fallback_count",
    }
    pvt.set_vector_kernels_bound(5)
    assert pvt.get_telemetry_snapshot()["intraday_vector_kernels_bound"] == 5
    pvt.increment_bind_failure()
    pvt.increment_scalar_fallback()
    snap = pvt.get_telemetry_snapshot()
    assert snap["intraday_vector_bind_failures"] >= 1
    assert snap["intraday_scalar_fallback_count"] >= 1
