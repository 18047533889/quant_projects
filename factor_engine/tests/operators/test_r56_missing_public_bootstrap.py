# -*- coding: utf-8 -*-
"""R56 acceptance: the 34 public operators that were missing from the normal
production surface must be safely registered without altering any existing
contract, alias, status, physical spec, or backend winner.

The check runs in a FRESH subprocess (no pytest conftest pre-heat) so it
validates the real production startup path (``load_all`` -> the R56 bootstrap
module appended to ``BOOTSTRAP_MODULE_SPECS``).  The structural dump is compared
against ``evidence/r56-missing-ops-baseline.json`` (the pre-fix surface) so the
only acceptable difference is exactly the 34 intended additions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_VENV_PY = _REPO / ".venv" / "bin" / "python"
_BASELINE = _REPO / "evidence" / "r56-missing-ops-baseline.json"

PUBLIC_ALLOW = frozenset(
    {
        "event_window_return_asof",
        "financial_snapshot_lag",
        "fiscal_capital_stock",
        "fiscal_delta",
        "fiscal_logit_score",
        "fiscal_rolling_slope",
        "fundamental_staleness_days",
        "intra_covariance_manifold_shift",
        "intra_critical_transition_score",
        "intra_dmd_koopman_features",
        "intra_functional_motif_score",
        "intra_kalman_latent_price",
        "intra_market_profile_corr_ex_self",
        "intra_matrix_profile_session_features",
        "intra_price_peak_ridge_valley_state",
        "intra_smart_money_fcm_score",
        "intra_state_space_volume_components",
        "intra_visibility_graph_features",
        "intra_volume_peak_ridge_valley_state",
        "intraday_value_at_extreme_state",
        "laborforce_efficiency",
        "panel_day_night_beta_gap",
        "pastor_stambaugh_beta",
        "price_delay_score",
        "report_asof",
        "same_calendar_day_mean",
        "same_calendar_month_return",
        "ts_mcginley_dynamic",
        "ts_nlms_filter",
        "ts_one_euro_filter",
        "ts_returns",
        "ts_rls_filter",
        "ts_vidya",
        "years_since_date",
    }
)

RESEARCH_ONLY = frozenset(
    {
        "ts_deviation_from_mean",
        "ts_jump_bipower",
        "ts_lag1_autocorr",
    }
)

# Dumper executed in a fresh interpreter: plain load_all(), no conftest pre-heat.
_DUMPER = '''
import sys, os, json
repo = os.environ["R56_REPO"]
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, "factor_engine"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import factor_engine.cleaned_operators as _pkg
from factor_engine.cleaned_operators.registry import OperatorRegistry

def pf(s):
    if s is None:
        return None
    return {
        "dtype": str(getattr(s, "dtype", None)),
        "min": getattr(s, "min", None),
        "max": getattr(s, "max", None),
        "choices": [str(c) for c in (getattr(s, "choices", None) or [])],
        "default": getattr(s, "default", None),
        "param_role": str(getattr(s, "param_role", None)),
        "role_source": str(getattr(s, "role_source", None)),
    }

def dump():
    cat = OperatorRegistry._catalog
    o = {}
    for c, e in cat.items():
        ps = e.get("param_specs") or {}
        o[c] = {
            "backends": sorted(e.get("backends") or []),
            "status": str(e.get("status")),
            "aliases": sorted(e.get("aliases") or []),
            "params": {k: pf(v) for k, v in ps.items()},
            "physical": repr(e.get("physical")) if e.get("physical") is not None else None,
            "backend_meta_keys": sorted((e.get("backend_meta") or {}).keys()),
        }
    o["__globals__"] = {
        "aliases": dict(OperatorRegistry._aliases),
        "operators": {c: sorted(v.keys()) for c, v in OperatorRegistry._operators.items()},
    }
    return o

_pkg.load_all()
json.dump(dump(), open(sys.argv[1], "w"), sort_keys=True, default=str)
'''


def _dump_fixed(path: str) -> None:
    env = dict(os.environ)
    env["R56_REPO"] = str(_REPO)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "2"
    env["PYTHONPATH"] = str(_REPO)
    proc = subprocess.run(
        [_VENV_PY, "-c", _DUMPER, path],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "fresh-subprocess load_all failed (rc=%d):\n%s\n%s"
            % (proc.returncode, proc.stdout, proc.stderr)
        )


def _fp(e):
    return (
        e.get("backends"),
        e.get("status"),
        e.get("aliases"),
        e.get("physical"),
        e.get("backend_meta_keys"),
        {k: tuple(sorted(v.items())) for k, v in (e.get("params") or {}).items()},
    )


def test_r56_missing_public_bootstrap_surface():
    if not _BASELINE.exists():
        pytest.skip(f"baseline evidence missing: {_BASELINE}")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        cand_path = tf.name
    try:
        _dump_fixed(cand_path)
        cand_raw = json.load(open(cand_path))
        base_raw = json.load(open(_BASELINE))
    finally:
        if os.path.exists(cand_path):
            os.remove(cand_path)

    base = {k: v for k, v in base_raw.items() if k != "__globals__"}
    cand = {k: v for k, v in cand_raw.items() if k != "__globals__"}

    added = sorted(set(cand) - set(base))
    removed = sorted(set(base) - set(cand))

    # 1) Exactly the 34 public names are added; nothing else, nothing missing.
    assert added == sorted(PUBLIC_ALLOW), (
        "added set != expected allowlist.\n"
        f"  added_not_in_allow={sorted(set(added) - PUBLIC_ALLOW)}\n"
        f"  allow_missing={sorted(PUBLIC_ALLOW - set(added))}"
    )
    assert removed == [], f"operators removed by the fix: {removed}"

    # 2) The 3 research_only names must NEVER enter the production registry.
    leaked = sorted(RESEARCH_ONLY & set(cand))
    assert leaked == [], f"research_only leaked into production registry: {leaked}"

    # 3) Every pre-existing operator's contract is byte-for-byte unchanged.
    changed = []
    for c in base:
        if c in PUBLIC_ALLOW:
            continue
        if _fp(base[c]) != _fp(cand[c]):
            changed.append(c)
    assert changed == [], (
        "pre-existing contracts changed by the fix: %s" % changed
    )

    # 4) Global alias table and existing operators' backend sets unchanged.
    gb = base_raw.get("__globals__", {})
    gc = cand_raw.get("__globals__", {})
    assert gb.get("aliases") == gc.get("aliases"), "global _aliases changed"
    base_ops = gb.get("operators", {})
    cand_ops = gc.get("operators", {})
    op_changed = [
        c for c in set(base_ops) & set(cand_ops)
        if c not in PUBLIC_ALLOW and base_ops[c] != cand_ops[c]
    ]
    assert op_changed == [], (
        "pre-existing operators' backend sets changed: %s" % op_changed
    )
