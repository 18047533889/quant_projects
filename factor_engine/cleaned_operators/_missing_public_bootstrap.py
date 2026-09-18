# -*- coding: utf-8 -*-
"""R56: safely surface the 34 public operators that are missing from the normal
production registry because their source modules are not part of the default
``BOOTSTRAP_MODULE_SPECS`` load list.

Why a dedicated module instead of just adding the 11 source modules to the load
list
---------------------------------------------------------------------------
The 34 operators are registered at *import time* of their source modules.  Those
modules are not in the default load list, so under a plain ``load_all()`` the 34
names are absent (``OperatorRegistry.get(name) is None``).  ``pytest`` only sees
them because ``tests/conftest.py`` pre-heats the modules during the building
window.

But three of the 11 source modules (notably ``common.polars_ts_basic``) also
register polars backends for *other*, already-existing operators (e.g.
``ts_argmax_age``, ``ts_coverage_ratio``, ...) with an *incomplete* param spec.
Because the canonical contract is first-registration-wins, importing those
modules during the building window overwrites the richer pandas reference
contract of those existing operators -- exactly the "move the preheat list into
startup" regression that breaks 20 existing winners.

This module imports the 11 source modules **during the building window** (it is
loaded by ``load_all`` before freeze), which registers the 34 target operators,
and then **reverts any collateral edit** those imports made to operators that
already existed beforehand.  The only acceptable net effect is adding the 34
public names; every pre-existing operator is restored to its pre-import
(exact, authoritative) state.

The 3 ``research_only`` names are deliberately excluded: they must never enter
the production registry and are reached only through the formal
``ResearchToolRegistry`` research entry point.

The snapshot/restore is field-level (param_specs / backends / status / aliases /
physical / backend_meta on the catalog entry, plus ``_operators`` / ``_aliases`` /
``_pending_aliases`` globals).  It is not a deep copy of the whole registry --
the registration side effects live inside those fields and are replaced (not
mutated in place) by the source modules, so restoring the authoritative field
references is sufficient and safe.
"""
from __future__ import annotations

import importlib
from factor_engine.cleaned_operators.registry import OperatorRegistry

# 34 public names that must be present on the production surface.
PUBLIC_ALLOW: frozenset[str] = frozenset(
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

# 3 research_only names that must NEVER enter the production registry here.
RESEARCH_ONLY: frozenset[str] = frozenset(
    {
        "ts_deviation_from_mean",
        "ts_jump_bipower",
        "ts_lag1_autocorr",
    }
)

# Source modules (relative to factor_engine.cleaned_operators) that define the 34
# public operators.  Importing them registers the 34; collateral edits to other
# operators are reverted by register_missing_public_ops().
_SOURCE_MODULES: tuple[str, ...] = (
    "factor_engine.cleaned_operators.panel_batch1",
    "factor_engine.cleaned_operators.time_semantic_gap",
    "factor_engine.cleaned_operators.fundamental.fiscal_logit_score_op",
    "factor_engine.cleaned_operators.common.fiscal_operators",
    "factor_engine.cleaned_operators.common.polars_ts_basic",
    "factor_engine.cleaned_operators.cross_section.panel_batch1",
    "factor_engine.cleaned_operators.intraday.state_space",
    "factor_engine.cleaned_operators.intraday.intra_state_space",
    "factor_engine.cleaned_operators.intraday.topology_manifold",
    "factor_engine.cleaned_operators.technical.adaptive_filters",
    "factor_engine.cleaned_operators.fundamental.fiscal_batch3",
)

_DONE = False


def register_missing_public_ops() -> None:
    """Register the 34 public operators and revert any collateral registry edits.

    Must be called while the registry is still in the *building* window (before
    freeze).  Safe to call once per process.
    """
    global _DONE
    if _DONE:
        return
    cat = OperatorRegistry._catalog
    ops = OperatorRegistry._operators

    # --- snapshot the authoritative pre-import state of every existing operator ---
    snap: dict[str, dict] = {}
    for c in list(cat):
        e = cat.get(c)
        if e is None:
            continue
        snap[c] = {
            "param_specs": dict(e.get("param_specs") or {}),
            "backends": list(e.get("backends") or []),
            "status": e.get("status"),
            "aliases": list(e.get("aliases") or []),
            "physical": e.get("physical"),
            "backend_meta": dict(e.get("backend_meta") or {}),
        }
    snap_ops = {c: dict(ops.get(c) or {}) for c in list(ops)}
    snap_aliases = dict(OperatorRegistry._aliases)
    snap_pending = list(OperatorRegistry._pending_aliases)

    # --- import the source modules (registers the 34 + collaterally edits some) ---
    for mod in _SOURCE_MODULES:
        importlib.import_module(mod)

    # --- revert collateral edits to pre-existing operators (keep only the 34) ---
    for c, s in snap.items():
        if c in PUBLIC_ALLOW:
            continue
        e = cat.get(c)
        if e is None:
            continue
        if e.get("param_specs") != s["param_specs"]:
            e["param_specs"] = s["param_specs"]
        if list(e.get("backends") or []) != s["backends"]:
            e["backends"] = s["backends"]
        if e.get("status") != s["status"]:
            e["status"] = s["status"]
        if list(e.get("aliases") or []) != s["aliases"]:
            e["aliases"] = s["aliases"]
        if e.get("physical") is not s["physical"]:
            e["physical"] = s["physical"]
        if dict(e.get("backend_meta") or {}) != s["backend_meta"]:
            e["backend_meta"] = s["backend_meta"]
    for c, s in snap_ops.items():
        if c in PUBLIC_ALLOW:
            continue
        if dict(ops.get(c) or {}) != s:
            ops[c] = s
    if dict(OperatorRegistry._aliases) != snap_aliases:
        OperatorRegistry._aliases = dict(snap_aliases)
    if list(OperatorRegistry._pending_aliases) != snap_pending:
        OperatorRegistry._pending_aliases = list(snap_pending)

    _DONE = True


# Auto-run while the registry is still in the building window (loaded by
# load_all before freeze).  Idempotent via _DONE.
register_missing_public_ops()
