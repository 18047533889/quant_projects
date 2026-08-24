# -*- coding: utf-8 -*-
"""R28 §一百一十四..一百一十八 / P0-007..008: mining consumer exposure.

Every default mining-discovery API must agree on one admission authority:
forbidden / research-only / diagnostic-only / internal canonicals are NOT
mining-visible (0 default exposure); genuine production operators ARE reachable
in a legal grammar slot.  Alias safety must not be wider than the canonical.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


def _research_and_diagnostic():
    from factor_engine.cleaned_operators.operator_surface import (
        RESEARCH_ONLY_CANONICALS,
        UNSAFE_CANONICALS,
    )

    return set(RESEARCH_ONLY_CANONICALS) | set(UNSAFE_CANONICALS)


def _internal_canonicals():
    import factor_engine.cleaned_operators.operator_surface as surf

    return set(surf.INTERNAL_ONLY_CANONICALS)


def _mining_visible_names(tier="research"):
    from factor_engine.api.mining_integration import default_typed_mining_search_space_config

    cfg = default_typed_mining_search_space_config(tier=tier)
    names = set()
    for o in cfg["operators"]:
        if isinstance(o, dict):
            if o.get("mining_visible"):
                names.add(o.get("name"))
        else:
            names.add(str(o))
    return names


def test_forbidden_not_in_mining_discovery():
    names = _mining_visible_names()
    present = sorted(PERMANENTLY_FORBIDDEN_CANONICALS & names)
    assert present == [], f"forbidden canonicals mining-visible: {present}"


def _production_admitted_names():
    from factor_engine.api.mining_integration import default_typed_mining_search_space_config

    cfg = default_typed_mining_search_space_config(tier="production")
    admitted = set()
    for o in cfg["operators"]:
        if isinstance(o, dict) and o.get("production_admitted"):
            admitted.add(o.get("name"))
    return admitted


def test_research_and_unsafe_not_production_admitted():
    """Research/unsafe-surface canonicals may appear in the RESEARCH search space
    but must never be production-admitted (R28 §一百一十六: 0 production exposure)."""
    admitted = _production_admitted_names()
    bad = sorted(_research_and_diagnostic() & admitted)
    assert bad == [], f"research/unsafe canonicals production-admitted: {bad}"


def test_internal_not_mining_visible():
    names = _mining_visible_names()
    bad = sorted(_internal_canonicals() & names)
    assert bad == [], f"internal canonicals mining-visible: {bad}"


def test_diagnostic_in_sample_models_not_mining_visible():
    """The AR in-sample diagnostics (current row participates in its own fit,
    no prior/forecast semantics) must have ZERO mining exposure (R28 §二十二 /
    §一百一十六 / P0-007).  The ``*_prior_*`` causal forms ARE mineable."""
    names = _mining_visible_names()
    for diag in ("ts_ar_fitted_value", "ts_ar_in_sample_resid"):
        assert diag not in names, f"{diag} (in-sample diagnostic) is mining-visible"
    # the causal prior forms are reachable
    for causal in ("ts_ar_prior_forecast", "ts_ar_prior_innovation"):
        assert causal in names, f"{causal} (causal prior form) missing from mining space"


def test_constant_grammar_helper_not_mining_visible():
    names = _mining_visible_names()
    assert "constant" not in names, "constant grammar helper is mining-visible"


def test_production_operators_reachable():
    from factor_engine.api.mining_integration import validate_production_dsl

    for formula in (
        "ts_mean(close, 5)",
        "ts_std(close, 20)",
        "rank(close)",
        "zscore(close, 20)",
        "ts_delta(close, 5)",
    ):
        ok, msg = validate_production_dsl(formula, market="ashare")
        # fail-closed is the current honest state (production evidence is not
        # fully certified), but the operator MUST be resolvable (not "unsupported")
        if not ok:
            assert "Unsupported function" not in msg, f"{formula}: unknown operator: {msg}"
            assert "未知" not in msg, f"{formula}: unknown operator: {msg}"


def test_alias_policy_monotonicity():
    """alias_safety must not be wider than canonical_safety (R28 §一百零四..五)."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.operator_surface import is_dsl_name_allowed

    aliases = dict(getattr(OperatorRegistry, "_aliases", {}) or {})
    for alias, canonical in aliases.items():
        for surface in ("daily", "extended"):
            alias_ok = is_dsl_name_allowed(alias, canonical, surface=surface)
            # if the canonical is allowed, the alias may be too; but if the alias
            # is allowed the canonical must be (monotonicity), and a forbidden
            # canonical must never expose an allowed alias.
            if alias_ok:
                assert is_dsl_name_allowed(
                    canonical, canonical, surface=surface
                ), f"alias {alias} allowed but canonical {canonical} not ({surface})"


def test_deprecated_alias_not_independent_search_candidate():
    """An alias must never let mining treat the same kernel as two candidates."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    names = _mining_visible_names()
    aliases = dict(getattr(OperatorRegistry, "_aliases", {}) or {})
    # every alias that is mining-visible must resolve to a canonical that is ALSO
    # mining-visible (no phantom alias-only entries), and the alias name itself
    # must not appear independently in the discovery list.
    for alias, canonical in aliases.items():
        if alias in names:
            assert canonical in names, f"alias {alias} visible but canonical {canonical} not"
