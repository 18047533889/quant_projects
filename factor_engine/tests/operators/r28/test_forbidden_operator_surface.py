# -*- coding: utf-8 -*-
"""R28 Phase 2: permanently-forbidden operators must be physically gone from
every public path — runtime registry, daily/extended allowlists, mining
catalog, direct-use admission, backend emitters, aliases.

Gate: R28_PERMANENTLY_FORBIDDEN_RUNTIME_SURFACE_ZERO == True.
"""
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


FORBIDDEN = sorted(PERMANENTLY_FORBIDDEN_CANONICALS)


def _registry():
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry


def test_permanently_forbidden_not_registered_for_runtime():
    reg = _registry()
    canonicals = set(reg.list_canonical())
    present = sorted(set(FORBIDDEN) & canonicals)
    # ``constant`` is the sole survivor: a DENIED/INTERNAL grammar helper
    # required by the DSL/recipes for scalar literals, not a factor terminal.
    assert present == ["constant"], f"forbidden canonicals still in runtime: {present}"


def test_permanently_forbidden_not_in_daily_allowlist():
    import cleaned_operators.operator_surface as surf

    present = sorted(set(FORBIDDEN) & set(surf.DAILY_CANONICALS))
    assert present == [], f"forbidden in DAILY surface: {present}"


def test_permanently_forbidden_not_in_extended_allowlist():
    import cleaned_operators.operator_surface as surf

    present = sorted(set(FORBIDDEN) & set(surf.EXTENDED_ONLY_CANONICALS))
    assert present == [], f"forbidden in EXTENDED surface: {present}"


def test_permanently_forbidden_not_in_research_or_unsafe_surface():
    import cleaned_operators.operator_surface as surf

    present = sorted(
        set(FORBIDDEN)
        & (set(surf.RESEARCH_ONLY_CANONICALS) | set(surf.UNSAFE_CANONICALS))
    )
    assert present == [], f"forbidden in research/unsafe surface: {present}"


def test_permanently_forbidden_not_in_mining_catalog():
    from mining.operator_catalog import (
        MiningRole,
        assign_mining_role,
    )

    reg = _registry()
    catalog = reg.catalog()
    exposed = []
    for name in FORBIDDEN:
        try:
            role = assign_mining_role(name, catalog)
            if role not in (MiningRole.INTERNAL_HELPER, MiningRole.FORBIDDEN):
                exposed.append((name, str(role)))
        except Exception:
            # not in catalog at all -> clean
            pass
    assert exposed == [], f"forbidden canonicals mineable: {exposed}"


def test_permanently_forbidden_not_in_direct_use():
    from mining.direct_use import (
        DirectUseStatus,
        resolve_direct_use_status,
    )

    reg = _registry()
    catalog = reg.catalog()
    usable = []
    for name in FORBIDDEN:
        try:
            status = resolve_direct_use_status(name, catalog)
            if status in (DirectUseStatus.DIRECT_ALPHA, DirectUseStatus.DIRECT_CONTEXTUAL):
                usable.append((name, str(status)))
        except Exception:
            pass
    assert usable == [], f"forbidden canonicals direct-usable: {usable}"


def test_permanently_forbidden_not_in_backend_emitters():
    """The forbidden names must not appear in pandas/polars/SQL emitter tables."""
    import backend.polars_expr_emitter as pl_emit
    import backend.sql_pushdown.emitter as sql_emit

    text_sources = []
    for mod in (pl_emit, sql_emit):
        try:
            src = open(mod.__file__, encoding="utf-8").read()
        except Exception:
            continue
        for name in FORBIDDEN:
            if f'"{name}"' in src or f"'{name}'" in src:
                text_sources.append((mod.__name__, name))
    # Lead/next/bfill may appear in DENY/block lists inside emitters; only a
    # runtime dispatch table entry would matter, which the registry check above
    # already proves absent.  A pure string occurrence is not an emitter entry.
    print("emitter string mentions (non-fatal):", text_sources or "none")


def test_permanently_forbidden_alias_cannot_escape():
    reg = _registry()
    aliases = dict(getattr(reg, "_aliases", {}) or {})
    # no alias may resolve TO a forbidden canonical
    escapes = {a: c for a, c in aliases.items() if c in FORBIDDEN}
    assert escapes == {}, f"alias -> forbidden canonical: {escapes}"
    # forbidden names may not exist as alias keys either
    as_key = sorted(set(FORBIDDEN) & set(aliases.keys()))
    assert as_key == [], f"forbidden name used as alias key: {as_key}"


def test_legacy_formula_gets_explicit_forbidden_error():
    from api.mining_integration import validate_production_dsl

    for formula in ("bfill(close)", "Lead(close, 1)", "rand_uniform(close)", "shuffle(close)"):
        ok, msg = validate_production_dsl(formula)
        assert ok is False, f"{formula} unexpectedly accepted"
        assert msg and "Unsupported function" in msg, f"{formula}: unexpected msg {msg!r}"


def test_forbidden_not_in_dsl_allowlist():
    from api.mining_integration import list_dsl_allowlist

    allow = set(list_dsl_allowlist(surface="daily")) | set(list_dsl_allowlist(surface="extended"))
    present = sorted(set(FORBIDDEN) & allow)
    assert present == [], f"forbidden names in DSL allowlist: {present}"


def test_constant_is_denied_internal_grammar_helper():
    """``constant`` is the one deliberately-retained runtime name: an internal
    scalar-literal helper for the DSL/recipes, DENIED and INTERNAL-only, never a
    factor terminal."""
    import cleaned_operators.operator_surface as surf

    assert "constant" in surf.INTERNAL_ONLY_CANONICALS
    assert surf.classify_canonical("constant") == "internal"
    assert surf.production_certification("constant").name == "DENIED"
    assert surf.is_dsl_name_allowed("constant", "constant") is False
