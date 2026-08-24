# -*- coding: utf-8 -*-
"""R28 §七: R28_FUTURE_FUNCTION_TERMINALS_ZERO and R28_NONCAUSAL_FILL_TERMINALS_ZERO.

No registered canonical may expose a future/lookahead primitive (Lead, next,
bfill, interpolate, centered rolling, filtfilt) as a factor terminal.  The
permanently-forbidden names are physically removed (proven by
``test_forbidden_operator_surface``); this file proves the *aliases* cannot
reintroduce them and the authoring surfaces carry no future primitive.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_spec import (
    PERMANENTLY_FORBIDDEN_CANONICALS,
    PRODUCTION_DENIED_CANONICALS,
)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


FUTURE_NAMES = {
    "Lead", "lead", "next", "bfill", "backfill", "fillna_interpolate",
    "interpolate", "fft", "ifft", "wavelet", "convolve", "correlate",
}


def test_no_future_names_registered():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    names = set(OperatorRegistry.list_canonical())
    present = sorted(FUTURE_NAMES & names)
    # ``constant`` is the only surviving PERMANENTLY_FORBIDDEN name (INTERNAL
    # grammar helper); every other future/fill primitive is physically gone.
    allowed_internal = {"constant"}
    assert set(present) <= allowed_internal, f"future-primitive canonicals registered: {present}"


def test_no_future_names_in_authoring_surfaces():
    import factor_engine.cleaned_operators.operator_surface as surf

    for surf_name, surf_set in (
        ("DAILY", surf.DAILY_CANONICALS),
        ("EXTENDED", surf.EXTENDED_ONLY_CANONICALS),
        ("RESEARCH", surf.RESEARCH_ONLY_CANONICALS),
    ):
        present = sorted(FUTURE_NAMES & set(surf_set))
        assert present == [], f"future primitive in {surf_name}: {present}"


def test_alias_cannot_resolve_to_future_primitive():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    aliases = dict(getattr(OperatorRegistry, "_aliases", {}) or {})
    escapes = {a: c for a, c in aliases.items() if c in FUTURE_NAMES}
    assert escapes == {}, f"alias -> future primitive: {escapes}"
    # a future primitive must not be reachable as an alias key either
    as_key = sorted(FUTURE_NAMES & set(aliases.keys()))
    assert as_key == [], f"future primitive as alias key: {as_key}"


def test_production_deny_covers_non_causal_names():
    """Names that were removed as factor terminals must remain denied so legacy
    expressions cannot silently re-enter production."""
    import factor_engine.cleaned_operators.operator_spec as spec

    from factor_engine.cleaned_operators.tombstones import is_tombstoned

    for name in ("bfill", "causal_bfill", "fillna_interpolate", "shuffle"):
        covered = spec.is_production_denied(name) or is_tombstoned(name)
        assert covered, f"{name} not denied and not tombstoned"
    # every non-causal name KNOWN to the system (registered, denied, forbidden,
    # or tombstoned) must be excluded from production authoring.  R30 moved
    # several permanently-forbidden names (Lead/bfill/next/...) to the tombstone
    # registry, so a name is "covered" if it is denied OR permanently-forbidden
    # OR tombstoned.  Names never canonicals (lowercase ``lead``, ``backfill``)
    # legitimately have no entry.
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.tombstones import is_tombstoned

    known = (
        set(OperatorRegistry.list_canonical())
        | set(PRODUCTION_DENIED_CANONICALS)
        | set(PERMANENTLY_FORBIDDEN_CANONICALS)
    )
    for name in FUTURE_NAMES:
        if name not in known:
            continue
        covered = (
            name in PERMANENTLY_FORBIDDEN_CANONICALS
            or spec.is_production_denied(name)
            or is_tombstoned(name)
        )
        assert covered, f"{name} neither denied, forbidden nor tombstoned"
