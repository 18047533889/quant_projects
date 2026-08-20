# -*- coding: utf-8 -*-
"""R30 §2/§3: random / future / non-causal fill primitives are physically
removed from the executable system and replaced by frozen tombstones.

Hard gates exercised here:
* ACTIVE_RANDOM_FACTOR_PRIMITIVES == 0
* ACTIVE_FUTURE_REFERENCE_PRIMITIVES == 0
* ACTIVE_NONCAUSAL_FILL_PRIMITIVES == 0
* tombstoned names never live in the runtime registry / aliases / catalog
* ``OperatorRegistry.get`` / ``resolve_canonical`` raise ``RemovedOperatorError``
* governance "operator category" sets are disjoint from tombstone names
"""
from __future__ import annotations

import pytest

import cleaned_operators  # noqa: F401  (registers _LOAD_MODULES)
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.tombstones import (
    RemovedOperatorError,
    RANDOM_TOMBSTONED_NAMES,
    FUTURE_TOMBSTONED_NAMES,
    NONCAUSAL_FILL_TOMBSTONED_NAMES,
    ALL_TOMBSTONED_NAMES,
    is_tombstoned,
)

RANDOM = ("rand_uniform", "rand_normal", "rand_lognormal", "rand_poisson", "rand_exp", "shuffle", "sample")
FUTURE = ("Lead", "lead", "next")
NONCAUSAL = ("bfill", "causal_bfill", "fillna_interpolate", "interpolate")


def _assert_no_executable(names, registry):
    for n in names:
        # direct dict inspection: resolve_canonical raises on tombstoned names
        assert n not in registry._operators, f"{n} still executable"
        assert n not in registry._catalog, f"{n} still in catalog"
        assert n not in registry._aliases, f"{n} still aliased"


def test_random_primitives_have_no_executable():
    _assert_no_executable(RANDOM, OperatorRegistry)


def test_future_primitives_have_no_executable():
    _assert_no_executable(FUTURE, OperatorRegistry)


def test_noncausal_fill_primitives_have_no_executable():
    _assert_no_executable(NONCAUSAL, OperatorRegistry)


def test_tombstone_get_raises():
    for n in ("rand_uniform", "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate"):
        with pytest.raises(RemovedOperatorError):
            OperatorRegistry.get(n)


def test_tombstone_resolve_raises():
    with pytest.raises(RemovedOperatorError):
        OperatorRegistry.resolve_canonical("Lead")


def test_tombstone_is_not_silently_replaced():
    # Lead must never silently lower to Delay/ffill; it must raise.
    with pytest.raises(RemovedOperatorError):
        OperatorRegistry.get("Lead")


def test_tombstone_names_are_all_tombstoned():
    for n in RANDOM + FUTURE + NONCAUSAL:
        assert is_tombstoned(n), n


def test_governance_sets_disjoint_from_tombstones():
    from cleaned_operators import operator_policy as op
    from cleaned_operators import operator_spec as osp
    from cleaned_operators import production_hardening as ph
    from backend import production_fastpath_tiers as pft
    from backend import polars_long_policy as plp

    sets = {
        "PIT_UNSAFE": op.PIT_UNSAFE_CANONICALS,
        "PANDAS_ONLY": op.INTENTIONALLY_PANDAS_ONLY,
        "EXPLICIT_POLICIES": set(op._EXPLICIT_POLICIES.keys()),
        "FORBIDDEN": osp.PERMANENTLY_FORBIDDEN_CANONICALS,
        "DENIED": osp.PRODUCTION_DENIED_CANONICALS,
        "PIT_EXEMPT": osp._PIT_EXEMPT,
        "NON_FACTOR": ph.NON_FACTOR_PRODUCTION_CANONICALS,
        "P2_FILL": pft.P2_FILL_INTERPOLATE_CANONICALS,
        "PLP_BLOCKED": plp.POLARS_LONG_BLOCKED_CAUSAL,
    }
    for name, s in sets.items():
        assert ALL_TOMBSTONED_NAMES.isdisjoint(s), f"{name} still carries tombstoned operator names"


def test_random_set_matches_expected_names():
    assert RANDOM_TOMBSTONED_NAMES == frozenset(RANDOM)
    assert FUTURE_TOMBSTONED_NAMES == frozenset(FUTURE)
    assert NONCAUSAL_FILL_TOMBSTONED_NAMES == frozenset(NONCAUSAL)
