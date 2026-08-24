# -*- coding: utf-8 -*-
"""R23 P0 close-out: PIT18_REVISION_EVENT_UNPROVEN fail-closed.

The 14 financial revision/restatement operators derive from point-in-time
revision events whose provenance (whether the revision was knowable at signal
time) is UNPROVEN.  factor_engine/docs/R23_PER_CANONICAL_AUDIT.json marks each as
P0 / blocker=PIT18_REVISION_EVENT_UNPROVEN / final=CERTIFIED_CONTEXTUAL —
contextual use only, never production terminal placement.

This test asserts the deny mechanism keeps every PIT18 canonical at
``production_terminal_usable=False`` AND ``production_admitted=False`` even
under a hypothetical full-certification overlay, and that NO PIT11/PIT18
operator may ever report ``production_terminal_usable=True``.
"""
from __future__ import annotations

import json
import os

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_spec import (
    PRODUCTION_DENIED_CANONICALS,
    is_production_denied,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.direct_use import _is_production_denied, build_direct_use_operator

# The 14 PIT18 canonicals, deduped from factor_engine/docs/R23_PER_CANONICAL_AUDIT.json by
# blocker=PIT18_REVISION_EVENT_UNPROVEN.  Note the audit list is UNAMBIGUOUS —
# the remediation plan's 14 rows overlap these exactly and there is no separate
# fin_expectation_revision_extra row.
PIT18_CANONICALS: tuple[str, ...] = tuple(
    sorted(
        {
            "fin_days_since_expectation_revision",
            "fin_days_since_update",
            "fin_expectation_revision",
            "fin_expectation_revision_count",
            "fin_expectation_revision_magnitude",
            "fin_expectation_revision_pct",
            "fin_expectation_revision_speed",
            "fin_restated_flag",
            "fin_revision_count",
            "fin_revision_delta",
            "fin_revision_direction",
            "fin_revision_magnitude",
            "fin_revision_pct",
            "fin_staleness",
        }
    )
)

_PIT11_CANONICALS: tuple[str, ...] = (
    "fin_actual_expectation_divergence",
    "fin_beat_streak",
    "fin_miss_streak",
    "fin_surprise",
    "fin_surprise_event_percentile",
    "fin_surprise_event_zscore",
    "fin_surprise_zscore",
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AUDIT = os.path.join(_ROOT, "factor_engine", "docs", "R23_PER_CANONICAL_AUDIT.json")


def _audit_pit18_canonicals() -> set[str]:
    """Dedupe exact PIT18 list straight from the R23 canonical audit."""
    with open(_AUDIT, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = (
        [v for v in data.values() if isinstance(v, dict) and "blocker" in v]
        if isinstance(data, dict)
        else list(data)
    )
    zero_blocker = {
        e["canonical"]
        for e in entries
        if isinstance(e, dict)
        and str(e.get("blocker", "")) == "PIT18_REVISION_EVENT_UNPROVEN"
    }
    revision_issue = {
        e["canonical"]
        for e in entries
        if isinstance(e, dict)
        and str(e.get("revision_issue", "")) == "PIT18_REVISION_EVENT_UNPROVEN"
    }
    return zero_blocker | revision_issue


class TestPit18FailClosed:
    """R23-303: PIT18 canonicals can never reach production terminal placement."""

    @pytest.fixture(autouse=True)
    def _loaded(self) -> None:
        load_all()

    def test_audit_list_matches_plan_count(self) -> None:
        """The 14 canonical set matches the remediation plan's 14-row PIT18 list."""
        audit = _audit_pit18_canonicals()
        assert audit == set(PIT18_CANONICALS)
        assert len(audit) == 14

    def test_pit18_listed_in_production_denied(self) -> None:
        """Every PIT18 canonical sits in PRODUCTION_DENIED_CANONICALS."""
        for canonical in PIT18_CANONICALS:
            assert canonical in PRODUCTION_DENIED_CANONICALS, canonical
            assert is_production_denied(canonical) is True, canonical
            # direct_use gate chain uses _is_production_denied as the deny hook
            assert _is_production_denied(canonical) is True, canonical

    def test_pit18_production_terminal_usable_false(self) -> None:
        """fail-closed: production_terminal_usable False for every PIT18 canonical."""
        for canonical in PIT18_CANONICALS:
            catalog = dict(OperatorRegistry._catalog.get(canonical, {}))
            row = build_direct_use_operator(canonical, catalog)
            assert canonical in OperatorRegistry._catalog, canonical
            assert row.production_terminal_usable is False, canonical
            assert row.production_admitted is False, canonical

    def test_pit18_failclosed_under_full_certification_overlay(self) -> None:
        """Even if certification/cost/sources were granted, the deny gate fires.

        The PIT18 rejection is the ``PRODUCTION_DENIED`` gate (R22-116..117) —
        a static hard gate, not a soft certification overlay.  Simulate a
        fully-certified catalog row and assert the deny still forces both
        flags to False.
        """
        for canonical in PIT18_CANONICALS:
            catalog = dict(OperatorRegistry._catalog.get(canonical, {}))
            overlay = dict(catalog)
            overlay["production_certified"] = True
            row = build_direct_use_operator(canonical, overlay)
            assert row.production_certified is True, canonical
            assert row.production_admitted is False, canonical
            assert row.production_terminal_usable is False, canonical
            # the deny mechanism is membership in the explicit deny set
            assert canonical in PRODUCTION_DENIED_CANONICALS, canonical

    def test_pit11_pit18_no_terminal_usable_true(self) -> None:
        """NO PIT11/PIT18 operator may report production_terminal_usable True."""
        for canonical in (*PIT18_CANONICALS, *_PIT11_CANONICALS):
            catalog = dict(OperatorRegistry._catalog.get(canonical, {}))
            row = build_direct_use_operator(canonical, catalog)
            assert row.production_terminal_usable is False, canonical
            assert row.production_admitted is False, canonical

    def test_pit18_not_in_eligible_admission(self) -> None:
        """No PIT18 canonical appears in the eligible (production) mining set."""
        from factor_engine.market.context import Market
        from factor_engine.mining.direct_use import DirectUseContext, get_direct_use_mining_operators

        operators = get_direct_use_mining_operators(
            context=DirectUseContext(market=Market.ASHARE),
            admission="eligible",
        )
        eligible = {op.canonical for op in operators}
        assert not (set(PIT18_CANONICALS) & eligible)
        assert not (set(_PIT11_CANONICALS) & eligible)