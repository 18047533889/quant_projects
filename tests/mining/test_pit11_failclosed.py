# -*- coding: utf-8 -*-
"""R23-P0-PIT11: expectation/surprise operators fail closed for production.

The R23 audit marks 7 financial expectation/surprise operators as P0 with
blocker ``PIT11_EXPECTATION_POST_EVENT_LEAK``: their values are only knowable
AFTER an earnings event, so placing them as production factor terminals is a
label-leak vector (the label is knowable before the signal).  They are
certified contextual only — never production-admitted.

This test pins the fail-closed gate chain for all 7:
  production_admitted = False        (deny gate via PRODUCTION_DENIED_CANONICALS)
  production_terminal_usable = False (terminal_usable ∧ production_admitted)
  directly_usable = False            (mining_visible ∧ production_admitted)
"""
from __future__ import annotations

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.direct_use import build_direct_use_operator, _is_production_denied

# R23-P0-PIT11: the audit blocker reason shared by every canonical below.
PIT11_EXPECTATION_POST_EVENT_LEAK = "PIT11_EXPECTATION_POST_EVENT_LEAK"

# The 7 operators whose expectations/surprises are only knowable AFTER an
# earnings event — the label is knowable before the signal, so production
# terminal placement leaks the label.  (docs/R23_PER_CANONICAL_AUDIT.json and
# docs/R23_OPERATOR_REMEDIATION_PLAN.md both list these 7 as P0.)
PIT11_LEAK_CANONICALS: tuple[str, ...] = (
    "fin_actual_expectation_divergence",
    "fin_beat_streak",
    "fin_miss_streak",
    "fin_surprise",
    "fin_surprise_event_percentile",
    "fin_surprise_event_zscore",
    "fin_surprise_zscore",
)


def _row(canonical: str):
    load_all()
    catalog = dict(OperatorRegistry._catalog.get(canonical, {}))
    return build_direct_use_operator(canonical, catalog)


class TestPit11ExpectationPostEventLeakFailClosed:
    """R23-P0-PIT11: all 7 leak canonicals are fail-closed for production."""

    def test_pit11_canonicals_are_registered(self) -> None:
        """Every audited PIT11 leak canonical is a real registered operator."""
        load_all()
        for canonical in PIT11_LEAK_CANONICALS:
            assert canonical in OperatorRegistry._catalog, (
                f"{canonical} is missing from the operator registry"
            )

    def test_pit11_canonicals_are_production_denied(self) -> None:
        """The deny gate matches — each canonical is in PRODUCTION_DENIED_CANONICALS."""
        for canonical in PIT11_LEAK_CANONICALS:
            assert canonical in PRODUCTION_DENIED_CANONICALS, (
                f"{canonical} is not in PRODUCTION_DENIED_CANONICALS"
            )
            assert _is_production_denied(canonical) is True, (
                f"{canonical}: _is_production_denied returned False"
            )

    def test_pit11_canonicals_production_terminal_usable_false(self) -> None:
        """production_terminal_usable is False for every PIT11 leak canonical.

        Reason pinned explicitly: PIT11_EXPECTATION_POST_EVENT_LEAK — these
        operators compute expectations/surprises only knowable AFTER an
        earnings event; a production terminal would leak the label.
        """
        for canonical in PIT11_LEAK_CANONICALS:
            row = _row(canonical)
            assert row.production_terminal_usable is False, (
                f"{canonical}: production_terminal_usable must be False "
                f"({PIT11_EXPECTATION_POST_EVENT_LEAK}); got {row.production_terminal_usable}"
            )

    def test_pit11_canonicals_production_admitted_false(self) -> None:
        """production_admitted is False for every PIT11 leak canonical."""
        for canonical in PIT11_LEAK_CANONICALS:
            row = _row(canonical)
            assert row.production_admitted is False, (
                f"{canonical}: production_admitted must be False "
                f"({PIT11_EXPECTATION_POST_EVENT_LEAK}); got {row.production_admitted}"
            )

    def test_pit11_canonicals_directly_usable_false(self) -> None:
        """directly_usable (admission alias) is False for every leak canonical."""
        for canonical in PIT11_LEAK_CANONICALS:
            row = _row(canonical)
            assert row.directly_usable is False, (
                f"{canonical}: directly_usable must be False "
                f"({PIT11_EXPECTATION_POST_EVENT_LEAK}); got {row.directly_usable}"
            )

    def test_pit11_deny_entry_is_static(self) -> None:
        """Fail-closed even for a hypothetical certified catalog row.

        The deny gate is the static ``_is_production_denied`` check inside
        ``production_admitted`` — it must hold regardless of catalog
        certification, so it also holds here without any certification patch.
        """
        for canonical in PIT11_LEAK_CANONICALS:
            assert _is_production_denied(canonical) is True