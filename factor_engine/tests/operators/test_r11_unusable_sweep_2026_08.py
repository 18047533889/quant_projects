# -*- coding: utf-8 -*-
"""R11 unusable-operators sweep: every production target structurally usable.

Root cause fixed in this sweep:
- ``apply_production_hardening`` wrote ``pit_safe = catalog.get("pit_safe", False)``
  over every target, and ``apply_lqtp_policy_patch`` pre-seeded a placeholder
  ``False`` for every un-reviewed canonical — so a stale evidence artifact
  made the WHOLE production surface fail the audit's structural ``pit_safe``
  check, deadlocking the certifier bootstrap (review §2.4).
- The audit script could not classify ~85 public parameters (fixture gaps) and
  passed domain-invalid generic scalars (``q=0.2``, ``missing_policy="break"``),
  and the minute-source ``intra_*`` grain-changing family was wrongly flagged
  ``shape_preserving=False``.

This test pins the RESULT: every production target that is not fail-closed and
not in ``PIT_UNSAFE_CANONICALS`` must have a structurally-causal policy, and the
audit fixture must resolve every declared parameter.  The genuinely non-causal
family (``Lead``/``next``/``bfill``/``causal_bfill``/``fillna_interpolate``/
``shuffle``) must stay fail-closed.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.production_hardening import factor_production_targets
from factor_engine.cleaned_operators.operator_policy import (
    PIT_UNSAFE_CANONICALS,
    infer_operator_policy,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.semantic_certification import should_fail_closed


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_all_non_fail_closed_targets_have_causal_policy() -> None:
    for canonical in sorted(factor_production_targets()):
        if should_fail_closed(canonical):
            continue
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canonical)
        assert policy.pit_safe is True, canonical
        assert policy.lag is None or policy.lag >= 0, canonical


def test_genuinely_non_causal_family_stays_fail_closed() -> None:
    for canonical in sorted(PIT_UNSAFE_CANONICALS):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canonical)
        assert policy.pit_safe is False, canonical


def test_newly_registered_stats_ops_are_surface_classified() -> None:
    """R11 unusable-operators sweep: the concurrent session's
    ``ts_mean_abs_deviation`` / ``ts_median_abs_deviation`` were registered but
    left out of every surface — layer_governance failed the whole load.  They
    are now classified extended (stats MAD family, alongside ``ts_mad``)."""
    from factor_engine.cleaned_operators.operator_surface import EXTENDED_ONLY_CANONICALS

    for canonical in ("ts_mean_abs_deviation", "ts_median_abs_deviation"):
        assert canonical in OperatorRegistry._operators, canonical
        assert canonical in EXTENDED_ONLY_CANONICALS, canonical
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(op, canonical=canonical)
        assert policy.pit_safe is True, canonical


def test_minute_source_family_is_pit_safe_but_shape_changing() -> None:
    from scripts.audit_all_factor_production import _minute_source

    minute = [
        c for c in factor_production_targets() if _minute_source(c)
    ]
    assert minute, "no minute-source targets found"
    for canonical in minute:
        if should_fail_closed(canonical):
            continue
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(op, canonical=canonical)
        assert policy.pit_safe is True, canonical
        # minute->daily aggregation is their design; not a shape defect
        assert policy.shape_preserving is False, canonical


def test_audit_fixture_resolves_every_target_parameter() -> None:
    from scripts.audit_all_factor_production import _build_call, _panels

    panels = _panels()
    unresolved: list[str] = []
    for canonical in sorted(factor_production_targets()):
        if should_fail_closed(canonical):
            continue
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is None:
            continue
        try:
            _build_call(canonical, op, panels)
        except KeyError as exc:  # noqa: PERF203
            unresolved.append(f"{canonical}: {exc}")
    assert not unresolved, "\n".join(unresolved)
