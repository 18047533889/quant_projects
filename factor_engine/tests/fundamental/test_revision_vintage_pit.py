# -*- coding: utf-8 -*-
"""R23-086..090 / R23-297: revision operators must NOT claim proven historical
revision from daily panel changes.

The revision family infers same-period revisions from daily panel state change.
That is only a valid proof with a true historical-vintage source; the operators
must DECLARE ``requires:RevisionEventSource`` + ``revision_vintage_pit_certified:false``
on their contract so production mining/admission can block them until the source
proves immutable vintage history.
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

_REVISION_CANONICALS = (
    "fin_revision_delta",
    "fin_revision_pct",
    "fin_revision_direction",
    "fin_revision_count",
    "fin_revision_magnitude",
    "fin_restated_flag",
    "fin_days_since_update",
    "fin_staleness",
)


def _tags(canonical):
    ops = OperatorRegistry._operators.get(canonical, {})
    pd_op = ops.get("pandas_numpy")
    if pd_op is not None:
        return list(getattr(getattr(pd_op, "metadata", None), "tags", None) or [])
    return []


def test_revision_family_declares_revision_event_source_requirement():
    for name in _REVISION_CANONICALS:
        assert name in OperatorRegistry._catalog, name
        tags = _tags(name)
        assert "requires:RevisionEventSource" in tags, name
        assert "revision_vintage_pit_certified:false" in tags, name


def test_non_revision_period_ops_do_not_claim_revision_source():
    # fin_positive_streak is an ordinary period operator — it must NOT carry the
    # revision-source requirement (it counts report-to-report changes, not
    # revisions of an already-visible period).
    tags = _tags("fin_positive_streak")
    assert "requires:RevisionEventSource" not in tags


def test_revision_policy_distinct_first_and_latest():
    # R23-229..232: first_available / latest_available are distinct revision
    # policies; the strict fiscal kernel must accept both.
    from cleaned_operators import fiscal_strict

    assert fiscal_strict._REVISION_POLICIES == {"latest_available", "first_available"}
