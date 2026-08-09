# -*- coding: utf-8 -*-
"""R11 relation rank-slot operator-semantics regression tests.

Two operator-semantic review items:

* ISSUE 1 — ``relation_topk_concentration`` must use RANK-SLOT semantics
  (numerator = s1 + .. + sk, any missing slot among the first k => NaN) instead
  of value-sorting the largest k values (which silently promoted a lower rank
  into the top-k when the true top rank was missing).
* ISSUE 2 — ``relation_rank_entity_mobility`` must declare a VALID unit contract
  (``same_as:share`` — the output carries the share unit of the value panels it
  consumes) instead of the ambiguous ``same_as:value`` / nonexistent
  ``same_as:valuemount``.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.relation import distribution as rdist


# ---------------------------------------------------------------------------
# ISSUE 1 — rank-slot top-k concentration
# ---------------------------------------------------------------------------
def test_topk_rank_slot_fail_closed_on_missing_top_rank():
    """s1 (the true #1 slot) missing => NaN, even when s2..s6 are large.

    An unknown top-1 must never be backfilled by a lower rank.  A stock whose
    first-k slots are all finite gets the slot sum over the total.
    """
    idx = pd.date_range("2024-01-01", periods=1)
    # Stock A: s1 missing, s2..s6 LARGE — value-sorting would happily promote
    # s2..s6 into the top-5; rank-slot semantics must fail closed instead.
    # Stock B: s1..s5 finite, s6 finite — slot top-5 = 98 / total = 0.98.
    s1 = pd.DataFrame({"A": [np.nan], "B": [50.0]}, index=idx)
    s2 = pd.DataFrame({"A": [100.0], "B": [30.0]}, index=idx)
    s3 = pd.DataFrame({"A": [90.0], "B": [10.0]}, index=idx)
    s4 = pd.DataFrame({"A": [80.0], "B": [5.0]}, index=idx)
    s5 = pd.DataFrame({"A": [70.0], "B": [3.0]}, index=idx)
    s6 = pd.DataFrame({"A": [60.0], "B": [2.0]}, index=idx)
    op = rdist.RelationTopkConcentration()
    out = op.calculate(s1, s2, s3, s4, s5, s6, k=5)
    # A: top rank missing -> NaN (fail closed, no silent promotion).
    assert np.isnan(out.iloc[0, 0])
    # B: s1..s5 finite -> (s1+..+s5) / (s1+..+s6) = (50+30+10+5+3) / 100.
    assert out.iloc[0, 1] == pytest.approx(0.98)


def test_topk_rank_slot_beats_value_sorted():
    """Slot semantics win even when the value-sorted top-k would differ.

    s1 is finite and larger than s6, but s2/s3 dwarf it: value-sorted top-2 is
    s2+s3, while rank-slot top-2 is s1+s2.  The operator must report the slot
    top-2, never the sorted largest values.
    """
    idx = pd.date_range("2024-01-01", periods=1)
    s1 = pd.DataFrame({"A": [5.0]}, index=idx)
    s2 = pd.DataFrame({"A": [100.0]}, index=idx)
    s3 = pd.DataFrame({"A": [90.0]}, index=idx)
    s4 = pd.DataFrame({"A": [80.0]}, index=idx)
    s5 = pd.DataFrame({"A": [70.0]}, index=idx)
    s6 = pd.DataFrame({"A": [1.0]}, index=idx)
    op = rdist.RelationTopkConcentration()
    out = op.calculate(s1, s2, s3, s4, s5, s6, k=2)
    total = 5.0 + 100.0 + 90.0 + 80.0 + 70.0 + 1.0
    assert out.iloc[0, 0] == pytest.approx((5.0 + 100.0) / total)
    # The value-sorted top-2 (100+90)/total is strictly larger — assert the
    # slot semantics won, not the value sort.
    assert out.iloc[0, 0] != pytest.approx((100.0 + 90.0) / total)


def test_topk_k_exceeds_provided_panels_rejected():
    """k must fit the actual number of provided rank slots (kept from P1-121)."""
    idx = pd.date_range("2024-01-01", periods=1)
    s1 = pd.DataFrame({"A": [1.0]}, index=idx)
    s2 = pd.DataFrame({"A": [2.0]}, index=idx)
    with pytest.raises(ValueError):
        rdist.RelationTopkConcentration().calculate(s1, s2, k=3)


# ---------------------------------------------------------------------------
# ISSUE 2 — valid unit contract on relation_rank_entity_mobility
# ---------------------------------------------------------------------------
def test_rank_entity_mobility_unit_contract_valid():
    """The mobility output carries the share unit of the value panels consumed.

    The operator consumes share/weight VALUE panels (s1..s10 / p1..p10) and
    emits a change of those values, so the declared unit must be
    ``same_as:share`` (matching ``relation_share_mobility``) — never the
    ambiguous ``same_as:value`` / nonexistent ``same_as:valuemount``.
    """
    meta = rdist.RelationRankEntityMobility.metadata
    assert meta.output_unit in ("same_as:share", "dimensionless")
    assert meta.output_unit == "same_as:share"
    assert any("unit:same_as:share" == t for t in (meta.tags or []))


def test_no_invalid_valuemount_string_remains_in_module():
    """No ``same_as:valuemount`` (or bare ``valuemount``) left in the file."""
    src = inspect.getsource(rdist)
    assert "same_as:valuemount" not in src
    assert "valuemount" not in src
