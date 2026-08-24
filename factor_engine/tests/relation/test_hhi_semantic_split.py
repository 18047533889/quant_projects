# -*- coding: utf-8 -*-
"""R24-011..018: the two HHI economic definitions are separate canonicals with
separate denominators, and holder-rank missing slots fail closed unless they are
structural zeros."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _pair(v1, v2):
    idx = pd.date_range("2024-01-01", periods=1)
    return (
        pd.DataFrame([[v1]], index=idx, columns=["A"]),
        pd.DataFrame([[v2]], index=idx, columns=["A"]),
    )


def test_two_hhi_definitions_are_distinct_canonicals() -> None:
    # R24-011/014: separate canonicals, never aliased.
    assert OperatorRegistry.get("holder_company_ownership_hhi") is not None
    assert OperatorRegistry.get("holder_observed_topk_hhi") is not None
    assert OperatorRegistry.get("relation_hhi") is not None
    assert (
        OperatorRegistry.get("holder_company_ownership_hhi").metadata.name
        != OperatorRegistry.get("holder_observed_topk_hhi").metadata.name
    )


def test_company_ownership_hhi_uses_raw_ratios() -> None:
    # R24-012: denominator = company total shares; the ratio is used as-is.
    a, b = _pair(0.2, 0.1)
    out = OperatorRegistry.get("holder_company_ownership_hhi").calculate(a, b)
    assert out.iloc[0, 0] == pytest.approx(0.2**2 + 0.1**2)  # 0.05


def test_observed_topk_hhi_normalizes_by_topk_subtotal() -> None:
    # R24-013: denominator = observed top-k subtotal.
    a, b = _pair(0.2, 0.1)
    out = OperatorRegistry.get("holder_observed_topk_hhi").calculate(a, b)
    assert out.iloc[0, 0] == pytest.approx((0.2**2 + 0.1**2) / (0.3**2))


def test_relation_hhi_matches_observed_topk_definition() -> None:
    a, b = _pair(0.2, 0.1)
    assert OperatorRegistry.get("relation_hhi").calculate(a, b).iloc[0, 0] == pytest.approx(
        OperatorRegistry.get("holder_observed_topk_hhi").calculate(a, b).iloc[0, 0]
    )


def test_missing_slot_unknown_fails_closed() -> None:
    # R24-017: an UNKNOWN empty slot is NOT a zero — the output is NaN.
    idx = pd.date_range("2024-01-01", periods=1)
    a = pd.DataFrame([[0.2, np.nan]], index=idx, columns=["A", "B"])
    b = pd.DataFrame([[0.1, np.nan]], index=idx, columns=["A", "B"])
    out = OperatorRegistry.get("relation_hhi").calculate(a, b, missing_semantic="unknown")
    assert np.isnan(out.iloc[0, 1])  # the all-NaN column stays NaN
    assert np.isfinite(out.iloc[0, 0])


def test_structural_zero_treats_empty_as_zero() -> None:
    # R24-016: STRUCTURAL_ZERO / OUTSIDE_TOP_K may count as 0.
    idx = pd.date_range("2024-01-01", periods=1)
    a = pd.DataFrame([[0.2, np.nan]], index=idx, columns=["A", "B"])
    b = pd.DataFrame([[0.1, np.nan]], index=idx, columns=["A", "B"])
    out = OperatorRegistry.get("relation_hhi").calculate(a, b, missing_semantic="outside_top_k")
    # B column: only zero-value slots -> total=0 -> NaN (no positive weights).
    assert np.isnan(out.iloc[0, 1])
    # A column unchanged.
    assert np.isclose(out.iloc[0, 0], (0.2**2 + 0.1**2) / (0.3**2))


def test_missing_policy_enum_strict() -> None:
    from factor_engine.cleaned_operators.relation.ops import HolderRankMissingSemantic

    assert HolderRankMissingSemantic.permits_zero("structural_zero")
    assert HolderRankMissingSemantic.permits_zero("outside_top_k")
    assert not HolderRankMissingSemantic.permits_zero("not_reported")
    assert not HolderRankMissingSemantic.permits_zero("source_missing")
    assert not HolderRankMissingSemantic.permits_zero("unknown")


def test_unknown_policy_value_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    a = pd.DataFrame([[0.2]], index=idx, columns=["A"])
    b = pd.DataFrame([[0.1]], index=idx, columns=["A"])
    with pytest.raises(ValueError, match="missing_semantic"):
        OperatorRegistry.get("relation_hhi").calculate(a, b, missing_semantic="bogus")
