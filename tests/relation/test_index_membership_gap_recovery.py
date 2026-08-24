# -*- coding: utf-8 -*-
"""R24-036..038: a provider gap must not reset index membership entry age; an
unprovable in-gap exit keeps the exact age NaN while ``lower_bound`` mode emits
the floor age."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry


def _member(values):
    return pd.DataFrame(
        [[v] for v in values], index=pd.date_range("2024-01-01", periods=len(values)),
        columns=["A"],
    )


def test_gap_does_not_reset_entry_age_in_lower_bound_mode() -> None:
    # R24-037: gap recovery with identical membership does NOT reset the age.
    m = _member([1.0, 1.0, np.nan, 1.0, 1.0])
    out = OperatorRegistry.get("index_membership_age").calculate(m, output_mode="lower_bound")
    # Floor age at recovery = row3 - entry(row0) = 3; continues 3, 4.
    assert out.iloc[3, 0] == pytest.approx(3.0)
    assert out.iloc[4, 0] == pytest.approx(4.0)


def test_gap_exact_mode_keeps_age_nan() -> None:
    # R24-038: an unprovable in-gap exit keeps the exact age NaN permanently.
    m = _member([1.0, 1.0, np.nan, 1.0, 1.0])
    out = OperatorRegistry.get("index_membership_age").calculate(m)
    assert np.isnan(out.iloc[2, 0])
    assert np.isnan(out.iloc[3, 0])
    assert np.isnan(out.iloc[4, 0])


def test_confirmed_exit_after_gap_resets_entry() -> None:
    m = _member([1.0, 1.0, np.nan, 0.0, 1.0])
    out = OperatorRegistry.get("index_membership_age").calculate(m)
    # Re-entry after a CONFIRMED non-member row is a fresh entry (age 0).
    assert out.iloc[4, 0] == pytest.approx(0.0)


def test_no_gap_unchanged() -> None:
    m = _member([1.0, 1.0, 1.0, 0.0, 1.0])
    out = OperatorRegistry.get("index_membership_age").calculate(m)
    assert out.iloc[0, 0] == pytest.approx(0.0)
    assert out.iloc[1, 0] == pytest.approx(1.0)
    assert out.iloc[2, 0] == pytest.approx(2.0)
    assert np.isnan(out.iloc[3, 0])
    assert out.iloc[4, 0] == pytest.approx(0.0)


def test_output_mode_enum_strict() -> None:
    m = _member([1.0, 1.0])
    with pytest.raises(ValueError, match="output_mode"):
        OperatorRegistry.get("index_membership_age").calculate(m, output_mode="bogus")
