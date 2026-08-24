# -*- coding: utf-8 -*-
"""Forward-label maturity + purged/embargoed split contracts (P0-K/P0-L)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.label_spec import (
    LabelSpec,
    assert_no_label_overlap,
    label_available_mask,
    purged_time_split,
)


def _dates(n: int = 40) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_label_maturity_requires_horizon():
    spec = LabelSpec(horizon=5)
    # A label anchored at s is mature only when s + 5 <= t.
    assert not spec.mature(anchor=10, fit_time=14)
    assert spec.mature(anchor=10, fit_time=15)
    assert spec.mature(anchor=10, fit_time=16)


def test_label_available_mask():
    dates = _dates(20)
    mask = label_available_mask(dates, horizon=5, fit_date=dates[12])
    assert not mask[8]  # 8 + 5 = 13 > 12, label not yet knowable
    assert mask[7]  # 7 + 5 = 12 <= 12
    assert not mask[-1]


def test_purged_time_split_no_overlap():
    dates = _dates(100)
    tr, va = purged_time_split(dates, train_frac=0.7, horizon=5, embargo=2)
    assert tr.size > 0 and va.size > 0
    assert tr[-1] < va[0]
    # No forward-5 label anchored in train reaches into validation.
    assert_no_label_overlap(tr, va, dates, horizon=5)


def test_purged_time_split_too_small_raises():
    with pytest.raises(ValueError):
        purged_time_split(_dates(4), train_frac=0.5, horizon=5, embargo=2)


def test_no_label_overlap_detects_violation():
    dates = _dates(30)
    tr = np.arange(0, 20)
    va = np.arange(22, 30)
    with pytest.raises(AssertionError):
        assert_no_label_overlap(tr, va, dates, horizon=5)
