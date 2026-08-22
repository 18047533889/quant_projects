# -*- coding: utf-8 -*-
"""PIT 标签层：forward return 与特征窗口隔离。"""

from __future__ import annotations

import pandas as pd
import pytest

from api.label_pit import (
    LabelWindowSpec,
    align_feature_and_label_windows,
    assert_label_feature_no_overlap,
    build_forward_return_series,
    default_mining_label_config,
)
from runtime.pit_audit import PitSafetyError

pd = pytest.importorskip("pandas")


def test_default_mining_label_config_has_gap():
    cfg = default_mining_label_config(horizon_bars=5, feature_lookback_bars=20, gap_bars=1)
    assert cfg["gap_bars"] == 1
    assert "forward_return" in cfg["label_formula"]


def test_assert_label_feature_no_overlap_raises():
    spec = LabelWindowSpec(horizon_bars=5, feature_lookback_bars=20, gap_bars=1)
    with pytest.raises(PitSafetyError):
        assert_label_feature_no_overlap(feature_end_bar=10, label_start_bar=10, spec=spec)


def test_build_forward_return_series():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=4, freq="D"), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx)
    fwd = build_forward_return_series(close, horizon=1)
    assert pd.isna(fwd.iloc[-1])
    assert fwd.iloc[0] == pytest.approx(0.1)


def test_align_feature_and_label_windows_trims_label():
    spec = LabelWindowSpec(horizon_bars=1, feature_lookback_bars=5, gap_bars=1)
    feat_idx = pd.date_range("2024-01-02", periods=5, freq="D")
    label_idx = pd.date_range("2024-01-05", periods=5, freq="D")
    f_out, l_out = align_feature_and_label_windows(feat_idx, label_idx, spec=spec)
    assert len(l_out) < len(label_idx)
