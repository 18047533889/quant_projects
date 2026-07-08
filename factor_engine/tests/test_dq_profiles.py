# -*- coding: utf-8
"""DQ Profile YAML 加载与阈值解析测试。"""

from __future__ import annotations

import pytest

from runtime.dq_gates import DQThresholds
from runtime.dq_profiles import (
    list_dq_profiles,
    resolve_input_dq_thresholds,
    resolve_output_dq_thresholds,
)
from runtime.input_dq import InputDQThresholds


def test_list_dq_profiles_includes_prod_and_research():
    names = list_dq_profiles()
    assert "research" in names
    assert "us_equity_daily_prod" in names
    assert "us_equity_daily_research" in names


def test_resolve_output_dq_thresholds_prod():
    th = resolve_output_dq_thresholds("us_equity_daily_prod")
    assert isinstance(th, DQThresholds)
    assert th.min_coverage == pytest.approx(0.85)
    assert th.max_nan_ratio == pytest.approx(0.15)
    assert th.min_instruments_per_day == 500


def test_resolve_input_dq_thresholds_research():
    th = resolve_input_dq_thresholds("us_equity_daily_research")
    assert isinstance(th, InputDQThresholds)
    assert th.min_non_null_ratio == pytest.approx(0.50)


def test_resolve_none_profile_returns_none():
    assert resolve_output_dq_thresholds(None) is None
    assert resolve_input_dq_thresholds(None) is None


def test_unknown_profile_raises():
    with pytest.raises(KeyError, match="未知 DQ profile"):
        resolve_output_dq_thresholds("does_not_exist")
