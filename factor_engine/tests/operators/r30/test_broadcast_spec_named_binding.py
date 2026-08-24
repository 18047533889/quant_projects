# -*- coding: utf-8 -*-
"""R30 §11: BroadcastSpec verification is bound by source_param/target_param
name, not by frame position.

* a spec verifies ONLY its declared source/target pair;
* a spec naming a param that is not a bound panel fails closed;
* a spec with only one of source/target named fails closed;
* kwargs reorder does not change which panel a spec verifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import (
    BroadcastSpec,
    OperatorMetadata,
    _verify_broadcast_specs,
)


def _md(name="op", spec=None):
    return OperatorMetadata(
        name=name,
        category="test",
        description="test",
        param_names=["close", "high_limit", "low_limit"],
        return_type="series",
        broadcast_specs=(spec,) if spec else (),
    )


def _panel(dates, vals=None, cols=("A",)):
    if vals is None:
        vals = np.arange(len(dates), dtype=float) + 1.0
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame(np.array(vals, dtype=float).reshape(-1, 1), index=idx, columns=list(cols))


def test_named_spec_verifies_declared_pair_only():
    spec = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="high_limit", target_param="close",
    )
    # base minute panel: all on 01-02; high_limit daily subset 01-02 -> in base.
    close = _panel(["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-02 09:33"])
    high_limit = _panel(["2024-01-02"])
    low_limit = _panel(["2024-01-04"])  # intentionally has a date ABSENT from base
    md = _md(spec=spec)
    # Only the high_limit<->close pair is verified; low_limit (absent-date) is
    # NOT touched by this spec — named binding must not sweep all frames.
    _verify_broadcast_specs(md, (spec,), [close, high_limit, low_limit],
                            param_names=("close", "high_limit", "low_limit"))


def test_absent_date_in_target_pair_fails_closed():
    spec = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="high_limit", target_param="close",
    )
    close = _panel(["2024-01-02 09:31"])
    high_limit = _panel(["2024-01-03"])  # 01-03 absent from base
    md = _md(spec=spec)
    with pytest.raises(ValueError):
        _verify_broadcast_specs(md, (spec,), [close, high_limit],
                                param_names=("close", "high_limit"))


def test_unknown_param_name_fails_closed():
    spec = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="not_a_panel", target_param="close",
    )
    close = _panel(["2024-01-02 09:31"])
    high_limit = _panel(["2024-01-02"])
    md = _md(spec=spec)
    with pytest.raises(ValueError):
        _verify_broadcast_specs(md, (spec,), [close, high_limit],
                                param_names=("close", "high_limit"))


def test_one_sided_spec_fails_closed():
    spec = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="high_limit", target_param=None,
    )
    close = _panel(["2024-01-02 09:31"])
    high_limit = _panel(["2024-01-02"])
    md = _md(spec=spec)
    with pytest.raises(ValueError):
        _verify_broadcast_specs(md, (spec,), [close, high_limit],
                                param_names=("close", "high_limit"))


def test_kwargs_reorder_does_not_switch_spec_pair():
    # Two specs bound to high_limit and low_limit respectively.  Swapping the
    # kwargs order must NOT make the low_limit spec verify high_limit.
    spec_high = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="high_limit", target_param="close",
    )
    spec_low = BroadcastSpec(
        mode="daily_to_minute", date_mapping="trading_date",
        instrument_policy="exact", source_param="low_limit", target_param="close",
    )
    close = _panel(["2024-01-02 09:31", "2024-01-03 09:31"])
    high_limit = _panel(["2024-01-02", "2024-01-03"])
    low_limit = _panel(["2024-01-04"])  # absent from base -> low spec must fail
    md = OperatorMetadata(
        name="op", category="test", description="test",
        param_names=["close", "high_limit", "low_limit"],
        return_type="series", broadcast_specs=(spec_high, spec_low),
    )
    # reverse kwargs order: low_limit first, high_limit second
    frames = [close, low_limit, high_limit]
    names = ("close", "low_limit", "high_limit")
    with pytest.raises(ValueError):
        _verify_broadcast_specs(md, (spec_high, spec_low), frames, param_names=names)
