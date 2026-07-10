# -*- coding: utf-8
"""numeric_semantics 契约。"""
from __future__ import annotations

from backend.numeric_semantics import semantics_for, std_ddof_value


def test_protected_div_semantics():
    sem = semantics_for("protected_div")
    assert sem.div_zero == "default"


def test_ts_std_sample_ddof():
    assert std_ddof_value("ts_std") == 1


def test_zscore_zero_std():
    assert semantics_for("zscore").zscore_zero_std == "zero"


def test_zscore_zero_std_fill():
    from backend.numeric_semantics import zscore_zero_std_fill

    assert zscore_zero_std_fill("zscore") == 0.0
    assert zscore_zero_std_fill("group_zscore") == 0.0


def test_group_rank_semantics():
    assert semantics_for("group_rank").rank_ignore_nan is True


def test_sql_stddev_fn_key_sample():
    from backend.numeric_semantics import sql_stddev_fn_key

    assert sql_stddev_fn_key("group_zscore") == "stddev"
