# -*- coding: utf-8 -*-
"""P0 fix — PanelDataset enterprise data constraints.

A pooled predictive panel MUST carry an instrument id (``stock_col``) and MUST
NOT contain duplicated ``(stock, date)`` observations (a duplicate would be
counted as an independent training sample, inflating raw_obs / weights / loss).

* ``stock_col`` missing -> ``ValueError`` at construction;
* duplicated ``(stock, date)`` -> ``ValueError`` by default
  (``duplicate_policy="error"``), or deduped keeping the first occurrence with
  ``duplicate_policy="aggregate"``.

The check runs in ``__post_init__`` so every construction path (constructor and
``from_frame``) is covered.
"""
from __future__ import annotations

import pandas as pd
import pytest

from modeling.dataset import PanelDataset


def _dup_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [1, 2, 2, 3],
            "stock": ["A", "A", "A", "B"],
            "x0": [1.0, 2.0, 2.0, 3.0],
            "x1": [0.5, 1.5, 1.5, 2.5],
            "label": [0.1, 0.2, 0.2, 0.3],
        }
    )


def test_missing_stock_col_raises():
    frame = _dup_frame().drop(columns=["stock"])
    with pytest.raises(ValueError, match="stock column"):
        PanelDataset(
            frame=frame, date_col="date", stock_col="stock",
            feature_cols=["x0", "x1"], label_col="label",
        )


def test_missing_date_col_raises():
    frame = _dup_frame().drop(columns=["date"])
    with pytest.raises(ValueError, match="date column"):
        PanelDataset(
            frame=frame, date_col="date", stock_col="stock",
            feature_cols=["x0", "x1"], label_col="label",
        )


def test_duplicate_stock_date_raises_by_default():
    with pytest.raises(ValueError, match="duplicated"):
        PanelDataset(
            frame=_dup_frame(), date_col="date", stock_col="stock",
            feature_cols=["x0", "x1"], label_col="label",
        )


def test_duplicate_policy_aggregate_dedupes():
    ds = PanelDataset(
        frame=_dup_frame(), date_col="date", stock_col="stock",
        feature_cols=["x0", "x1"], label_col="label",
        duplicate_policy="aggregate",
    )
    assert ds.n_rows == 3  # (2, A) appeared twice -> one kept
    # keep="first" semantics
    dup_rows = _dup_frame()[(_dup_frame()["date"] == 2) & (_dup_frame()["stock"] == "A")]
    first_x0 = float(dup_rows["x0"].iloc[0])
    assert ds.frame.loc[(ds.frame["date"] == 2) & (ds.frame["stock"] == "A"), "x0"].iloc[0] == first_x0


def test_duplicate_policy_via_from_frame():
    ds = PanelDataset.from_frame(
        _dup_frame(), date_col="date", stock_col="stock",
        feature_cols=["x0", "x1"], label_col="label", duplicate_policy="aggregate",
    )
    assert ds.n_rows == 3
    with pytest.raises(ValueError, match="duplicated"):
        PanelDataset.from_frame(
            _dup_frame(), date_col="date", stock_col="stock",
            feature_cols=["x0", "x1"], label_col="label",
        )


def test_invalid_duplicate_policy_raises():
    with pytest.raises(ValueError, match="duplicate_policy"):
        PanelDataset(
            frame=_dup_frame().drop_duplicates(subset=["date", "stock"]),
            date_col="date", stock_col="stock",
            feature_cols=["x0", "x1"], label_col="label",
            duplicate_policy="bogus",
        )


def test_telemetry_supplies_missing_fraction_and_date_coverage():
    ds = PanelDataset.from_frame(
        _dup_frame(), date_col="date", stock_col="stock",
        feature_cols=["x0", "x1"], label_col="label", duplicate_policy="aggregate",
    )
    t = ds.telemetry()
    assert "missing_fraction" in t
    assert "date_coverage" in t
    assert t["date_coverage"] == 1.0  # integer bar ordinals: no calendar notion
    assert t["raw_obs"] == 3
