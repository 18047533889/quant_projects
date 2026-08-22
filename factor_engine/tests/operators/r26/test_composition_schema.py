# -*- coding: utf-8 -*-
"""R26-096..102: composition schema economic validity.

* revenue / net_income / total_assets are NOT valid generic part-whole parts;
* net_profit / net_income alias duplicate is rejected;
* standard wide-panel columns (instrument codes) are never PartIds;
* genuine custom share-of-total parts still pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.composition import _part_field_name, _same_composition


def _part(name, n=5):
    idx = pd.date_range("2024-01-01", periods=n)
    df = pd.DataFrame(np.linspace(1.0, float(n), n)[:, None], index=idx, columns=["v"])
    df.name = name
    return df


def test_financial_statement_fields_rejected():
    with pytest.raises(ValueError, match="not a mutually-exclusive part|double-count"):
        _same_composition([_part("revenue"), _part("net_income"), _part("total_assets")])


def test_alias_duplicate_rejected():
    with pytest.raises(ValueError, match="not a mutually-exclusive part|duplicate"):
        _same_composition([_part("net_profit"), _part("net_income"), _part("revenue")])


def test_wide_panel_instrument_column_not_part_id():
    idx = pd.date_range("2024-01-01", periods=3)
    wide = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0]}, index=idx)
    assert _part_field_name(wide) is None


def test_custom_share_parts_pass():
    _same_composition(
        [_part("share_of_volume_bucket_1"), _part("share_of_volume_bucket_2"),
         _part("share_of_volume_bucket_3")]
    )  # no raise
