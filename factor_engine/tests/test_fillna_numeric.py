# -*- coding: utf-8 -*-
"""fillna 数值参数语义单测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


def test_fillna_numeric_zero():
    load_all()
    op = OperatorRegistry.get("fillna", backend="pandas_numpy")
    x = pd.DataFrame({"A": [np.nan, 2.0, np.nan]})
    out = op.calculate(x, 0)
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == 2.0
    assert out.iloc[2, 0] == 0.0
