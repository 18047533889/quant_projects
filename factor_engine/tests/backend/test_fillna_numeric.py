# -*- coding: utf-8 -*-
"""Production constant-fill numeric semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_fillna_const_numeric_zero():
    load_all()
    op = OperatorRegistry.get("fillna_const", backend="pandas_numpy")
    x = pd.DataFrame({"A": [np.nan, 2.0, np.nan]})
    out = op.calculate(x, 0)
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == 2.0
    assert out.iloc[2, 0] == 0.0
