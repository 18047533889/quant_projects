from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


def _panel():
    return pd.DataFrame(
        {"A": [0.0, 1.0, np.nan], "B": [2.0, -1.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_scalar_left_comparison_broadcasts():
    load_all()
    out = OperatorRegistry.get("lt").calculate(0.25, _panel())
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[0, 1] == 1.0
    assert np.isnan(out.iloc[2, 0])


def test_scalar_right_comparison_broadcasts():
    load_all()
    out = OperatorRegistry.get("ge").calculate(_panel(), 1.0)
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[1, 1] == 0.0
    assert np.isnan(out.iloc[2, 0])
