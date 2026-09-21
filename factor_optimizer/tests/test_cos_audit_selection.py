import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "cos_audit", Path(__file__).parents[1] / "examples/cos_batch_audit.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_cos_universe_ignores_future_coverage():
    dates = pd.date_range("2024-01-01", periods=20)
    p = pd.DataFrame({"a.SZ": [1.]*4+[np.nan]*16,
                      "b.SZ": [np.nan]*4+[1.]*16}, index=dates)
    assert example.choose_assets([p], dates[:4], n_assets=1) == ["a.SZ"]


@pytest.mark.parametrize("n_assets", [0, -1, True, 1.5])
def test_cos_universe_invalid_size_is_rejected(n_assets):
    p = pd.DataFrame({"a.SZ": [1.]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="positive integer"):
        example.choose_assets([p], p.index, n_assets=n_assets)
