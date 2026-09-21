"""The real-data example must choose its universe on exact TRAIN dates."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from data_access.core.exceptions import ValidationError

spec = importlib.util.spec_from_file_location(
    "real_audit", Path(__file__).parents[1] / "examples/real_batch_audit.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_universe_uses_requested_train_dates_not_file_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_READ_URI_ROOTS", str(tmp_path))
    monkeypatch.setattr(example, "N_ASSETS", 1)
    dates = pd.date_range("2024-01-01", periods=20)
    # A is valid on TRAIN only; B on future dates only. Files are unsorted.
    panel = pd.DataFrame({"a.SZ": [1.]*4 + [np.nan]*16,
                          "b.SZ": [np.nan]*4 + [1.]*16}, index=dates)
    panel.index.name = "date"
    path = tmp_path / "factor.parquet"
    panel.iloc[::-1].to_parquet(path)
    assert example._choose_assets([path], dates[:4]) == ["a.SZ"]


def test_real_sample_reader_respects_dataaccess_path_authorization(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_READ_URI_ROOTS", raising=False)
    monkeypatch.delenv("DATA_ACCESS_EXTRA_ALLOWED_ROOTS", raising=False)
    path = tmp_path / "outside_authorized_roots.parquet"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2),
                  "a.SZ": [1., 2.]}).to_parquet(path, index=False)
    with pytest.raises(ValidationError, match="白名单"):
        example._load_factor(path, ["a.SZ"])
