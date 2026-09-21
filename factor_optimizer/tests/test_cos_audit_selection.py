import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "cos_audit", Path(__file__).parents[1] / "examples/cos_batch_audit.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_manifest_sample_keeps_nested_uri_and_excludes_blocked_factors():
    sha = "a"*64
    def record(name, status):
        return dict(uri=example.POOL+"/"+sha+"/"+name+".parquet", sha256=sha,
                    bytes=100, verified=True, status=status)
    records = {"blocked": record("blocked", "materialized_quality_blocked"),
               "good": record("good", "evaluated_optimization_pending")}
    selected = example.select_manifest_records([{"factors": records}], 1)
    assert selected == [("good", records["good"])]
    with pytest.raises(ValueError, match="eligible"):
        example.select_manifest_records([{"factors": records}], 2)
    records["good"]["uri"] = "cos://other-bucket/"+sha+"/good.parquet"
    with pytest.raises(ValueError, match="URI"):
        example.select_manifest_records([{"factors": records}], 1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_manifest_sample_requires_bounded_positive_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        example.select_manifest_records([{"factors": {}}], count)


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
