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

def test_loaded_cos_universe_uses_purged_training_not_warmup(monkeypatch):
    # Defect: using the first 300 dates rewards warmup coverage and selects b,
    # although a has better coverage on the optimizer's actual training dates.
    from types import SimpleNamespace
    import pyarrow as pa
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
    import real_batch_audit
    from factor_preprocess.contracts.treatment_lineage import TransformLineage
    from factor_optimizer.research_batch import automatic_time_split
    calendar = pd.bdate_range("2022-01-03", periods=510)
    dates = calendar[-502:-2]
    panel = pd.DataFrame({"timestamp": calendar,
                          "a.SZ": np.ones(510), "b.SZ": np.ones(510)})
    panel.loc[panel.timestamp.isin(dates[:30]), "a.SZ"] = np.nan
    panel.loc[panel.timestamp.isin(dates[50:53]), "b.SZ"] = np.nan
    sha = "a" * 64
    record = dict(uri=f"{example.POOL}/{sha}/test.parquet", sha256=sha,
                  bytes=100, verified=True, status="evaluated_optimization_pending")
    monkeypatch.setattr(example, "read_declared_cos_object",
        lambda *a, **k: SimpleNamespace(table=pa.Table.from_pylist([{"factors": {"test": record}}])))
    bound = SimpleNamespace(
        factor=SimpleNamespace(table=pa.Table.from_pandas(panel), source_uri=record["uri"],
                               source_etag="test", content_sha256=sha, downloaded_bytes=100),
        treatment_signature=TransformLineage(()), manifest_sha256="b"*64,
        source_status=record["status"], expression="col('Volume')")
    monkeypatch.setattr(example, "read_bound_factor", lambda *a, **k: bound)
    monkeypatch.setattr(real_batch_audit, "DAILY_ADJ",
        SimpleNamespace(glob=lambda _: [Path(str(d.date()) + ".parquet") for d in calendar]))
    monkeypatch.setattr(real_batch_audit, "_load_vwap",
        lambda start, end, assets: pd.DataFrame(100., index=calendar, columns=assets))
    batch, labels, provenance = example.load_cos_sample(n_factors=1, n_assets=1)
    assert list(batch.asset_axis.values) == ["a.SZ"]
    split = automatic_time_split(labels)
    assert len(split.train_indices) == 267
    assert provenance["asset_selection_split"] == split.identity
    assert provenance["asset_selection_train_days"] == 267
