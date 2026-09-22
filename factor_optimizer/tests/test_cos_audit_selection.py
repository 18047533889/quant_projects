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

@pytest.mark.parametrize("coverage_case", ["single", "sparse", "future_missing", "all_missing"])
def test_loaded_cos_universe_uses_purged_training_not_warmup(monkeypatch, coverage_case):
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
    records = {"test": record}
    if coverage_case != "single":
        records["sparse"] = dict(record, uri=f"{example.POOL}/{sha}/sparse.parquet")
    monkeypatch.setattr(example, "read_declared_cos_object",
        lambda *a, **k: SimpleNamespace(table=pa.Table.from_pylist([{"factors": records}])))
    def bound_read(*args, **kwargs):
        name = args[-1]
        values = panel.copy()
        if coverage_case == "all_missing" or (name == "sparse" and coverage_case == "sparse"):
            values[["a.SZ", "b.SZ"]] = np.nan
        elif name == "sparse" and coverage_case == "future_missing":
            values.loc[values.timestamp >= dates[300], ["a.SZ", "b.SZ"]] = np.nan
        return SimpleNamespace(
        factor=SimpleNamespace(table=pa.Table.from_pandas(values), source_uri=records[name]["uri"],
                               source_etag="test", content_sha256=sha, downloaded_bytes=100),
        treatment_signature=TransformLineage(()), manifest_sha256="b"*64,
        source_status=record["status"], expression="col('Volume')")
    monkeypatch.setattr(example, "read_bound_factor", bound_read)
    monkeypatch.setattr(real_batch_audit, "DAILY_ADJ",
        SimpleNamespace(glob=lambda _: [Path(str(d.date()) + ".parquet") for d in calendar]))
    monkeypatch.setattr(real_batch_audit, "_load_vwap",
        lambda start, end, assets: pd.DataFrame(100., index=calendar, columns=assets))
    if coverage_case == "all_missing":
        with pytest.raises(ValueError, match="all requested factors"):
            example.load_cos_sample(n_factors=len(records), n_assets=1)
        return
    batch, labels, provenance, lineages = example.load_cos_sample(
        n_factors=len(records), n_assets=1, include_lineages=True)
    expected_ids = ("sparse", "test") if coverage_case == "future_missing" else ("test",)
    assert batch.factor_ids == expected_ids
    assert set(lineages) == set(expected_ids)
    if coverage_case == "sparse":
        assert provenance["quarantined_factors"] == [{"factor": "sparse",
            "eligible_assets": 0, "required_assets": 1,
            "reason": "insufficient_training_coverage"}]
        assert len(provenance["sources"]) == 2
        with pytest.raises(ValueError, match="TRAIN-covered"):
            example.load_cos_sample(n_factors=2, n_assets=1, coverage_policy="strict")
    elif coverage_case == "future_missing":
        assert provenance["quarantined_factors"] == []
    assert list(batch.asset_axis.values) == ["a.SZ"]
    split = automatic_time_split(labels)
    assert len(split.train_indices) == 267
    assert provenance["asset_selection_split"] == split.identity
    assert provenance["asset_selection_train_days"] == 267

def test_cli_exposes_bounded_manifest_sampling():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(Path(example.__file__)), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--manifest" in result.stdout
    assert "--factors" in result.stdout
    assert "--max-factor-mib" in result.stdout


def test_larger_factors_are_opt_in_and_total_download_is_bounded():
    sha = "c" * 64
    records = {f"f{i}": dict(uri=f"{example.POOL}/{sha}/f{i}.parquet", sha256=sha,
                           bytes=48*1024**2, verified=True,
                           status="evaluated_optimization_pending") for i in range(3)}
    with pytest.raises(ValueError, match="eligible"):
        example.select_manifest_records([{"factors": records}], 1)
    selected = example.select_manifest_records([{"factors": records}], 2,
                                               max_factor_bytes=64*1024**2)
    assert [name for name, _ in selected] == ["f0", "f1"]
    with pytest.raises(ValueError, match="batch"):
        example.select_manifest_records([{"factors": records}], 3,
                                        max_factor_bytes=64*1024**2)


@pytest.mark.parametrize("uri", [
    "cos://other/metadata/" + "a"*64 + "/landing_manifest.json",
    example.MANIFEST + "/../landing_manifest.json",
    example.MANIFEST + "/landing_manifest.json?token=secret",
    "/tmp/landing_manifest.json",
])
def test_manifest_selection_rejects_out_of_pool_before_io(monkeypatch, uri):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid manifest must not construct a data engine")
    monkeypatch.setattr(example, "DuckDBEngine", forbidden)
    with pytest.raises(ValueError, match="manifest"):
        example.load_cos_sample(manifest_uri=uri)


@pytest.mark.parametrize("cap", [True, 0, -1, 65*1024**2, 1.5])
def test_factor_budget_cannot_exceed_existing_dataaccess_cap(cap):
    with pytest.raises(ValueError, match="max_factor_bytes"):
        example.select_manifest_records([{"factors": {}}], 1, max_factor_bytes=cap)
