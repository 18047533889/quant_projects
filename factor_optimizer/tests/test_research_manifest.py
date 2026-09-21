import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.storage import StorageSpec
from data_access.registry.loader import DatasetRegistry, StaticDataset
from data_access.read.formats import FormatSpec
from data_access.store import DataAccessStore


@pytest.fixture
def declared_source(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_CLI", "test-gateway")
    monkeypatch.setenv("DATA_ACCESS_COS_CACHE_ROOT", str(tmp_path / "cache"))
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"timestamp": [1, 2], "a": [1., 2.]}), sink)
    factor = sink.getvalue().to_pybytes()
    uri = "cos://bucket/pool/f.parquet"
    record = dict(uri=uri, sha256=hashlib.sha256(factor).hexdigest(),
                  bytes=len(factor), verified=True, status="evaluated_optimization_pending",
                  fe_dsl="rank(subtract(col('close'), col('open')))")
    datasets = {}
    for name, filename, fmt in (("manifest", "landing_manifest.json", "json"),
                                ("factor", "f.parquet", "parquet")):
        datasets[name] = StaticDataset(name=name, access_mode="published", layout="plain",
            time_column=None, instrument_column=None, hive_partitioning=False,
            union_by_name=True, root=tmp_path / name, glob=filename,
            format_spec=FormatSpec.from_yaml(fmt),
            storage=StorageSpec(type="cos", uri="cos://bucket/pool", layout="plain"))
    engine = DuckDBEngine(threads=1)
    store = DataAccessStore(DatasetRegistry(datasets), engine)
    transfers = []
    def payload(uri):
        return (json.dumps({"factors": {"f": record}}).encode()
                if uri.endswith(".json") else factor)
    def head(uri, **kwargs):
        data = payload(uri)
        return dict(key=uri.removeprefix("cos://bucket/"), size=len(data),
                    etag=hashlib.md5(data).hexdigest())
    def gateway(cmd, **kwargs):
        transfers.append(cmd[2])
        Path(cmd[3]).write_bytes(payload(cmd[2]))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    from data_access.cos import remote
    monkeypatch.setattr(remote, "cos_cli_head", head)
    monkeypatch.setattr(subprocess, "run", gateway)
    yield store, record, transfers, tmp_path
    engine.close()


def read(source):
    assert importlib.util.find_spec("factor_optimizer.research_manifest") is not None, (
        "verified manifest-to-factor reader is missing")
    from factor_optimizer.research_manifest import read_bound_factor
    return read_bound_factor(source[0], "manifest", "factor", "f", allow_research=True)


def test_bound_reader_verifies_actual_data_and_reports_rank_scope(declared_source):
    bound = read(declared_source)
    assert bound.factor.table.to_pydict() == {"timestamp": [1, 2], "a": [1., 2.]}
    assert bound.factor.content_sha256 == declared_source[1]["sha256"]
    assert bound.contains_cs_rank and bound.output_is_cs_rank
    assert bound.manifest_sha256
    assert not list((declared_source[3] / "cache").rglob("object.*"))


@pytest.mark.parametrize("field,value", [
    ("uri", "cos://bucket/pool/other.parquet"),
    ("verified", False),
    ("status", "materialized_quality_blocked"),
    ("status", "unexpected_future_status"),
])
def test_invalid_binding_is_rejected_before_factor_transfer(declared_source, field, value):
    declared_source[1][field] = value
    with pytest.raises(ValueError):
        read(declared_source)
    assert declared_source[2] == ["cos://bucket/pool/landing_manifest.json"]


@pytest.mark.parametrize("field,value", [("sha256", "0"*64), ("bytes", 1)])
def test_downloaded_content_must_match_manifest(declared_source, field, value):
    declared_source[1][field] = value
    with pytest.raises(ValueError):
        read(declared_source)
    assert declared_source[2][-1] == "cos://bucket/pool/f.parquet"


def test_conditional_rank_is_not_final_output_rank(declared_source):
    declared_source[1]["fe_dsl"] = "where(gt(close, open), rank(close), neg(rank(close)))"
    bound = read(declared_source)
    assert bound.contains_cs_rank
    assert not bound.output_is_cs_rank


def test_rank_inside_column_string_is_not_an_operator(declared_source):
    declared_source[1]["fe_dsl"] = "col('rank')"
    bound = read(declared_source)
    assert not bound.contains_cs_rank
    assert not bound.output_is_cs_rank


@pytest.mark.parametrize("expression,operations", [
    ("ts_delta(col('valuation.free_cap'), 20)", ("winsor", "cs_rank")),
    ("rank(subtract(col('close'), col('open')))", ("winsor",)),
])
def test_known_bound_recipe_drives_baseline_without_repeating_rank(
        declared_source, expression, operations):
    from factor_optimizer.research_baseline import compile_baseline
    declared_source[1]["fe_dsl"] = expression
    bound = read(declared_source)
    assert bound.lineage is not None
    plan = compile_baseline(bound.lineage, training_context_ref="verified-manifest")
    assert plan.operations == operations


@pytest.mark.parametrize("expression", [
    "where(gt(close, open), rank(close), neg(rank(close)))",
    "undocumented_custom_transform(col('close'))",
])
def test_unknown_or_branched_recipe_is_not_claimed_untreated(declared_source, expression):
    declared_source[1]["fe_dsl"] = expression
    assert read(declared_source).lineage is None
