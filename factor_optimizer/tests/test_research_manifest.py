import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
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
    heads = []
    def payload(uri):
        return (json.dumps({"factors": {"f": record}}).encode()
                if uri.endswith(".json") else factor)
    def head(uri, **kwargs):
        heads.append(uri)
        data = payload(uri)
        return dict(key=uri.removeprefix("cos://bucket/"), size=len(data),
                    etag=hashlib.md5(data).hexdigest())
    real_run = subprocess.run
    def gateway(cmd, **kwargs):
        if not cmd or cmd[0] != "test-gateway":
            return real_run(cmd, **kwargs)
        transfers.append(cmd[2])
        Path(cmd[3]).write_bytes(payload(cmd[2]))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    from data_access.cos import remote
    monkeypatch.setattr(remote, "cos_cli_head", head)
    monkeypatch.setattr(subprocess, "run", gateway)
    yield store, record, transfers, tmp_path, heads
    engine.close()


def test_gateway_fixture_leaves_unrelated_subprocesses_real(declared_source, tmp_path):
    import sys

    output = tmp_path / "native-process-output"
    result = subprocess.run(
        [sys.executable, "-c", "print('native')", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "native"
    assert not output.exists()


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


@pytest.mark.parametrize("alias", ["rank", "cs_rank", "CS_RANK", "RANK", "c_rank", "cs_rank_01"])
def test_rank_scope_recognizes_factor_engine_rank_aliases(alias):
    from factor_optimizer.research_manifest import _rank_scope

    contains, output = _rank_scope(f"{alias}(col('close'))")
    assert contains and output


def test_rank_scope_excludes_rank_corr_and_panel_rank():
    from factor_optimizer.research_manifest import _rank_scope

    assert _rank_scope("rank_corr(col('close'), col('volume'))") == (False, False)
    assert _rank_scope("panel_rank(col('close'))") == (False, False)


def test_cs_rank_01_is_exact_duplicate():
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    import numpy as np
    import pandas as pd
    from factor_optimizer.research_manifest import _rank_scope

    rank = OperatorRegistry.get("rank", backend="pandas_numpy", mode="any")
    cs_rank_01 = OperatorRegistry.get("cs_rank_01", backend="pandas_numpy", mode="any")
    sample = pd.DataFrame([[1.0, 1.0, 3.0], [np.nan, 2.0, 4.0]])
    pd.testing.assert_frame_equal(rank.calculate(sample), cs_rank_01.calculate(sample))
    assert _rank_scope("cs_rank_01(col('close'))") == (True, True)



def test_reused_manifest_keeps_each_factor_head_get_head_and_binding_checks(declared_source):
    from factor_optimizer.research_manifest import read_bound_factor, read_bound_manifest

    store, record, transfers, _, heads = declared_source
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    manifest_uri = "cos://bucket/pool/landing_manifest.json"
    assert snapshot.manifest_uri == manifest_uri
    assert snapshot.manifest_sha256 == hashlib.sha256(
        json.dumps({"factors": {"f": record}}).encode()).hexdigest()
    assert transfers == [manifest_uri]
    assert heads == [manifest_uri, manifest_uri]

    heads.clear()
    transfers.clear()
    for _ in range(2):
        bound = read_bound_factor(store, "manifest", "factor", "f",
                                  allow_research=True, manifest_snapshot=snapshot)
        assert bound.manifest_sha256 == snapshot.manifest_sha256
        assert bound.source_status == record["status"]
        assert bound.expression == record["fe_dsl"]
    factor_uri = "cos://bucket/pool/f.parquet"
    assert transfers == [factor_uri, factor_uri]
    assert heads == [factor_uri] * 4


def test_reused_manifest_rejects_invalid_status_or_uri_before_factor_get(declared_source):
    from factor_optimizer.research_manifest import read_bound_factor, read_bound_manifest

    store, record, transfers, _, _ = declared_source
    record["status"] = "materialized_quality_blocked"
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    transfers.clear()
    with pytest.raises(ValueError, match="status"):
        read_bound_factor(store, "manifest", "factor", "f", allow_research=True,
                          manifest_snapshot=snapshot)
    assert transfers == []

    record["status"] = "evaluated_optimization_pending"
    record["uri"] = "cos://bucket/pool/other.parquet"
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    transfers.clear()
    with pytest.raises(ValueError, match="URI"):
        read_bound_factor(store, "manifest", "factor", "f", allow_research=True,
                          manifest_snapshot=snapshot)
    assert transfers == []


def test_reused_manifest_must_match_registered_manifest_dataset(declared_source):
    from dataclasses import replace
    from factor_optimizer.research_manifest import read_bound_factor, read_bound_manifest

    store, _, transfers, _, _ = declared_source
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    transfers.clear()
    with pytest.raises(ValueError, match="snapshot identity"):
        read_bound_factor(store, "manifest", "factor", "f", allow_research=True,
                          manifest_snapshot=replace(snapshot, manifest_uri="cos://other/manifest.json"))
    assert transfers == []


def test_manifest_snapshot_digest_normalizes_type_tagged_datetimes():
    from factor_optimizer.research_manifest import _manifest_factors_sha256

    naive = datetime(2026, 9, 29, 12, 34, 56, 123456)
    aware = datetime(2026, 9, 29, 12, 34, 56, 123456,
                     tzinfo=timezone.utc, fold=1)
    assert _manifest_factors_sha256({"when": naive}) == _manifest_factors_sha256(
        {"when": datetime(2026, 9, 29, 12, 34, 56, 123456)})
    assert _manifest_factors_sha256({"when": naive}) != _manifest_factors_sha256(
        {"when": naive.isoformat(timespec="microseconds")})
    assert _manifest_factors_sha256({"when": aware}) != _manifest_factors_sha256(
        {"when": datetime(2026, 9, 29, 12, 34, 56, 123456, tzinfo=timezone.utc)})
    with pytest.raises(ValueError, match="unsupported value type"):
        _manifest_factors_sha256({"when": object()})


def test_forged_reused_manifest_snapshot_is_rejected(declared_source):
    from dataclasses import replace
    from factor_optimizer.research_manifest import read_bound_factor, read_bound_manifest

    store, _, transfers, _, _ = declared_source
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    forged_factors = {"f": dict(snapshot.factors["f"], status="materialized_quality_blocked")}
    forged = replace(snapshot, factors=forged_factors)
    transfers.clear()
    with pytest.raises(ValueError, match="snapshot identity"):
        read_bound_factor(store, "manifest", "factor", "f", allow_research=True,
                          manifest_snapshot=forged)
    assert transfers == []


def test_manifest_change_during_factor_pass_is_detected(declared_source):
    from factor_optimizer.research_manifest import (
        read_bound_manifest, verify_bound_manifest_unchanged,
    )

    store, record, _, _, _ = declared_source
    snapshot = read_bound_manifest(store, "manifest", allow_research=True)
    record["fe_dsl"] = "rank(col('changed'))"
    with pytest.raises(ValueError, match="identity changed"):
        verify_bound_manifest_unchanged(store, "manifest", snapshot)


def test_bound_reader_passes_object_opt_in_only_to_factor_read(declared_source, monkeypatch):
    from data_access.cos import research
    from factor_optimizer.research_manifest import read_bound_factor

    original = research.read_declared_cos_object
    calls = []

    def tracked(store, dataset, **kwargs):
        calls.append((dataset, kwargs.get("max_object_mib")))
        return original(store, dataset, **kwargs)

    monkeypatch.setattr(research, "read_declared_cos_object", tracked)
    read_bound_factor(declared_source[0], "manifest", "factor", "f",
                      allow_research=True, max_object_mib=128)
    assert calls == [("manifest", None), ("factor", 128)]


@pytest.mark.parametrize("value", [0, 129, True, 1.5])
def test_bound_reader_rejects_unbounded_object_opt_in_before_manifest(value, declared_source):
    from factor_optimizer.research_manifest import read_bound_factor

    with pytest.raises(ValueError, match="max_object_mib"):
        read_bound_factor(declared_source[0], "manifest", "factor", "f",
                          allow_research=True, max_object_mib=value)
    assert declared_source[2] == []

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
    from factor_optimizer.research_baseline import compile_baseline
    assert bound.treatment_signature.cs_rank
    assert bound.treatment_signature.is_unknown_or_incomplete
    plan = compile_baseline(bound.treatment_signature, training_context_ref='source')
    assert plan.operations == ()
    assert 'cs_rank_already_present' in plan.omissions


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
