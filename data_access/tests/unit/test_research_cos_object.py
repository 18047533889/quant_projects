"""Bounded, authorized exact COS reads through the real DataAccess store."""
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.core.storage import StorageSpec
from data_access.registry.loader import DatasetRegistry, ParametricDataset
from data_access.read.query_budget import QueryBudget
from data_access.store import DataAccessStore
from data_access.read.formats import FormatSpec


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DATA_ACCESS_COS_CLI", "test-gateway")
    ds = ParametricDataset(
        name="research_panel", access_mode="published", layout="plain",
        time_column=None, instrument_column=None, hive_partitioning=False,
        union_by_name=True, root_template=str(tmp_path / "unused"),
        glob_template="{factor_id}.parquet", params_schema={"factor_id": "str"},
        static_root=tmp_path / "unused",
        storage=StorageSpec(type="cos", uri="cos://bucket/pool", layout="plain"))
    engine = DuckDBEngine(threads=1)
    store = DataAccessStore(DatasetRegistry({ds.name: ds}), engine)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"x": [1., 2.], "y": [3., 4.]}), sink)
    payload = sink.getvalue().to_pybytes()
    metadata = {"key": "pool/f.parquet", "size": len(payload),
                "etag": hashlib.md5(payload).hexdigest()}
    from data_access.cos import remote
    monkeypatch.setattr(remote, "cos_cli_head", lambda *a, **k: dict(metadata))
    calls = []
    def gateway(cmd, **kwargs):
        assert cmd[:3] == ["test-gateway", "cp", "cos://bucket/pool/f.parquet"]
        assert 0 < kwargs["timeout"] <= 60
        calls.append(cmd)
        Path(cmd[3]).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", gateway)
    yield store, metadata, calls, tmp_path
    engine.close()


def read(source, **kwargs):
    from data_access.cos.research import read_declared_cos_object
    return read_declared_cos_object(
        source[0], "research_panel", params={"factor_id": "f"},
        allow_research=True, **kwargs)


@pytest.fixture
def json_source(source, monkeypatch):
    payload = b'{"factor_id":"f","expression":"cs_rank(close)","unrelated":42}'
    ds = replace(source[0]._registry.get('research_panel'),
                 glob_template='{factor_id}.json', format_spec=FormatSpec.from_yaml('json'))
    engine = DuckDBEngine(threads=1)
    store = DataAccessStore(DatasetRegistry({ds.name: ds}), engine)
    source[1].update(key='pool/f.json', size=len(payload), etag=hashlib.md5(payload).hexdigest())
    def gateway(cmd, **kwargs):
        assert cmd[2] == 'cos://bucket/pool/f.json'
        source[2].append(cmd)
        Path(cmd[3]).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', gateway)
    yield store, source[1], source[2], source[3]
    engine.close()


def test_declared_json_metadata_uses_dataaccess_projection_and_cleanup(json_source):
    result = read(json_source, columns=['factor_id', 'expression'])
    assert result.table.to_pydict() == {'factor_id': ['f'], 'expression': ['cs_rank(close)']}
    assert result.source_uri == 'cos://bucket/pool/f.json'
    assert not list((json_source[3] / 'cache').rglob('*.json'))


def test_json_metadata_has_small_admission_budget(json_source):
    json_source[1]['size'] = 3*1024**2
    with pytest.raises(ValidationError, match='budget'):
        read(json_source)
    assert json_source[2] == []


def test_recursive_metadata_listing_can_discover_nested_json(monkeypatch):
    from data_access.cos.remote import cos_cli_ls
    def gateway(cmd, **kwargs):
        text = ('pool/hash/recipe.json | STANDARD | 2026-09-22 | "12345678abcdef" | 100 B |'
                if '--recursive' in cmd else 'pool/hash/ | DIR | | | |')
        return subprocess.CompletedProcess(cmd, 0, text, '')
    monkeypatch.setattr(subprocess, 'run', gateway)
    assert cos_cli_ls('cos://bucket/pool/', cli='test-gateway') == []
    rows = cos_cli_ls('cos://bucket/pool/', cli='test-gateway', recursive=True)
    assert [(r['key'], r['size']) for r in rows] == [('pool/hash/recipe.json', 100)]


def test_real_store_projection_and_ephemeral_cleanup(source):
    result = read(source, columns=["x"])
    assert result.table.to_pydict() == {"x": [1., 2.]}
    assert result.source_uri == "cos://bucket/pool/f.parquet"
    assert result.source_etag == source[1]["etag"]
    assert len(result.content_sha256) == 64
    assert len(source[2]) == 1
    assert not list((source[3] / "cache").rglob("*.parquet"))


def test_download_budget_rejects_before_transfer(source):
    with pytest.raises(ValidationError, match="budget|预算"):
        read(source, query_budget=QueryBudget(max_scan_bytes=10))
    assert source[2] == []


def test_mismatched_object_metadata_rejects_before_transfer(source):
    source[1]["key"] = "pool/other.parquet"
    with pytest.raises(ValidationError, match="object|对象"):
        read(source)
    assert source[2] == []


def test_content_digest_mismatch_fails_and_cleans_up(source):
    source[1]["etag"] = "0" * 32
    with pytest.raises(ValidationError, match="digest|摘要"):
        read(source)
    assert not list((source[3] / "cache").rglob("*.parquet"))


def test_explicit_opt_in_and_strict_rejection(source, monkeypatch):
    from data_access.cos.research import read_declared_cos_object
    with pytest.raises(ValidationError, match="research"):
        read_declared_cos_object(source[0], "research_panel", params={"factor_id": "f"})
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    with pytest.raises(ValidationError, match="research|strict"):
        read(source)
    assert source[2] == []


@pytest.mark.parametrize("factor_id", ["../escape", "*", "f/other"])
def test_parameter_paths_cannot_expand_scope(source, factor_id):
    from data_access.cos.research import read_declared_cos_object
    with pytest.raises((ValidationError, ValueError)):
        read_declared_cos_object(source[0], "research_panel",
                                 params={"factor_id": factor_id}, allow_research=True)
    assert source[2] == []


def test_dataset_authorization_precedes_network(source, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied dataset")
    monkeypatch.setattr(source[0], "authorize_dataset", denied)
    with pytest.raises(PermissionError, match="denied"):
        read(source)
    assert source[2] == []


def test_remote_change_during_transfer_fails_and_cleans_up(source, monkeypatch):
    from data_access.cos import remote
    count = [0]
    def changing(*args, **kwargs):
        count[0] += 1
        meta = dict(source[1])
        if count[0] > 1:
            meta["etag"] = "replacement-version"
        return meta
    monkeypatch.setattr(remote, "cos_cli_head", changing)
    with pytest.raises(ValidationError, match="changed"):
        read(source)
    assert not list((source[3] / "cache").rglob("*.parquet"))


def test_unknown_object_size_is_not_assumed_small(source):
    source[1]["size"] = None
    with pytest.raises(ValidationError, match="metadata"):
        read(source)
    assert source[2] == []


def test_transfer_failure_cleans_up_and_does_not_expose_gateway_error(source, monkeypatch):
    def failed(cmd, **kwargs):
        Path(cmd[3]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(77, cmd, stderr="private provider detail")
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValidationError, match="transfer failed") as caught:
        read(source)
    assert "private provider" not in str(caught.value)
    assert not list((source[3] / "cache").rglob("*.parquet"))
