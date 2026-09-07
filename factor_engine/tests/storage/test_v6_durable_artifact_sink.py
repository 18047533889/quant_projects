import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
from factor_engine.runtime.durable_artifact_sink import (
    ArtifactResourceRequirementError, write_verified_factor_artifact,
    verify_factor_artifact_receipt,
    reconcile_factor_artifact,
)


def test_cell_fingerprint_temporaries_are_bounded_and_preserve_zero_sign(monkeypatch):
    from factor_engine.runtime.durable_artifact_sink import _cell_fingerprints_equal

    left = pd.Series(np.zeros(20001))
    right = left.copy()
    calls = []
    original = pd.util.hash_pandas_object

    def observed(value, **kwargs):
        calls.append(len(value))
        return original(value, **kwargs)

    monkeypatch.setattr(pd.util, "hash_pandas_object", observed)
    assert _cell_fingerprints_equal(left, right)
    assert len(calls) == 6 and max(calls) <= 8192
    right.iloc[-1] = -0.0
    assert not _cell_fingerprints_equal(left, right)


def values():
    return pd.Series([0., -0., 3., np.nan, 5., 6.], name="test",
                     index=pd.MultiIndex.from_product(
                         [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
                         names=["timestamp", "instrument"]), dtype="float64")


def test_known_generation_reconciliation_is_exact_and_read_only(tmp_path):
    generation = "a" * 32
    receipt = write_verified_factor_artifact(
        tmp_path, "isolated", 0, "test", values(), generation=generation,
        policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    recovered = reconcile_factor_artifact(
        tmp_path, "isolated", 0, "test", values(), generation=generation,
        policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert recovered["verified"] and recovered["reconciled"]
    assert recovered["sha256"] == receipt["sha256"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        write_verified_factor_artifact(
            tmp_path, "isolated", 0, "test", values(), generation=generation,
            policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_reconciliation_does_not_search_other_generations_or_rewrite(tmp_path):
    write_verified_factor_artifact(
        tmp_path, "isolated", 0, "test", values(), generation="a" * 32,
        policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileNotFoundError):
        reconcile_factor_artifact(
            tmp_path, "isolated", 0, "test", values(), generation="b" * 32,
            policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_reconciliation_rejects_self_consistent_wrong_output(tmp_path):
    wrong = values()
    wrong.iloc[2] = 99
    write_verified_factor_artifact(
        tmp_path, "isolated", 0, "test", wrong, generation="a" * 32,
        policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    with pytest.raises(AssertionError):
        reconcile_factor_artifact(
            tmp_path, "isolated", 0, "test", values(), generation="a" * 32,
            policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_reconciliation_cannot_claim_durability_after_sync_failure(tmp_path, monkeypatch):
    write_verified_factor_artifact(
        tmp_path, "isolated", 0, "test", values(), generation="a" * 32,
        policy=DefaultExecutionPolicy(), budget_bytes=2**20)

    def sync_failure(*args):
        raise OSError("injected durability failure")

    monkeypatch.setattr("factor_engine.runtime.durable_artifact_sink.os.fsync", sync_failure)
    with pytest.raises(OSError, match="durability failure"):
        reconcile_factor_artifact(
            tmp_path, "isolated", 0, "test", values(), generation="a" * 32,
            policy=DefaultExecutionPolicy(), budget_bytes=2**20)


@pytest.mark.parametrize("generation", ["../bad", "", "A" * 32, True])
def test_invalid_intent_generation_refused_before_writing(tmp_path, generation):
    with pytest.raises(ValueError, match="generation"):
        write_verified_factor_artifact(
            tmp_path, "isolated", 0, "test", values(), generation=generation,
            policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert not list(tmp_path.iterdir())


def test_real_writer_durable_exact_readback_and_no_publication(tmp_path):
    source = values()
    receipt = write_verified_factor_artifact(tmp_path, "isolated", 0, "test", source,
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert receipt["committed"] and receipt["verified"]
    assert not receipt["production_published"]
    assert not list(tmp_path.rglob("_catalog.sqlite"))
    manifest_path = __import__("pathlib").Path(receipt["path"])
    manifest = json.loads(manifest_path.read_text())
    reconstructed = pd.concat([
        pq.ParquetFile(manifest_path.parent / chunk["path"]).read().to_pandas()["factor_value"]
        for chunk in manifest["chunks"]
    ])
    pd.testing.assert_series_equal(reconstructed, source.rename("factor_value"), check_exact=True)
    assert np.signbit(reconstructed.iloc[1])
    assert manifest["dq"]["passed"]
    assert manifest["policy_digest"] == DefaultExecutionPolicy().digest
    events = manifest["chunks"][0]["conversions"]
    assert [(e["from"], e["to"]) for e in events] == [
        ("pandas_series", "arrow_table"), ("arrow_table", "pandas_series")]
    assert all(e["logical_arrow_bytes"] > 0 and e["seconds"] >= 0 for e in events)
    assert all(e["copied_bytes"] is None for e in events)  # allocation not measured


def test_budget_refusal_precedes_any_write(tmp_path):
    with pytest.raises(ArtifactResourceRequirementError):
        write_verified_factor_artifact(tmp_path, "isolated", 0, "test", values(),
                                       policy=DefaultExecutionPolicy(), budget_bytes=1)
    assert not list(tmp_path.iterdir())


def test_infinity_fails_existing_dq_not_cleaned_into_success(tmp_path):
    source = values()
    source.iloc[0] = np.inf
    with pytest.raises(Exception, match="DQ"):
        write_verified_factor_artifact(tmp_path, "isolated", 0, "test", source,
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert not list(tmp_path.iterdir())


def test_unknown_unlabelled_result_not_a_factor_artifact(tmp_path):
    with pytest.raises(TypeError):
        write_verified_factor_artifact(tmp_path, "isolated", 0, "test", "debug-plan",
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_external_receipt_independently_verified(tmp_path):
    receipt = write_verified_factor_artifact(tmp_path, "isolated", 0, "test", values(),
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    receipt["verified"] = False
    checked = verify_factor_artifact_receipt(receipt, tmp_path, "isolated", 0, "test", values(),
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert checked["verified"] is True


def test_external_verified_boolean_not_commit_proof(tmp_path):
    with pytest.raises(ValueError, match="isolated values scope"):
        verify_factor_artifact_receipt({"committed": True, "verified": True},
                                       tmp_path, "isolated", 0, "test", values(),
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_external_wrong_scope_rejected(tmp_path):
    receipt = write_verified_factor_artifact(tmp_path, "other_run", 0, "test", values(),
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    with pytest.raises(ValueError, match="isolated values scope"):
        verify_factor_artifact_receipt(receipt, tmp_path, "isolated", 0, "test", values(),
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_external_self_consistent_hashes_do_not_hide_wrong_values(tmp_path):
    other = values()
    other.iloc[2] = 100
    receipt = write_verified_factor_artifact(tmp_path, "isolated", 0, "test", other,
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    with pytest.raises(AssertionError):
        verify_factor_artifact_receipt(receipt, tmp_path, "isolated", 0, "test", values(),
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_external_chunk_path_cannot_escape_generation(tmp_path):
    receipt = write_verified_factor_artifact(tmp_path, "isolated", 0, "test", values(),
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    path = Path(receipt["path"])
    manifest = json.loads(path.read_text())
    manifest["chunks"][0]["path"] = "../outside.parquet"
    payload = json.dumps(manifest).encode()
    path.write_bytes(payload)
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="contiguous output coverage"):
        verify_factor_artifact_receipt(receipt, tmp_path, "isolated", 0, "test", values(),
                                       policy=DefaultExecutionPolicy(), budget_bytes=2**20)


def test_commit_syncs_directory_chain_through_approved_root(tmp_path, monkeypatch):
    import factor_engine.runtime.durable_artifact_sink as sink_module

    original = sink_module._fsync_directory
    synced = []

    def track(path):
        synced.append(Path(path))
        original(path)

    monkeypatch.setattr(sink_module, "_fsync_directory", track)
    receipt = write_verified_factor_artifact(tmp_path, "isolated", 0, "test", values(),
                                             policy=DefaultExecutionPolicy(), budget_bytes=2**20)
    assert receipt["committed"]
    assert synced[-3:] == [tmp_path / "isolated" / "values", tmp_path / "isolated", tmp_path]
