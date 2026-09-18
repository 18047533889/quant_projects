import hashlib
import json
from pathlib import Path

import campaign
from campaign import campaign_protocol_hash, execution_status, successful_fingerprints


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _success(backend, fingerprint):
    return {"canonical": "op", "backend": backend, "status": "EXECUTED_FINITE",
            "fingerprint": fingerprint, "future_prefix_invariant_all_inputs": True}


def test_prefix_failure_is_not_success():
    status, _ = execution_status(10, False, None)
    assert status == "FAILED_EXECUTION_PENDING_TRIAGE"


def test_protocol_hash_binds_campaign_source_bytes():
    assert campaign_protocol_hash() == hashlib.sha256(Path(campaign.__file__).read_bytes()).hexdigest()


def test_cache_requires_bound_successful_pair(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [_success("pandas_numpy", "p1"), _success("polars", "q1"),
            {"canonical": "op", "status": "CANONICAL_PARITY",
             "pandas_polars_equal": True}]
    _write(ledger, rows)
    assert successful_fingerprints(ledger) == {}

    rows[-1]["backend_fingerprints"] = {"pandas_numpy": "p1", "polars": "q1"}
    _write(ledger, rows)
    assert successful_fingerprints(ledger) == {
        ("op", "pandas_numpy"): "p1", ("op", "polars"): "q1"
    }


def test_one_changed_backend_invalidates_pair_cache(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [_success("pandas_numpy", "p1"), _success("polars", "q1"),
            {"canonical": "op", "status": "CANONICAL_PARITY",
             "pandas_polars_equal": True,
             "backend_fingerprints": {"pandas_numpy": "p1", "polars": "q1"}},
            _success("polars", "q2")]
    _write(ledger, rows)
    assert successful_fingerprints(ledger) == {}


def test_failed_parity_is_not_cached(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [_success("pandas_numpy", "p1"), _success("polars", "q1"),
            {"canonical": "op", "status": "CANONICAL_PARITY_FAILED",
             "pandas_polars_equal": False,
             "backend_fingerprints": {"pandas_numpy": "p1", "polars": "q1"}}]
    _write(ledger, rows)
    assert successful_fingerprints(ledger) == {}


def test_success_status_without_explicit_prefix_proof_is_not_cached(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    left = _success("pandas_numpy", "p1")
    right = _success("polars", "q1")
    del right["future_prefix_invariant_all_inputs"]
    rows = [left, right,
            {"canonical": "op", "status": "CANONICAL_PARITY",
             "pandas_polars_equal": True,
             "backend_fingerprints": {"pandas_numpy": "p1", "polars": "q1"}}]
    _write(ledger, rows)
    assert successful_fingerprints(ledger) == {}
