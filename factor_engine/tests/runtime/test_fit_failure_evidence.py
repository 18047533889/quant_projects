from __future__ import annotations

import copy
from datetime import date, datetime, timezone
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
from factor_engine.runtime import fit_failure_evidence as evidence


def _failure(reason="forced", details=()):
    return rc.FitResult(None, rc.FitStatus(False, reason, details))


def _scope(instrument="AAA"):
    return rc.FitScope(
        canonical="ts_test", backend="pandas_numpy", profile="research",
        instrument=instrument, window_start=date(2026, 1, 1),
        window_end=date(2026, 1, 2), output_row=date(2026, 1, 3),
        fit_cutoff=date(2026, 1, 2), maturity_cutoff=date(2026, 1, 2),
        execution_id="exec", run_id="run", task_id="task", factor_id="factor",
    )


def test_none_and_observed_empty_are_distinct():
    assert evidence.snapshot_fit_failures(None) is None
    snapshot = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    assert snapshot["availability"] == "observed"
    assert snapshot["coverage"] == "instrumented_fit_producers_only"
    assert snapshot["groups"] == [] and snapshot["details"] == []
    assert evidence.validate_fit_failure_snapshot(snapshot) == snapshot


def test_tagged_values_are_json_safe_and_canonical():
    sink = rc.BoundedFitFailureSink()
    status = _failure(details=(("blob", b"\x00\xff"), ("number", 2 + 3j)))
    sink.record(status, _scope())
    snapshot = evidence.snapshot_fit_failures(sink)
    encoded = evidence.encode_fit_failure_snapshot(snapshot)
    assert encoded == evidence.encode_fit_failure_snapshot(snapshot)
    assert json.loads(encoded) == snapshot
    text = encoded.decode()
    assert '"$type":"bytes"' in text
    assert '"$type":"complex"' in text
    assert '"$type":"date"' in text

    tagged_numpy = evidence._tag(np.int64(7))
    tagged_datetime = evidence._tag(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert tagged_numpy == {"$type": "numpy_scalar", "dtype": "int64", "value": 7}
    assert tagged_datetime["$type"] == "datetime"


def test_real_sink_nonfinite_diagnostics_are_explicitly_tagged():
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(details=(("nan", np.nan), ("pos", np.inf), ("neg", -np.inf),
                                  ("z", complex(np.inf, np.nan)))), _scope())
    snapshot = evidence.snapshot_fit_failures(sink)
    encoded = evidence.encode_fit_failure_snapshot(snapshot)
    assert b'"value":"nan"' in encoded
    assert b'"value":"+inf"' in encoded
    assert b'"value":"-inf"' in encoded
    assert b'"$type":"complex"' in encoded


def test_snapshot_is_detached_from_later_thread_writes():
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(), _scope())
    snapshot = evidence.snapshot_fit_failures(sink)

    def write_many():
        for _ in range(50):
            sink.record(_failure(), _scope())

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(write_many).result()
    assert snapshot["groups"][0]["count"] == 1
    assert len(snapshot["details"]) == 1
    assert evidence.validate_fit_failure_snapshot(snapshot) == snapshot


def test_sink_lock_is_not_pickleable_but_snapshot_is():
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(), _scope())
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(sink)
    snapshot = evidence.snapshot_fit_failures(sink)
    assert pickle.loads(pickle.dumps(snapshot)) == snapshot


def test_snapshot_truncates_old_details_to_hard_byte_bound():
    sink = rc.BoundedFitFailureSink(detail_capacity=64, group_capacity=128)
    details = tuple((f"key-{i}", "x" * 256) for i in range(64))
    for i in range(64):
        sink.record(_failure(details=details), _scope(f"instrument-{i}"))
    snapshot = evidence.snapshot_fit_failures(sink)
    assert snapshot["truncated"] is True
    assert "snapshot_bytes_details" in snapshot["truncation_reasons"]
    assert snapshot["dropped_details"] > 0
    sequences = [detail["sequence"] for detail in snapshot["details"]]
    assert sequences == sorted(sequences)
    assert sequences[-1] == 64
    assert len(evidence.encode_fit_failure_snapshot(snapshot)) <= evidence.MAX_SNAPSHOT_BYTES


def test_snapshot_truncates_sorted_group_tail_into_overflow_count():
    sink = rc.BoundedFitFailureSink(detail_capacity=0, group_capacity=200)
    for index in range(130):
        sink.record(_failure(f"reason-{index:03d}"), _scope())
    snapshot = evidence.snapshot_fit_failures(sink)
    assert len(snapshot["groups"]) == evidence.MAX_GROUPS
    assert snapshot["overflow_group_count"] == 2
    assert snapshot["truncation_reasons"] == ["sink_detail_eviction", "group_count_limit"]
    assert [group["reason"] for group in snapshot["groups"]] == sorted(
        group["reason"] for group in snapshot["groups"]
    )


def test_validation_rejects_schema_extra_fields_and_bad_scalars():
    valid = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    cases = []
    wrong_schema = copy.deepcopy(valid)
    wrong_schema["schema_version"] = "wrong"
    cases.append(wrong_schema)
    extra = copy.deepcopy(valid)
    extra["extra"] = 1
    cases.append(extra)
    bad_nan = copy.deepcopy(valid)
    bad_nan["unknown_status_count"] = float("nan")
    cases.append(bad_nan)
    huge_int = copy.deepcopy(valid)
    huge_int["dropped_details"] = 1 << 1000
    cases.append(huge_int)
    for payload in cases:
        with pytest.raises(ValueError):
            evidence.validate_fit_failure_snapshot(payload)


def test_validation_checks_structure_before_encoding_and_rejects_oversize():
    valid = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    cyclic = copy.deepcopy(valid)
    cyclic["groups"].append(cyclic)
    with pytest.raises(ValueError):
        evidence.validate_fit_failure_snapshot(cyclic)

    oversized = copy.deepcopy(valid)
    oversized["groups"] = [
        {"canonical": f"c{i}-" + "x" * 4000, "reason": "r" + "y" * 4000, "count": 1}
        for i in range(128)
    ]
    with pytest.raises(ValueError, match="byte bound"):
        evidence.validate_fit_failure_snapshot(oversized)


def test_arbitrary_objects_and_invalid_tag_shapes_fail_closed():
    class Secret:
        def __repr__(self):
            raise AssertionError("repr must not be called")

    with pytest.raises(ValueError, match="unsupported"):
        evidence._tag(Secret())
    valid = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    valid["details"] = [{
        "sequence": 1,
        "status": {"converged": False, "reason": "forced",
                   "details": {"$type": "bytes", "base64": "!"}},
        "scope": {name: None for name in evidence._SCOPE_FIELDS},
        "scope_kind": "kernel_only",
    }]
    with pytest.raises(ValueError, match="tagged bytes"):
        evidence.validate_fit_failure_snapshot(valid)


def test_owned_scope_and_tag_component_shapes_are_exact():
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(), _scope())
    valid = evidence.snapshot_fit_failures(sink)
    forged = copy.deepcopy(valid)
    forged["details"][0]["scope"] = {name: None for name in evidence._SCOPE_FIELDS}
    with pytest.raises(ValueError, match="scope kind"):
        evidence.validate_fit_failure_snapshot(forged)

    bad_tuple = copy.deepcopy(valid)
    bad_tuple["details"][0]["status"]["details"] = {"$type": "tuple", "items": {}}
    with pytest.raises(ValueError, match="sequence items"):
        evidence.validate_fit_failure_snapshot(bad_tuple)
    bad_complex = copy.deepcopy(valid)
    bad_complex["details"][0]["status"]["details"] = {
        "$type": "complex", "real": "1", "imag": 0.0,
    }
    with pytest.raises(ValueError, match="complex component"):
        evidence.validate_fit_failure_snapshot(bad_complex)
    bad_numpy = copy.deepcopy(valid)
    bad_numpy["details"][0]["status"]["details"] = {
        "$type": "numpy_scalar", "dtype": "object", "value": 1,
    }
    with pytest.raises(ValueError, match="numpy scalar dtype"):
        evidence.validate_fit_failure_snapshot(bad_numpy)


def test_factor_window_validation_preserves_nanoseconds():
    sink = rc.BoundedFitFailureSink()
    instant = pd.Timestamp("2026-01-03T00:00:00.000000001")
    scope = rc.FitScope(
        canonical="ts_test", backend="pandas_numpy", profile="research",
        instrument="AAA", window_start=pd.Timestamp("2026-01-01"),
        window_end=instant, output_row=instant, fit_cutoff=instant,
        maturity_cutoff=instant, execution_id="exec", run_id="run",
        task_id="task", factor_id="factor",
    )
    sink.record(_failure(), scope)
    payload = evidence.snapshot_fit_failures(sink)
    payload["details"][0]["scope"]["window_end"] = {
        "$type": "datetime", "iso8601": "2026-01-03T00:00:00.000000002",
    }
    with pytest.raises(ValueError, match="scope kind"):
        evidence.validate_fit_failure_snapshot(payload)


@pytest.mark.parametrize("depth", range(9))
def test_nested_core_diagnostics_never_prevent_summary_snapshot(depth):
    nested = "leaf"
    for _ in range(depth):
        nested = (nested,)
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(details=(("nested", nested),)), _scope())
    payload = evidence.snapshot_fit_failures(sink)
    assert payload["groups"] == [
        {"canonical": "ts_test", "reason": "forced", "count": 1},
    ]
    assert evidence.validate_fit_failure_snapshot(payload) == payload
    if not payload["details"]:
        assert "snapshot_structure_details" in payload["truncation_reasons"]


def test_mixed_numpy_datetime_units_and_timestamp_round_trip_as_owned_scope():
    sink = rc.BoundedFitFailureSink()
    scope = rc.FitScope(
        canonical="ts_test", backend="pandas_numpy", profile="research",
        instrument="AAA", window_start=np.datetime64("2026-01-01", "ms"),
        window_end=np.datetime64("2026-01-02T00:00:00.000000001", "ns"),
        output_row=pd.Timestamp("2026-01-03T00:00:00.000000002"),
        fit_cutoff=np.datetime64("2026-01-02T00:00:00.000", "ms"),
        maturity_cutoff=np.datetime64("2026-01-03T00:00:00.000000001", "ns"),
        execution_id="exec", run_id="run", task_id="task", factor_id="factor",
    )
    assert scope.scope_kind == "factor_window"
    sink.record(_failure(), scope)
    payload = evidence.snapshot_fit_failures(sink)
    assert payload["details"][0]["scope_kind"] == "factor_window"
    assert evidence.validate_fit_failure_snapshot(payload) == payload


def test_validation_preflights_encoded_size_without_full_canonical_dump(monkeypatch):
    payload = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    payload["groups"] = [
        {"canonical": f"c-{index}-" + "界" * 4000,
         "reason": "\u0001" * 4096, "count": 1}
        for index in range(evidence.MAX_GROUPS)
    ]
    monkeypatch.setattr(
        evidence, "_canonical_bytes",
        lambda _payload: (_ for _ in ()).throw(AssertionError("full dump called")),
    )
    with pytest.raises(ValueError, match="byte bound"):
        evidence.validate_fit_failure_snapshot(payload)


def test_bounded_encoder_preserves_legal_unicode_and_escaped_strings():
    payload = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    payload["groups"] = [{
        "canonical": "因子-🧪", "reason": "\u0001" * 4096, "count": 1,
    }]
    validated = evidence.validate_fit_failure_snapshot(payload)
    encoded = evidence.encode_fit_failure_snapshot(validated)
    assert json.loads(encoded) == payload
    assert len(encoded) <= evidence.MAX_SNAPSHOT_BYTES


def test_validation_rejects_duplicate_groups():
    payload = evidence.snapshot_fit_failures(rc.BoundedFitFailureSink())
    group = {"canonical": "ts_test", "reason": "forced", "count": 1}
    payload["groups"] = [group, copy.deepcopy(group)]
    with pytest.raises(ValueError, match="duplicate failure group"):
        evidence.validate_fit_failure_snapshot(payload)


def test_validation_requires_strictly_increasing_detail_sequences():
    sink = rc.BoundedFitFailureSink()
    sink.record(_failure(), _scope("AAA"))
    sink.record(_failure(), _scope("BBB"))
    payload = evidence.snapshot_fit_failures(sink)
    payload["details"][1]["sequence"] = payload["details"][0]["sequence"]
    with pytest.raises(ValueError, match="strictly increasing"):
        evidence.validate_fit_failure_snapshot(payload)
