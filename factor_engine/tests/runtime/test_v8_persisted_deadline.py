from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from factor_engine.runtime.run_deadline import (
    RunDeadlineError,
    create_run_deadline_record,
    restore_run_deadline,
)


RUN_ID = "a" * 32
POLICY_DIGEST = "b" * 64
BOOT_ID = "12345678-1234-1234-1234-123456789abc"


def _record(**overrides):
    values = {
        "run_id": RUN_ID,
        "policy_digest": POLICY_DIGEST,
        "job_unknown_seconds": 30,
        "job_max_seconds": 20,
        "monotonic_ns": 10_000_000_000,
        "realtime_ns": 100_000_000_000,
        "boot_id": BOOT_ID,
    }
    values.update(overrides)
    return create_run_deadline_record(**values)


def _restore(record, **overrides):
    values = {
        "expected_run_id": RUN_ID,
        "expected_policy_digest": POLICY_DIGEST,
        "monotonic_ns": 15_000_000_000,
        "realtime_ns": 105_000_000_000,
        "boot_id": BOOT_ID,
    }
    values.update(overrides)
    return restore_run_deadline(record, **values)


def _create_real_record(connection):
    connection.send(create_run_deadline_record(
        run_id=RUN_ID,
        policy_digest=POLICY_DIGEST,
        job_unknown_seconds=2,
        job_max_seconds=1,
    ))
    connection.close()


def test_duration_uses_original_policy_minimum_and_same_boot_never_refreshes():
    record = _record()
    assert record["schema_version"] == "factor_engine.run_deadline.v1"
    assert record["duration_ns"] == 20_000_000_000

    first = _restore(record)
    later = _restore(
        record, monotonic_ns=19_000_000_000, realtime_ns=109_000_000_000
    )
    expired = _restore(
        record, monotonic_ns=31_000_000_000, realtime_ns=121_000_000_000
    )

    assert first.deadline_monotonic_ns == later.deadline_monotonic_ns == 30_000_000_000
    assert first.remaining_ns == 15_000_000_000
    assert later.remaining_ns == 11_000_000_000
    assert expired.remaining_ns == 0 and expired.expired is True


def test_real_cross_process_restore_keeps_absolute_deadline_and_spends_time():
    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(target=_create_real_record, args=(child,))
    process.start()
    child.close()
    record = parent.recv()
    process.join(2)
    assert process.exitcode == 0

    first = restore_run_deadline(
        record, expected_run_id=RUN_ID, expected_policy_digest=POLICY_DIGEST
    )
    time.sleep(0.03)
    second = restore_run_deadline(
        record, expected_run_id=RUN_ID, expected_policy_digest=POLICY_DIGEST
    )

    assert first.deadline_monotonic_ns == second.deadline_monotonic_ns
    assert 0 <= second.remaining_ns < first.remaining_ns < record["duration_ns"]


@pytest.mark.parametrize("key", sorted({
    "schema_version", "run_id", "policy_digest", "boot_id",
    "start_monotonic_ns", "start_realtime_ns", "duration_ns",
}))
def test_missing_fields_are_rejected(key):
    record = _record()
    del record[key]
    with pytest.raises(RunDeadlineError, match="exact fields"):
        _restore(record)


def test_extra_field_and_unknown_schema_are_rejected():
    record = _record()
    with pytest.raises(RunDeadlineError, match="exact fields"):
        _restore({**record, "deadline_monotonic_ns": 30_000_000_000})
    with pytest.raises(RunDeadlineError, match="unsupported schema"):
        _restore({**record, "schema_version": "factor_engine.run_deadline.v2"})


@pytest.mark.parametrize("value", [None, True, 1, 1.0, ["factor_engine.run_deadline.v1"]])
def test_schema_version_requires_the_exact_string(value):
    with pytest.raises(RunDeadlineError, match="unsupported schema"):
        _restore({**_record(), "schema_version": value})


@pytest.mark.parametrize(
    "field",
    ["boot_id", "start_monotonic_ns", "start_realtime_ns", "duration_ns"],
)
def test_persisted_null_never_uses_runtime_defaults(field):
    record = _record()
    record[field] = None
    with pytest.raises(RunDeadlineError):
        _restore(record)


@pytest.mark.parametrize("field", ["start_monotonic_ns", "start_realtime_ns", "duration_ns"])
@pytest.mark.parametrize("value", [True, 1.0, -1])
def test_record_integer_fields_are_strict(field, value):
    record = _record()
    record[field] = value
    with pytest.raises(RunDeadlineError):
        _restore(record)


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["job_unknown_seconds", "job_max_seconds"])
def test_invalid_policy_durations_are_rejected(name, value):
    with pytest.raises(RunDeadlineError):
        _record(**{name: value})


def test_sub_nanosecond_duration_is_rejected():
    with pytest.raises(RunDeadlineError, match="below one nanosecond"):
        _record(job_unknown_seconds=1e-12)


def test_unrepresentable_duration_is_a_typed_deadline_error():
    for value in (1e308, 10**1000):
        with pytest.raises(RunDeadlineError, match="safe nanosecond range"):
            _record(job_unknown_seconds=value, job_max_seconds=value)


def test_unrepresentable_absolute_deadline_is_rejected():
    with pytest.raises(RunDeadlineError, match="deadline exceeds"):
        _record(monotonic_ns=(1 << 63) - 2, job_unknown_seconds=1)

    record = _record()
    record["start_monotonic_ns"] = (1 << 63) - 2
    with pytest.raises(RunDeadlineError, match="record deadline exceeds"):
        _restore(record, monotonic_ns=(1 << 63) - 1)


@pytest.mark.parametrize("expected", [
    {"expected_run_id": "c" * 32},
    {"expected_policy_digest": "d" * 64},
])
def test_expected_identity_replacement_is_rejected(expected):
    with pytest.raises(RunDeadlineError, match="identity differs"):
        _restore(_record(), **expected)


@pytest.mark.parametrize("mutation", [
    {"run_id": "c" * 32},
    {"policy_digest": "d" * 64},
])
def test_record_identity_replacement_is_rejected(mutation):
    with pytest.raises(RunDeadlineError, match="identity differs"):
        _restore({**_record(), **mutation})


def test_boot_change_and_clock_rollback_fail_closed():
    record = _record()
    with pytest.raises(RunDeadlineError, match="boot change"):
        _restore(record, boot_id="87654321-4321-4321-4321-cba987654321")
    with pytest.raises(RunDeadlineError, match="monotonic clock moved backwards"):
        _restore(record, monotonic_ns=9_999_999_999)
    with pytest.raises(RunDeadlineError, match="realtime clock moved backwards"):
        _restore(record, realtime_ns=99_999_999_999)


@pytest.mark.parametrize("name", ["monotonic_ns", "realtime_ns"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_creation_and_restore_clock_inputs_are_strict(name, value):
    with pytest.raises(RunDeadlineError):
        _record(**{name: value})
    with pytest.raises(RunDeadlineError):
        _restore(_record(), **{name: value})
