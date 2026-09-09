"""Independent wire checks on legitimate producer diagnostics and eviction."""
import copy
import json

import pytest
import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
from factor_engine.runtime.fit_failure_evidence import (
    MAX_SNAPSHOT_BYTES,
    encode_fit_failure_snapshot,
    snapshot_fit_failures,
    validate_fit_failure_snapshot,
)


def test_nonfinite_numerical_failure_details_remain_observable():
    sink = rc.BoundedFitFailureSink()
    status = rc.FitStatus(False, "numerical_failure", (
        ("scale", float("inf")), ("residual", float("nan")),
        ("complex_diagnostic", complex(float("-inf"), 1)),
    ))
    sink.record(rc.FitResult(None, status))
    snapshot = snapshot_fit_failures(sink)
    assert snapshot["groups"] == [
        {"canonical": "<kernel_only>", "reason": "numerical_failure", "count": 1}
    ]
    assert len(snapshot["details"]) == 1
    wire = encode_fit_failure_snapshot(snapshot)
    json.loads(wire, parse_constant=lambda value: pytest.fail(f"bare JSON constant {value}"))


def test_preexisting_sink_eviction_and_group_overflow_are_explicit():
    sink = rc.BoundedFitFailureSink(detail_capacity=1, group_capacity=1)
    sink.record(rc.FitResult(None, rc.FitStatus(False, "singular")))
    sink.record(rc.FitResult(None, rc.FitStatus(False, "insufficient_sample")))
    snapshot = snapshot_fit_failures(sink)
    assert snapshot["dropped_details"] == 1
    assert snapshot["overflow_group_count"] == 1
    assert snapshot["truncated"] is True
    assert snapshot["truncation_reasons"]


def test_high_detail_load_is_truncated_without_losing_summary():
    sink = rc.BoundedFitFailureSink(detail_capacity=96)
    # Each individual FitStatus is valid under the existing core contract.
    for number in range(96):
        detail = tuple((str(i), "x" * 256) for i in range(64))
        sink.record(rc.FitResult(None, rc.FitStatus(False, "non_converged", detail)))
    snapshot = snapshot_fit_failures(sink)
    assert snapshot["groups"][0]["count"] == 96
    assert len(snapshot["details"]) <= 64
    assert snapshot["dropped_details"] == 96 - len(snapshot["details"])
    assert snapshot["truncated"]
    assert len(encode_fit_failure_snapshot(snapshot)) <= MAX_SNAPSHOT_BYTES


def test_unowned_receipt_cannot_be_promoted_by_changing_wire_label():
    sink = rc.BoundedFitFailureSink()
    sink.record(rc.FitResult(None, rc.FitStatus(False, "singular")))
    invalid = copy.deepcopy(snapshot_fit_failures(sink))
    invalid["details"][0]["scope_kind"] = "factor_window"
    with pytest.raises((TypeError, ValueError)):
        validate_fit_failure_snapshot(invalid)


@pytest.mark.parametrize("depth", range(9))
def test_valid_core_nested_diagnostics_never_abort_snapshot(depth):
    nested = 1
    for _ in range(depth):
        nested = (nested,)
    sink = rc.BoundedFitFailureSink()
    sink.record(rc.FitResult(None, rc.FitStatus(False, "singular", (("nested", nested),))))
    snapshot = snapshot_fit_failures(sink)
    assert snapshot["groups"][0]["count"] == 1
    assert len(snapshot["details"]) + snapshot["dropped_details"] == 1
    assert len(encode_fit_failure_snapshot(snapshot)) <= MAX_SNAPSHOT_BYTES


def test_mixed_numpy_datetime_units_preserve_real_owned_scope():
    start = np.datetime64("2026-01-01T00:00:00", "ms")
    end = np.datetime64("2026-01-02T00:00:00.000000001", "ns")
    row = pd.Timestamp("2026-01-02T00:00:00.000000002")
    scope = rc.FitScope(
        canonical="op", backend="pandas_numpy", profile="profile", instrument="A",
        window_start=start, window_end=end, output_row=row,
        fit_cutoff=end, maturity_cutoff=end,
        execution_id="exec", run_id="run", task_id="task", factor_id="factor",
    )
    assert scope.scope_kind == "factor_window"
    sink = rc.BoundedFitFailureSink()
    sink.record(rc.FitResult(None, rc.FitStatus(False, "singular")), scope)
    snapshot = snapshot_fit_failures(sink)
    assert len(snapshot["details"]) == 1
    assert snapshot["details"][0]["scope_kind"] == "factor_window"
