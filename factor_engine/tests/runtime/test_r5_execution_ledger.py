from __future__ import annotations

import json

import pytest

from factor_engine.runtime.execution_ledger import (
    ExecutionLedgerError,
    MAX_LEDGER_BYTES,
    summarize_execution_ledger,
)


class _Secret:
    def __repr__(self):
        raise AssertionError("unknown objects must not be represented")


def test_extracts_only_bounded_planned_and_actual_scalar_evidence():
    output = {
        "executor": "adaptive_batch_scheduler",
        "results": {"secret_factor": _Secret()},
        "dag": _Secret(),
        "analyses": _Secret(),
        "scheduler_stats": {
            "task_timing": {"secret_factor": [1, 2, 3]},
            "run_peak": {"peak_family_rss": 1234, "peak_family_pss": 1000},
            "scheduler_stats": {
                "real_task_done": 4,
                "future_count": 2,
                "factor_count": 3,
                "resource_decision": {
                    "cse_cache_bytes": 4096,
                    "read_wave_bytes": 8192,
                    "unknown": _Secret(),
                },
            },
        },
        "physical_plan": {
            "plan_id": "plan-1",
            "planned_backend": "polars_long",
            "estimated_peak_memory_bytes": 999,
            "memory_basis": "estimated-not-measured",
            "unknown": _Secret(),
        },
        "backend_paths": {
            "secret_factor": {
                "physical_plan": {
                    "planned_backend": "polars_long",
                    "actual_backend": "duckdb_sql",
                    "transfers": [{
                        "edge_id": "never-persisted",
                        "source_backend": "polars_long",
                        "target_backend": "duckdb_sql",
                        "source_representation": "polars_long",
                        "target_representation": "arrow_table",
                        "actual_bytes": 100,
                        "actual_rows": 5,
                        "actual_transfer_ms": 1.25,
                        "sort_applied": True,
                    }],
                }
            },
            "another_secret": {
                "physical_plan": {
                    "planned_backend": "polars_long",
                    "actual_backend": "duckdb_sql",
                    "transfers": [{
                        "source_backend": "polars_long",
                        "target_backend": "duckdb_sql",
                        "source_representation": "polars_long",
                        "target_representation": "arrow_table",
                        "actual_bytes": 50,
                        "actual_rows": 2,
                        "actual_transfer_ms": 0.75,
                        "reshape_applied": True,
                    }],
                }
            },
        },
    }
    ledger = summarize_execution_ledger(output, run_id="run", evidence_id="ev")
    assert ledger["planned_physical"]["estimated_peak_memory_bytes"] == 999
    assert ledger["actual_scheduler"]["run_peak"]["peak_family_rss"] == 1234
    assert ledger["actual_scheduler"]["resource_decision"]["cse_cache_bytes"] == 4096
    actual = ledger["actual_physical"]
    assert actual["route_counts"] == [{
        "planned_backend": "polars_long", "actual_backend": "duckdb_sql",
        "root_count": 2,
    }]
    transfer = actual["transfer_groups"][0]
    assert transfer["event_count"] == 2
    assert transfer["actual_bytes"] == 150
    assert transfer["actual_rows"] == 7
    assert transfer["actual_transfer_ms"] == 2.0
    assert transfer["sort_applied_count"] == 1
    assert transfer["reshape_applied_count"] == 1
    encoded = json.dumps(
        ledger, indent=2, sort_keys=True, allow_nan=False
    ).encode()
    assert len(encoded) <= MAX_LEDGER_BYTES
    text = encoded.decode()
    assert "secret_factor" not in text
    assert "another_secret" not in text
    assert "edge_id" not in text
    assert "task_timing" not in text


def test_missing_evidence_is_omitted_not_fabricated_as_zero():
    ledger = summarize_execution_ledger({}, run_id="run", evidence_id="ev")
    assert "executor" not in ledger
    assert "planned_physical" not in ledger
    assert "actual_scheduler" not in ledger
    assert "actual_physical" not in ledger


def test_transfer_events_alias_and_nonfinite_or_unknown_values_are_omitted():
    output = {"backend_paths": {"private": {"physical_plan": {
        "planned_backend": "pandas_numpy",
        "actual_backend": "pandas_numpy",
        "transfer_events": [{
            "source_backend": "pandas_numpy",
            "target_backend": "pandas_numpy",
            "source_representation": "pandas_long",
            "target_representation": "pandas_long",
            "actual_transfer_ms": float("nan"),
            "actual_bytes": _Secret(),
        }],
    }}}}
    ledger = summarize_execution_ledger(output, run_id="run", evidence_id="ev")
    transfer = ledger["actual_physical"]["transfer_groups"][0]
    assert "actual_transfer_ms" not in transfer
    assert "actual_bytes" not in transfer
    json.dumps(ledger, allow_nan=False)


def test_many_roots_aggregate_without_names_or_unbounded_output():
    shared = {"physical_plan": {
        "planned_backend": "polars_long", "actual_backend": "polars_long"
    }}
    output = {"backend_paths": {f"private-{i}": shared for i in range(100_000)}}
    ledger = summarize_execution_ledger(output, run_id="run", evidence_id="ev")
    assert ledger["actual_physical"]["root_paths_observed"] == 100_000
    assert ledger["actual_physical"]["route_counts"][0]["root_count"] == 100_000
    encoded = json.dumps(
        ledger, indent=2, sort_keys=True, allow_nan=False
    ).encode()
    assert len(encoded) < MAX_LEDGER_BYTES
    assert b"private-" not in encoded


def test_size_gate_matches_indented_durable_receipt_serialization():
    semantics = {
        "planned": "planner estimates and selected routes",
        "actual": "runtime counters and measured transfer/run-peak telemetry",
    }
    selected = None
    for count in range(1, 2000):
        candidate = {
            "schema_version": "factor_engine.execution_ledger.v1",
            "run_id": "run",
            "evidence_id": "ev",
            "evidence_semantics": semantics,
            "actual_physical": {
                "root_paths_observed": count,
                "route_counts": [
                    {
                        "planned_backend": f"planned-{i}",
                        "actual_backend": f"actual-{i}",
                        "root_count": 1,
                    }
                    for i in range(count)
                ],
            },
        }
        compact = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        durable = json.dumps(
            candidate, indent=2, sort_keys=True, allow_nan=False
        ).encode()
        if len(compact) <= MAX_LEDGER_BYTES < len(durable):
            selected = count
            break
    assert selected is not None
    output = {
        "backend_paths": {
            f"private-{i}": {
                "physical_plan": {
                    "planned_backend": f"planned-{i}",
                    "actual_backend": f"actual-{i}",
                }
            }
            for i in range(selected)
        }
    }
    with pytest.raises(ExecutionLedgerError, match="64 KiB"):
        summarize_execution_ledger(output, run_id="run", evidence_id="ev")
