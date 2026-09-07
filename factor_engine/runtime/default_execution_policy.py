"""Single, strict v2 execution policy; not a resource allocator or admission grant."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math


@dataclass(frozen=True)
class DefaultExecutionPolicy:
    policy_id: str = "ashare_auto_durable_v2"
    schema_version: str = "factor_engine.auto80_policy.v1"
    backend: str = "auto"
    result_mode: str = "artifact"
    on_factor_error: str = "report_and_continue"
    on_integrity_error: str = "abort_affected_scope"
    memory_fraction: float = 0.80
    optional_admin_cap_bytes: int | None = None
    sample_interval_seconds: float = 1.0
    grow_after_healthy_samples: int = 5
    max_growth_step_of_target: float = 0.05
    task_peak_uncertainty_multiplier: float = 1.25
    result_queue_fraction: float = 0.05
    result_queue_initial_cap_bytes: int = 268435456
    initial_lookahead_factors: int = 512
    max_manifest_factors: int = 1000000
    max_definition_bytes: int = 131072
    input_ingestion_deadline_seconds: float = 300.0
    beam_width: int = 8
    candidate_limit_per_group: int = 128
    optimization_budget_ms_per_group: float = 250.0
    region_hysteresis_gain_fraction: float = 0.08
    work_item_max_attempts: int = 3
    root_max_recovery_actions: int = 2
    max_transient_retries: int = 2
    max_oom_replans: int = 1
    max_deterministic_retries: int = 0
    max_compute_timeout_retries: int = 0
    retry_backoff_cap_seconds: float = 8.0
    retry_backoff_total_budget_seconds: float = 30.0
    exact_scope_circuit_breaker_failures: int = 5
    circuit_half_open_attempts_per_lineage: int = 1
    resource_wait_seconds: float = 300.0
    compute_unknown_seconds: float = 900.0
    compute_prediction_multiplier: float = 10.0
    compute_prediction_add_seconds: float = 30.0
    compute_min_seconds: float = 300.0
    compute_max_seconds: float = 3600.0
    connect_seconds: float = 5.0
    read_attempt_seconds: float = 60.0
    sink_flush_seconds: float = 300.0
    reconcile_seconds: float = 120.0
    idle_no_inflight_progress_seconds: float = 120.0
    cooperative_cancel_grace_seconds: float = 5.0
    worker_exit_observation_seconds: float = 10.0
    job_unknown_seconds: float = 86400.0
    job_prediction_multiplier: float = 6.0
    job_prediction_add_seconds: float = 3600.0
    job_min_seconds: float = 3600.0
    job_max_seconds: float = 86400.0
    target_file_bytes: int = 134217728
    max_file_bytes: int = 268435456
    max_open_writers: int = 16
    initial_writer_workers: int = 1
    max_writer_workers: int = 4
    max_error_groups_in_response: int = 20
    max_examples_per_error_group: int = 3
    automatic_production_publish: bool = False

    def __post_init__(self):
        enums = {
            "policy_id": {"ashare_auto_durable_v2"},
            "schema_version": {"factor_engine.auto80_policy.v1"},
            "backend": {"auto", "pandas", "pandas_numpy", "polars_long", "duckdb_sql"},
            "result_mode": {"artifact"},
            "on_factor_error": {"report_and_continue"},
            "on_integrity_error": {"abort_affected_scope"},
        }
        zero_allowed = {"max_transient_retries", "max_oom_replans", "max_deterministic_retries",
                        "max_compute_timeout_retries", "root_max_recovery_actions"}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in enums:
                if not isinstance(value, str) or value not in enums[field.name]:
                    raise ValueError(f"invalid {field.name}")
            elif field.name == "automatic_production_publish":
                if value is not False:
                    raise ValueError("v2 artifact execution cannot authorize production publication")
            elif field.name == "optional_admin_cap_bytes" and value is None:
                continue
            elif field.type in ("int", "int | None"):
                if type(value) is not int or value < (0 if field.name in zero_allowed else 1):
                    raise ValueError(f"{field.name} must be a valid integer")
            else:
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field.name} must be finite and positive")
                # Canonical digest: semantically equal numeric inputs have equal identity.
                object.__setattr__(self, field.name, float(value))
        for name in ("memory_fraction", "result_queue_fraction", "max_growth_step_of_target",
                     "region_hysteresis_gain_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        if self.memory_fraction > .85:
            raise ValueError("memory_fraction exceeds the supported broker maximum of 0.85")
        if self.task_peak_uncertainty_multiplier < 1:
            raise ValueError("task_peak_uncertainty_multiplier must be >= 1")
        if self.work_item_max_attempts > 3 or self.root_max_recovery_actions > 2:
            raise ValueError("v2 total attempt/recovery safety bounds exceeded")
        if self.max_transient_retries > 2 or self.max_oom_replans > 1:
            raise ValueError("v2 retry safety bounds exceeded")
        if self.max_deterministic_retries or self.max_compute_timeout_retries:
            raise ValueError("deterministic errors and compute timeouts must not retry")
        for low, high in (("compute_min_seconds", "compute_max_seconds"),
                          ("job_min_seconds", "job_max_seconds"),
                          ("target_file_bytes", "max_file_bytes"),
                          ("initial_writer_workers", "max_writer_workers")):
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"{low} cannot exceed {high}")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_default_policy(request: Mapping | None = None,
                           profile: Mapping | None = None) -> DefaultExecutionPolicy:
    """Explicit request > approved profile > v2 defaults, validating each layer.

    Validation before merging prevents a malformed profile from being hidden by
    an overriding request. No environment variables change this policy implicitly.
    """
    allowed = {f.name for f in fields(DefaultExecutionPolicy)}
    merged = {}
    for label, layer in (("profile", profile), ("request", request)):
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise ValueError(f"{label} policy must be a mapping")
        unknown = set(layer) - allowed
        if unknown:
            raise ValueError(f"unknown {label} policy fields: {sorted(unknown, key=str)}")
        DefaultExecutionPolicy(**layer)
        merged.update(layer)
    return DefaultExecutionPolicy(**merged)
