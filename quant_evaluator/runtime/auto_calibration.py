"""Explicit exact-batch measured auto selection for the public facade."""
from dataclasses import dataclass, field, replace

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy, calibration_identity,
    evaluate_calibrated_batch,
)


@dataclass(frozen=True)
class AutoCalibrationOptions:
    """Reuse this object in one process to retain bounded exact-request routes.

    Cold misses run both backends; this optimizes subsequent measured route
    selection, not first-call latency. Advanced evaluation inputs are rejected
    by the facade rather than silently discarded from calibration requests.
    """

    policy: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    cache: BoundedCalibrationCache = field(default_factory=BoundedCalibrationCache)

    def __post_init__(self):
        if not isinstance(self.policy, CalibrationPolicy):
            raise InvalidContractError("auto calibration policy must be CalibrationPolicy")
        if not isinstance(self.cache, BoundedCalibrationCache):
            raise InvalidContractError("auto calibration cache must be BoundedCalibrationCache")


def evaluate_measured_auto(batch, label, *, metrics, options, gpu_policy,
                           requested_backend):
    """Attach audited route metadata without mutating the underlying bundle."""
    if not isinstance(options, AutoCalibrationOptions):
        raise InvalidContractError("auto_calibration must be AutoCalibrationOptions")
    if requested_backend not in (None, "auto"):
        raise InvalidContractError("auto_calibration requires backend=None or 'auto'")
    from quant_evaluator.runtime.measured_auto_registry import register_candidate
    if isinstance(metrics, (str, bytes)):
        raise InvalidContractError("metrics must be a sequence")
    metrics = tuple(metrics)
    effective_gpu_policy = gpu_policy or GPUExecutionPolicy()
    result = evaluate_calibrated_batch(
        batch, label, metrics=metrics, calibration_policy=options.policy,
        cache=options.cache, gpu_policy=effective_gpu_policy,
    )
    descriptor = measured_auto_descriptor(batch, label, metrics)
    register_candidate(
        options.cache, result.metadata.get("cache_key", ""),
        status=result.metadata.get("status", ""),
        winner=result.metadata.get("winner", ""),
        calibration_record=result.metadata.get("calibration_record", {}),
        source_check=result.metadata.get("source_check", {}),
        process_source_drifted=result.metadata.get("process_source_drifted", False),
        calibration_policy=options.policy, gpu_policy=effective_gpu_policy,
        descriptor=descriptor, setup_seconds=result.metadata.get("setup_seconds", 0),
        batch=batch, label=label, metrics=metrics,
        request_digest=result.metadata.get("request_digest"),
    )
    bundle = result.bundle
    metadata = dict(bundle.metadata)
    route = {
        "backend_requested": "default" if requested_backend is None else "auto",
        "backend_strategy": "calibrated_auto",
        "backend_used": metadata["backend_used"],
        "auto_backend_policy": "exact_batch_calibration_v1",
        "auto_backend_profile": "exact_content_process_local",
        "auto_backend_reason": result.metadata["status"],
        "metric_backends": metadata["metric_backends"],
    }
    metadata.update(route)
    metadata["auto_calibration"] = result.metadata
    metadata["execution_receipt"] = {
        **route,
        "config_hash": bundle.config_hash,
        "receipt_hash": stable_content_hex(
            tag="EvaluationExecutionReceipt.v1",
            fields={"config_hash": bundle.config_hash, **route},
        ),
    }
    return replace(bundle, metadata=metadata)


def measured_auto_descriptor(batch, label, metrics):
    """Build the no-array-scan hint used before exact candidate validation."""
    from quant_evaluator.runtime.measured_auto_registry import InputDescriptor

    label_axis = None if label.asset_axis is None else (
        label.asset_axis.name, label.asset_axis.dtype, label.asset_axis.size,
    )
    return InputDescriptor.from_values(
        factor_ids=batch.factor_ids, shape=batch.values.shape,
        value_hash=batch.value_hash, label_content_hash=label.content_hash,
        dtype=batch.dtype,
        axis_schema=(
            (batch.time_axis.name, batch.time_axis.dtype, batch.time_axis.size),
            (batch.asset_axis.name, batch.asset_axis.dtype, batch.asset_axis.size),
            label_axis,
        ),
        metrics=tuple(metrics),
    )


def select_measured_auto_backend(batch, label, *, metrics, gpu_policy,
                                 static_backend, static_reason):
    """Adopt only if validation cost is below the measured median margin.

    This post-validation gate cannot recover time already spent checking identity
    and does not promise that the next complete request will be faster.
    """
    import time
    from quant_evaluator.runtime.measured_auto_registry import lookup_candidate_details

    effective_gpu_policy = gpu_policy or GPUExecutionPolicy()
    descriptor = measured_auto_descriptor(batch, label, metrics)

    def validate(candidate):
        started = time.monotonic()
        try:
            candidate_policy = CalibrationPolicy(**dict(candidate.calibration_policy))
            from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
            from quant_evaluator.contracts.label_bundle import LabelBundle
            exact_axes = (
                type(batch.time_axis) is AxisRef
                and type(batch.asset_axis) is AxisRef
                and (label.asset_axis is None or type(label.asset_axis) is AxisRef)
            )
            exact_objects = (
                type(batch) is FactorBatch
                and type(label) is LabelBundle
                and exact_axes
                and candidate.batch_ref is not None
                and candidate.label_ref is not None
                and candidate.batch_ref() is batch
                and candidate.label_ref() is label
                and candidate.metrics == tuple(metrics)
                and candidate.request_digest is not None
            )
            if exact_objects:
                from quant_evaluator.runtime import backend_calibration
                identity = backend_calibration._calibration_identity_from_request_digest(
                    batch, label, metrics=metrics,
                    calibration_policy=candidate_policy,
                    gpu_policy=effective_gpu_policy,
                    request_digest=candidate.request_digest,
                    started=started,
                )
            else:
                identity = calibration_identity(
                    batch, label, metrics=metrics,
                    calibration_policy=candidate_policy,
                    gpu_policy=effective_gpu_policy,
                    started=started,
                )
        except Exception:
            return {
                "valid": False,
                "validation_seconds": time.monotonic() - started,
                "rejection_reason": "identity_validation_failed",
            }
        validation_seconds = float(identity.get(
            "setup_seconds", time.monotonic() - started))
        if identity["device_admission"] is not None:
            return {"valid": False, "validation_seconds": validation_seconds,
                    "rejection_reason": "device_admission_failed"}
        if identity["process_source_drifted"]:
            return {"valid": False, "validation_seconds": validation_seconds,
                    "rejection_reason": "source_drift"}
        if identity["source_metadata"].get("mode") != "strict_full_content":
            return {"valid": False, "validation_seconds": validation_seconds,
                    "rejection_reason": "source_mode_mismatch"}
        if identity["key"] != candidate.cache_key:
            return {"valid": False, "validation_seconds": validation_seconds,
                    "rejection_reason": "identity_mismatch"}
        return {"valid": True, "validation_seconds": validation_seconds}

    lookup = lookup_candidate_details(
        descriptor=descriptor, calibration_policy=None,
        gpu_policy=effective_gpu_policy, static_backend=static_backend,
        validator=validate,
    )
    validation_metadata = {
        "validation_seconds": lookup.validation_seconds,
        "rejection_reason": lookup.rejection_reason,
    }
    if lookup.winner is None:
        return static_backend, static_reason, False, validation_metadata
    return lookup.winner, "measured_auto_exact_candidate", True, validation_metadata
