"""Explicit exact-batch measured auto selection for the public facade."""
from dataclasses import dataclass, field, replace

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy, evaluate_calibrated_batch,
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
    result = evaluate_calibrated_batch(
        batch, label, metrics=metrics, calibration_policy=options.policy,
        cache=options.cache, gpu_policy=gpu_policy,
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
