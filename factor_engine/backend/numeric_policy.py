# -*- coding: utf-8 -*-
"""Unified numeric policy and float32 quantization certification.

This module composes the numeric contracts introduced in R40 into immutable,
hashable values. It deliberately does not select backend implementations yet;
the policy is the shared identity/evidence surface those selectors can consume.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from factor_engine.backend.elementwise_semantics import compute_precision_policy
from factor_engine.cleaned_operators._numpy_kernels import DEFAULT_DEGENERACY_POLICY
from factor_engine.runtime.execution_traits import NumericDeterminismLevel


def _stable_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToleranceProfile:
    """Scale-aware numeric and rank-level tolerance contract."""

    absolute: float = 1e-6
    relative: float = 1e-5
    ulp: int = 8
    min_rank_correlation: float = 0.999
    min_top_decile_overlap: float = 0.99
    max_sign_flip_rate: float = 1e-4
    max_ic_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.absolute < 0 or self.relative < 0 or self.ulp < 0:
            raise ValueError("numeric tolerances must be non-negative")
        for name in ("min_rank_correlation", "min_top_decile_overlap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in ("max_sign_flip_rate", "max_ic_delta"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericPolicy:
    """R42 numeric identity spanning compute, accumulation, output and edges."""

    compute_dtype: str = "float64"
    accumulation_dtype: str = "float64"
    output_dtype: str = "float64"
    determinism_level: str = NumericDeterminismLevel.DETERMINISTIC_WITHIN_TOLERANCE.value
    reduction_algorithm: str = "pairwise_or_compensated"
    division_policy: str = "protected_truth_table_v1"
    overflow_policy: str = "preserve_or_contractual_null"
    underflow_policy: str = "preserve_subnormal"
    degeneracy_policy: Mapping[str, Any] = field(
        default_factory=DEFAULT_DEGENERACY_POLICY.identity_payload
    )
    tolerance_profile: ToleranceProfile = field(default_factory=ToleranceProfile)
    rank_tie_policy: str = "operator_contract"
    quantile_interpolation: str = "operator_contract"
    moment_convention: str = "operator_contract"
    ewm_semantics: str = "operator_contract"

    def __post_init__(self) -> None:
        for name in ("compute_dtype", "accumulation_dtype", "output_dtype"):
            if str(getattr(self, name)) not in {"float32", "float64"}:
                raise ValueError(f"{name} must be float32 or float64")
        valid_determinism = {item.value for item in NumericDeterminismLevel}
        if self.determinism_level not in valid_determinism:
            raise ValueError(
                f"determinism_level must be one of {sorted(valid_determinism)}"
            )

    @classmethod
    def default(cls, *, output_dtype: str | None = None) -> "NumericPolicy":
        compute = compute_precision_policy()
        return cls(
            compute_dtype=compute.compute_dtype,
            accumulation_dtype="float64",
            output_dtype=output_dtype or compute.output_storage_precision,
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["degeneracy_policy"] = dict(self.degeneracy_policy)
        return payload

    def identity_hash(self) -> str:
        return _stable_digest(self.identity_payload())


@dataclass(frozen=True)
class QuantizationMetrics:
    max_abs_error: float
    max_relative_error: float
    max_ulp_error: int
    rank_correlation: float
    top_decile_overlap: float
    sign_flip_rate: float
    ic_delta: float
    finite_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Stable average-tie ranks without a scipy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 1.0
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    )
    if denom == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def quantization_metrics(
    reference: Sequence[float] | np.ndarray,
    quantized: Sequence[float] | np.ndarray,
    *,
    forward_returns: Sequence[float] | np.ndarray | None = None,
) -> QuantizationMetrics:
    """Measure float64-to-float32 impact using numeric and alpha-level metrics."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    out = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != out.shape:
        raise ValueError("reference and quantized arrays must have identical shape")
    mask = np.isfinite(ref) & np.isfinite(out)
    ref = ref[mask]
    out = out[mask]
    if ref.size == 0:
        raise ValueError("quantization certification requires at least one finite pair")

    diff = np.abs(ref - out)
    scale = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    rank_corr = _correlation(_rankdata(ref), _rankdata(out))
    top_n = max(1, int(np.ceil(ref.size * 0.1)))
    ref_top = set(np.argsort(ref, kind="mergesort")[-top_n:].tolist())
    out_top = set(np.argsort(out, kind="mergesort")[-top_n:].tolist())
    top_overlap = len(ref_top & out_top) / top_n
    nonzero = (ref != 0.0) & (out != 0.0)
    sign_flip = (
        float(np.mean(np.signbit(ref[nonzero]) != np.signbit(out[nonzero])))
        if np.any(nonzero)
        else 0.0
    )

    ic_delta = 0.0
    if forward_returns is not None:
        returns = np.asarray(forward_returns, dtype=np.float64).reshape(-1)
        if returns.shape != mask.shape:
            raise ValueError("forward_returns must match the unfiltered input shape")
        returns = returns[mask]
        valid_returns = np.isfinite(returns)
        if np.count_nonzero(valid_returns) >= 2:
            ref_ic = _correlation(
                _rankdata(ref[valid_returns]), _rankdata(returns[valid_returns])
            )
            out_ic = _correlation(
                _rankdata(out[valid_returns]), _rankdata(returns[valid_returns])
            )
            ic_delta = abs(ref_ic - out_ic)

    ref32 = ref.astype(np.float32)
    out32 = out.astype(np.float32)
    ordered_ref = ref32.view(np.int32).astype(np.int64)
    ordered_out = out32.view(np.int32).astype(np.int64)
    max_ulp = int(np.max(np.abs(ordered_ref - ordered_out)))
    return QuantizationMetrics(
        max_abs_error=float(diff.max()),
        max_relative_error=float((diff / scale).max()),
        max_ulp_error=max_ulp,
        rank_correlation=rank_corr,
        top_decile_overlap=float(top_overlap),
        sign_flip_rate=sign_flip,
        ic_delta=float(ic_delta),
        finite_count=int(ref.size),
    )


@dataclass(frozen=True)
class QuantizationCertificate:
    """Integrity-bound evidence permitting a production float32 publication."""

    factor_id: str
    source_dtype: str
    target_dtype: str
    numeric_policy_hash: str
    metrics: QuantizationMetrics
    tolerance_profile: ToleranceProfile
    production_eligible: bool
    certificate_hash: str
    schema_version: str = "quantization_certificate_v1"

    @classmethod
    def certify(
        cls,
        factor_id: str,
        reference: Sequence[float] | np.ndarray,
        *,
        forward_returns: Sequence[float] | np.ndarray | None = None,
        policy: NumericPolicy | None = None,
        tolerance_profile: ToleranceProfile | None = None,
    ) -> "QuantizationCertificate":
        policy = policy or NumericPolicy.default(output_dtype="float32")
        tolerance = tolerance_profile or policy.tolerance_profile
        ref = np.asarray(reference, dtype=np.float64)
        quantized = ref.astype(np.float32).astype(np.float64)
        metrics = quantization_metrics(
            ref, quantized, forward_returns=forward_returns
        )
        eligible = (
            metrics.max_abs_error <= tolerance.absolute
            and metrics.max_relative_error <= tolerance.relative
            and metrics.max_ulp_error <= tolerance.ulp
            and metrics.rank_correlation >= tolerance.min_rank_correlation
            and metrics.top_decile_overlap >= tolerance.min_top_decile_overlap
            and metrics.sign_flip_rate <= tolerance.max_sign_flip_rate
            and metrics.ic_delta <= tolerance.max_ic_delta
        )
        base = {
            "schema_version": "quantization_certificate_v1",
            "factor_id": str(factor_id),
            "source_dtype": "float64",
            "target_dtype": "float32",
            "numeric_policy_hash": policy.identity_hash(),
            "metrics": metrics.to_dict(),
            "tolerance_profile": tolerance.to_dict(),
            "production_eligible": eligible,
        }
        return cls(
            factor_id=str(factor_id),
            source_dtype="float64",
            target_dtype="float32",
            numeric_policy_hash=policy.identity_hash(),
            metrics=metrics,
            tolerance_profile=tolerance,
            production_eligible=eligible,
            certificate_hash=_stable_digest(base),
        )

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "factor_id": self.factor_id,
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "numeric_policy_hash": self.numeric_policy_hash,
            "metrics": self.metrics.to_dict(),
            "tolerance_profile": self.tolerance_profile.to_dict(),
            "production_eligible": self.production_eligible,
        }

    def validate_for(self, *, factor_id: str, target_dtype: str = "float32") -> bool:
        return bool(
            self.schema_version == "quantization_certificate_v1"
            and self.factor_id == str(factor_id)
            and self.source_dtype == "float64"
            and self.target_dtype == target_dtype
            and self.production_eligible
            and self.certificate_hash == _stable_digest(self._unsigned_payload())
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_payload(), "certificate_hash": self.certificate_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantizationCertificate":
        return cls(
            factor_id=str(payload.get("factor_id") or ""),
            source_dtype=str(payload.get("source_dtype") or ""),
            target_dtype=str(payload.get("target_dtype") or ""),
            numeric_policy_hash=str(payload.get("numeric_policy_hash") or ""),
            metrics=QuantizationMetrics(**dict(payload.get("metrics") or {})),
            tolerance_profile=ToleranceProfile(
                **dict(payload.get("tolerance_profile") or {})
            ),
            production_eligible=bool(payload.get("production_eligible")),
            certificate_hash=str(payload.get("certificate_hash") or ""),
            schema_version=str(payload.get("schema_version") or ""),
        )


def quantization_certificate_from(value: Any) -> QuantizationCertificate | None:
    if isinstance(value, QuantizationCertificate):
        return value
    if isinstance(value, Mapping):
        try:
            return QuantizationCertificate.from_dict(value)
        except (TypeError, ValueError):
            return None
    return None
