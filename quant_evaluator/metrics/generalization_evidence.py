"""Train-vs-validation generalization evidence (plan §13.6, R61-FI-022).

Plan §13.6 asks for evidence comparing **authorized** train and validation
artifacts — and explicitly: *"Do not compute this by exposing sealed Test."*
This module provides exactly that boundary:

- :class:`TrainVsValidationArtifact` — a frozen dataclass carrying the
  train/validation predictive dimensions, retention, and the four deltas
  (rankic / icir / sharpe / shape), plus the versioned policy constants that
  produced the grade.  It NEVER contains a Test artifact, a sealed split
  reference, or any hook to one: there is no Test field, and
  ``check_sealed_split_overlap`` (``contracts/sealed_split.py``) remains the
  QE-side gate that rejects any evaluation bound to a sealed split.  Tests
  pin the absence of a Test field by inspecting the dataclass.
- :class:`RetentionPolicy` — the versioned policy holding the retention
  grade anchors as constants (plan §13.6: ``S+ >= 0.90`` … ``D < 0.25``).
  Anchors are POLICY objects only; callers never hard-code them.
- :func:`compute_validation_retention` — the ROBUST retention measure
  required by plan §13.6.  The naive ``validation/train`` ratio is unstable
  when train is near zero or changes sign; the plan mandates a policy that
  combines, per factor:

  1. **absolute delta** ``|validation - train|`` (always well-defined);
  2. **relative retention** ``validation/train`` but ONLY where the
     denominator is stable (``|train| >= min_abs_train`` and the train
     metric is not near sign flip — see the note below);
  3. **confidence intervals** (a simple standard-error band on both means,
     recorded on the artifact);
  4. **sign consistency** (whether validation agrees with the train sign).

  The relative retention is `NaN` (never a fabricated number) whenever the
  train denominator is unstable — the artifact also carries the reason.

QE-FI-022-P0-001 (no blind division): ``compute_validation_retention``
never divides by an unstable denominator.  With ``train == 0`` the relative
retention is ``NaN`` and the grade falls back to absolute-delta evidence.
QE-FI-022-P0-002 (no sign-flip blind division): when ``train`` is a
positive number smaller than the sign-flip guard but non-zero, the ratio is
not blind-divided either — the reason field records ``sign_flip_guard`` and
retention is reported as ``NaN`` with the absolute delta still present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TrainVsValidationArtifact",
    "RetentionPolicy",
    "RetentionGrade",
    "compute_validation_retention",
    "compute_validation_retention_array",
    "compute_generalization_deltas",
    "project_generalization_metric",
    "RetentionGrade.from_value",
]


class RetentionGrade(str):
    """Closed vocabulary for retention grades (plan §13.6 anchors).

    Not a new "evidence status" — the 8-token ``EvidenceStatus`` vocabulary
    (contracts/evidence_status.py) stays the only status vocabulary; these
    are policy-controlled GRADE values computed FROM evidence, stored in the
    frozen artifact.
    """

    S_PLUS = "S+"
    S = "S"
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"

    @classmethod
    def from_value(cls, retention: Optional[float], policy: "RetentionPolicy" = None):
        """Map a retention value to a grade using the policy anchors.

        ``None``/``NaN`` retention maps to ``D`` with the reason recorded by
        the caller (missing evidence, never 0).
        """
        if policy is None:
            policy = RetentionPolicy()
        if retention is None or not np.isfinite(retention):
            return cls.D
        anchors = policy.grade_anchors  # list of (min_retention, grade)
        for threshold, grade in anchors:
            if retention >= threshold:
                return grade
        return cls.D


# ---------------------------------------------------------------------------
# Versioned policy (plan §13.6 anchors live ONLY here).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionPolicy:
    """Versioned retention policy (plan §13.6 anchors as constants).

    Attributes:
        policy_id: Stable policy family id.
        policy_version: SemVer policy version.
        grade_anchors: Ordered ``(min_retention, grade)`` pairs from the
            plan's initial anchors: ``S+ >= 0.90``, ``S >= 0.80``,
            ``A+ >= 0.70``, ``A >= 0.60``, ``B+ >= 0.50``, ``B >= 0.40``,
            ``C >= 0.25``, ``D < 0.25``.
        min_abs_train: Minimum ``abs(train)`` below which the relative
            retention is declared unstable (plan §13.6: do not blindly
            divide by tiny Train IC).  Default 0.05.
        sign_flip_guard: A train metric is "near sign flip" when
            ``abs(train) < sign_flip_guard * max(abs(validation), abs(train),
            1e-9)`` — i.e. validation disagrees with the train sign while
            train is small.  When triggered, relative retention is ``NaN``.
    """

    policy_id: str = "QE_RETENTION_POLICY"
    policy_version: str = "1.0.0"
    grade_anchors: Tuple[Tuple[float, str], ...] = (
        (0.90, "S+"),
        (0.80, "S"),
        (0.70, "A+"),
        (0.60, "A"),
        (0.50, "B+"),
        (0.40, "B"),
        (0.25, "C"),
    )
    min_abs_train: float = 0.05
    sign_flip_guard: float = 0.5

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.policy_version.strip():
            raise ValueError(
                "RetentionPolicy policy_id/policy_version must be non-empty"
            )
        for threshold, grade in self.grade_anchors:
            if grade not in {
                "S+", "S", "A+", "A", "B+", "B", "C", "D",
            }:
                raise ValueError(f"Unknown retention grade {grade!r}")
        prev = float("inf")
        for threshold, _grade in self.grade_anchors:
            if threshold >= prev:
                raise ValueError(
                    "grade_anchors must be strictly descending by threshold"
                )
            prev = threshold
        if self.min_abs_train < 0:
            raise ValueError("min_abs_train must be >= 0")


# ---------------------------------------------------------------------------
# The frozen comparison artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainVsValidationArtifact:
    """Frozen train-vs-validation comparison (plan §13.6).

    Fields (all per-factor arrays or scalars):

    - ``train_predictive_dimension`` / ``validation_predictive_dimension``:
      the predictive dimension used for each side (e.g. the mean daily rank
      IC of the train / validation evaluation).
    - ``validation_retention``: robust retention (``validation/train`` where
      the denominator is stable, else NaN with the reason recorded).
    - ``retention_reason``: why the relative retention is or is not defined
      (``"computed"`` / ``"train_near_zero"`` / ``"sign_flip_guard"`` /
      ``"insufficient_data"`` / ``"not_computed"``).
    - ``train_validation_rankic_delta`` / ``train_validation_icir_delta`` /
      ``train_validation_sharpe_delta`` / ``train_validation_shape_delta``:
      absolute deltas between validation and train (never a ratio).
    - ``parameter_generalization``: an optional scalar summary (e.g. a
      weighted combination of the retention evidence).  When not
      computable it is ``None`` (never 0).
    - ``retention_grade``: the versioned-policy grade for the retention.
    - ``confidence_interval_95``: ``(lower, upper)`` band on the retention
      (None when not computable).
    - ``sign_consistent``: whether validation agrees with the train sign.
    - ``policy_id`` / ``policy_version``: which policy produced the grade.

    **No Test exposure**: this artifact has NO sealed/Test split fields and
    no reference to one.  Constructing it is the only way consumers read
    train-vs-validation evidence; the sealed-test gate remains
    ``contracts/sealed_split.py``.
    """

    train_predictive_dimension: Tuple[Optional[float], ...]
    validation_predictive_dimension: Tuple[Optional[float], ...]
    validation_retention: Tuple[Optional[float], ...]
    retention_reason: Tuple[str, ...]
    train_validation_rankic_delta: Tuple[Optional[float], ...]
    train_validation_icir_delta: Tuple[Optional[float], ...]
    train_validation_sharpe_delta: Tuple[Optional[float], ...]
    train_validation_shape_delta: Tuple[Optional[float], ...]
    parameter_generalization: Tuple[Optional[float], ...] = ()
    retention_grade: Tuple[str, ...] = ()
    confidence_interval_95: Optional[Tuple[Optional[float], Optional[float]]] = None
    confidence_intervals_95: Tuple[Tuple[Optional[float], Optional[float]], ...] = ()
    confidence_interval_status: Tuple[str, ...] = ()
    factor_ids: Tuple[str, ...] = ()
    factor_versions: Tuple[str, ...] = ()
    metric_instance: str = "predictive_dimension"
    metric_instance_refs: Tuple[Tuple[str, str], ...] = ()
    train_split_ref: str = "train"
    validation_split_ref: str = "validation"
    sample_unit: Optional[str] = None
    sample_dependence: Optional[str] = None
    train_sample_ids: Tuple[str, ...] = ()
    validation_sample_ids: Tuple[str, ...] = ()
    n_train_samples: Tuple[int, ...] = ()
    n_validation_samples: Tuple[int, ...] = ()
    sign_consistent: Tuple[Optional[bool], ...] = ()
    policy_id: str = "QE_RETENTION_POLICY"
    policy_version: str = "1.0.0"
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        n = len(self.train_predictive_dimension)
        if not n:
            raise ValueError(
                "TrainVsValidationArtifact requires at least one factor"
            )
        if len(self.validation_predictive_dimension) != n:
            raise ValueError("predictive dimensions must have the same length")
        if len(self.validation_retention) != n:
            raise ValueError("validation_retention length mismatch")
        if len(self.retention_reason) != n:
            raise ValueError("retention_reason length mismatch")
        if len(self.train_validation_rankic_delta) != n:
            raise ValueError("rankic delta length mismatch")
        if len(self.train_validation_icir_delta) != n:
            raise ValueError("icir delta length mismatch")
        if len(self.train_validation_sharpe_delta) != n:
            raise ValueError("sharpe delta length mismatch")
        if len(self.train_validation_shape_delta) != n:
            raise ValueError("shape delta length mismatch")
        if self.retention_grade and len(self.retention_grade) != n:
            raise ValueError("retention_grade length mismatch")
        if self.parameter_generalization and len(self.parameter_generalization) != n:
            raise ValueError("parameter_generalization length mismatch")
        if self.sign_consistent and len(self.sign_consistent) != n:
            raise ValueError("sign_consistent length mismatch")
        for name in ("confidence_intervals_95", "confidence_interval_status",
                     "factor_ids", "factor_versions", "n_train_samples",
                     "n_validation_samples"):
            value = getattr(self, name)
            if value and len(value) != n:
                raise ValueError(f"{name} length mismatch")
        if self.factor_ids and len(set(zip(self.factor_ids, self.factor_versions))) != n:
            raise ValueError("factor_id/factor_version identities must be unique")
        if not self.metric_instance.strip():
            raise ValueError("metric_instance must be non-empty")
        if self.metric_instance_refs:
            refs = dict(self.metric_instance_refs)
            if len(refs) != len(self.metric_instance_refs) or any(
                not key or not isinstance(value, str) or not value.strip()
                for key, value in self.metric_instance_refs
            ):
                raise ValueError("metric_instance_refs must contain unique non-empty bindings")
        if not self.train_split_ref.strip() or not self.validation_split_ref.strip():
            raise ValueError("train and validation split refs must be non-empty")
        if self.train_split_ref == self.validation_split_ref:
            raise ValueError("train and validation split refs must be distinct")
        if not self.policy_id.strip() or not self.policy_version.strip():
            raise ValueError("policy_id/policy_version must be non-empty")
        from quant_evaluator.contracts.metric_artifacts import FrozenMapping
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "train_predictive_dimension": list(self.train_predictive_dimension),
            "validation_predictive_dimension": list(self.validation_predictive_dimension),
            "validation_retention": list(self.validation_retention),
            "retention_reason": list(self.retention_reason),
            "train_validation_rankic_delta": list(self.train_validation_rankic_delta),
            "train_validation_icir_delta": list(self.train_validation_icir_delta),
            "train_validation_sharpe_delta": list(self.train_validation_sharpe_delta),
            "train_validation_shape_delta": list(self.train_validation_shape_delta),
            "parameter_generalization": self.parameter_generalization,
            "retention_grade": list(self.retention_grade),
            "confidence_interval_95": (
                None
                if self.confidence_interval_95 is None
                else list(self.confidence_interval_95)
            ),
            "confidence_intervals_95": [list(v) for v in self.confidence_intervals_95],
            "confidence_interval_status": list(self.confidence_interval_status),
            "factor_ids": list(self.factor_ids),
            "factor_versions": list(self.factor_versions),
            "metric_instance": self.metric_instance,
            "metric_instance_refs": [list(value) for value in self.metric_instance_refs],
            "train_split_ref": self.train_split_ref,
            "validation_split_ref": self.validation_split_ref,
            "sample_unit": self.sample_unit,
            "sample_dependence": self.sample_dependence,
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
            "n_train_samples": list(self.n_train_samples),
            "n_validation_samples": list(self.n_validation_samples),
            "sign_consistent": list(self.sign_consistent),
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainVsValidationArtifact":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            train_predictive_dimension=tuple(payload.get("train_predictive_dimension", ())),
            validation_predictive_dimension=tuple(payload.get("validation_predictive_dimension", ())),
            validation_retention=tuple(payload.get("validation_retention", ())),
            retention_reason=tuple(payload.get("retention_reason", ())),
            train_validation_rankic_delta=tuple(payload.get("train_validation_rankic_delta", ())),
            train_validation_icir_delta=tuple(payload.get("train_validation_icir_delta", ())),
            train_validation_sharpe_delta=tuple(payload.get("train_validation_sharpe_delta", ())),
            train_validation_shape_delta=tuple(payload.get("train_validation_shape_delta", ())),
            parameter_generalization=tuple(payload.get("parameter_generalization", ())),
            retention_grade=tuple(payload.get("retention_grade", ())),
            confidence_interval_95=(
                None
                if payload.get("confidence_interval_95") is None
                else (payload["confidence_interval_95"][0], payload["confidence_interval_95"][1])
            ),
            confidence_intervals_95=tuple(tuple(v) for v in payload.get("confidence_intervals_95", ())),
            confidence_interval_status=tuple(payload.get("confidence_interval_status", ())),
            factor_ids=tuple(payload.get("factor_ids", ())),
            factor_versions=tuple(payload.get("factor_versions", ())),
            metric_instance=payload.get("metric_instance", "predictive_dimension"),
            metric_instance_refs=tuple(tuple(value) for value in payload.get("metric_instance_refs", ())),
            train_split_ref=payload.get("train_split_ref", "train"),
            validation_split_ref=payload.get("validation_split_ref", "validation"),
            sample_unit=payload.get("sample_unit"),
            sample_dependence=payload.get("sample_dependence"),
            train_sample_ids=tuple(payload.get("train_sample_ids", ())),
            validation_sample_ids=tuple(payload.get("validation_sample_ids", ())),
            n_train_samples=tuple(payload.get("n_train_samples", ())),
            n_validation_samples=tuple(payload.get("n_validation_samples", ())),
            sign_consistent=tuple(payload.get("sign_consistent", ())),
            policy_id=payload.get("policy_id", "QE_RETENTION_POLICY"),
            policy_version=payload.get("policy_version", "1.0.0"),
            metadata=payload.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Robust retention computation (plan §13.6 four-way combination).
# ---------------------------------------------------------------------------


def _standard_error(values: np.ndarray) -> Optional[float]:
    """Std-error of the mean over finite values, or None for <2 finite obs."""
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return None
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def _confidence_band(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    covariance_of_means: float = 0.0,
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Per-factor 95% delta-method band from actual sample observations.

    ``train`` and ``validation`` are observations for one factor, never
    summaries belonging to different factors.  Independent splits imply a
    zero covariance term.  The gradient is evaluated directly, so a zero
    validation mean with non-zero validation variance retains non-zero
    uncertainty.
    """
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    if train.ndim != 1 or validation.ndim != 1:
        raise ValueError("confidence evidence must be one-dimensional per factor")
    finite_t = train[np.isfinite(train)]
    finite_v = validation[np.isfinite(validation)]
    se_t = _standard_error(finite_t)
    se_v = _standard_error(finite_v)
    if se_t is None or se_v is None:
        return (None, None)
    mean_t = float(np.mean(finite_t))
    mean_v = float(np.mean(finite_v))
    if abs(mean_t) < 1e-12:
        return (None, None)
    rel = mean_v / mean_t
    ratio_variance = (se_v ** 2 / mean_t ** 2
                      + mean_v ** 2 * se_t ** 2 / mean_t ** 4
                      - 2 * mean_v * covariance_of_means / mean_t ** 3)
    width = 1.96 * np.sqrt(max(0.0, ratio_variance))
    return (rel - width, rel + width)


def _strict_summary(name: str, values: np.ndarray) -> np.ndarray:
    """Validate a scalar-per-factor summary without erasing its axis."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional factor summary")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    return result


def _strict_pair(train_values, validation_values, *, prefix="evidence"):
    train = _strict_summary(f"train_{prefix}", train_values)
    valid = _strict_summary(f"validation_{prefix}", validation_values)
    if train.shape != valid.shape:
        raise ValueError(
            f"train/validation {prefix} must have identical factor counts; "
            "provide explicit factor identities to the artifact builder for reordering"
        )
    return train, valid


def _sign_flip_near(train: float, validation: float, guard: float) -> bool:
    """True when train is small AND validation disagrees with train's sign."""
    if train == 0.0:
        return True
    if not (np.isfinite(train) and np.isfinite(validation)):
        return False
    if np.sign(train) * np.sign(validation) < 0:
        # Opposite signs and train is relatively small.
        return abs(train) < guard * max(abs(validation), abs(train), 1e-9)
    return False


def compute_validation_retention_array(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    policy: Optional[RetentionPolicy] = None,
) -> Tuple[Tuple[Optional[float], ...], Tuple[str, ...]]:
    """Robust per-factor validation retention, (retention, reason).

    Plan §13.6 robust policy, per factor:

    - denominator stable (``abs(train) >= min_abs_train``) AND train not
      near sign flip: ``validation / train`` with reason ``"computed"``.
    - ``abs(train) < min_abs_train``: relative retention is ``NaN`` with
      reason ``"train_near_zero"`` (absolute delta still available on the
      artifact — never a fabricated ratio).
    - train near sign flip (opposite sign to validation with small train):
      ``NaN`` with reason ``"sign_flip_guard"``.
    - insufficient finite data (fewer than 2 finite values on either side):
      ``NaN`` with reason ``"insufficient_data"``.
    """
    if policy is None:
        policy = RetentionPolicy()
    train, valid = _strict_pair(train_values, validation_values, prefix="retention evidence")
    n = train.shape[0]
    retentions: List[Optional[float]] = []
    reasons: List[str] = []
    for i in range(n):
        t = train[i]
        v = valid[i]
        if not (np.isfinite(t) and np.isfinite(v)):
            retentions.append(None)
            reasons.append("insufficient_data")
            continue
        if abs(t) < policy.min_abs_train:
            retentions.append(None)
            reasons.append("train_near_zero")
            continue
        if _sign_flip_near(t, v, policy.sign_flip_guard):
            retentions.append(None)
            reasons.append("sign_flip_guard")
            continue
        retentions.append(float(v / t))
        reasons.append("computed")
    return tuple(retentions), tuple(reasons)


def compute_validation_retention(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    policy: Optional[RetentionPolicy] = None,
) -> float:
    """Mean robust retention across factors, ``NaN`` when all are unstable.

    Averages only the factors whose relative retention is well-defined
    (reason ``"computed"``); if none are, returns ``NaN`` (never 0 — a
    fabricated retention is a capability lie).  Intended for the case where
    ``train_values`` / ``validation_values`` are 1-D scalars per factor.
    """
    retentions, reasons = compute_validation_retention_array(
        train_values, validation_values, policy
    )
    computed = [
        r for r, reason in zip(retentions, reasons)
        if r is not None and reason == "computed"
    ]
    if not computed:
        return float("nan")
    return float(np.mean(computed))


def compute_generalization_deltas(
    train_values: np.ndarray,
    validation_values: np.ndarray,
) -> Tuple[Tuple[Optional[float], ...], ...]:
    """Absolute deltas ``validation - train`` per factor.

    Returns a tuple of four delta tuples (rankic / icir / sharpe / shape)
    when each input pair is provided; missing inputs (``None``) produce an
    all-``None`` tuple for that metric.  Each element is ``None`` when
    either side is not finite.  Deltas are absolute differences — NEVER a
    ratio — so they are well-defined even when train is near zero.
    """
    if train_values is None or validation_values is None:
        return ((), (), (), ())
    train, valid = _strict_pair(train_values, validation_values, prefix="delta evidence")
    n = train.shape[0]
    deltas: List[Optional[float]] = []
    for i in range(n):
        t = train[i]
        v = valid[i]
        if not (np.isfinite(t) and np.isfinite(v)):
            deltas.append(None)
        else:
            deltas.append(float(v - t))
    return (tuple(deltas),)


def build_train_validation_artifact(
    train_dim: np.ndarray,
    validation_dim: np.ndarray,
    *,
    train_rankic: Optional[np.ndarray] = None,
    validation_rankic: Optional[np.ndarray] = None,
    train_icir: Optional[np.ndarray] = None,
    validation_icir: Optional[np.ndarray] = None,
    train_sharpe: Optional[np.ndarray] = None,
    validation_sharpe: Optional[np.ndarray] = None,
    train_shape: Optional[np.ndarray] = None,
    validation_shape: Optional[np.ndarray] = None,
    train_factor_ids: Optional[Tuple[str, ...]] = None,
    validation_factor_ids: Optional[Tuple[str, ...]] = None,
    train_factor_versions: Optional[Tuple[str, ...]] = None,
    validation_factor_versions: Optional[Tuple[str, ...]] = None,
    metric_instance: str = "predictive_dimension",
    metric_instance_refs: Optional[Dict[str, str]] = None,
    train_split_ref: str = "train",
    validation_split_ref: str = "validation",
    train_samples: Optional[np.ndarray] = None,
    validation_samples: Optional[np.ndarray] = None,
    sample_unit: Optional[str] = None,
    sample_dependence: Optional[str] = None,
    train_sample_ids: Optional[Tuple[str, ...]] = None,
    validation_sample_ids: Optional[Tuple[str, ...]] = None,
    policy: Optional[RetentionPolicy] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TrainVsValidationArtifact:
    """Build a :class:`TrainVsValidationArtifact` from train/validation arrays.

    Computes the robust retention (with reasons), the four absolute deltas,
    sign consistency, the versioned-policy retention grade, and a parameter
    generalization summary (the mean of the computed relative retentions,
    or ``None`` when none are stable — never 0).

    The inputs are per-factor arrays; each factor contributes one entry.
    ``train_dim`` / ``validation_dim`` are the predictive dimensions
    (required).  Any missing delta input contributes an all-``None`` delta
    tuple with the corresponding reason (missing evidence ≠ 0).
    """
    if policy is None:
        policy = RetentionPolicy()
    train_dim = _strict_summary("train_dim", train_dim)
    validation_dim = _strict_summary("validation_dim", validation_dim)

    identity_args = (train_factor_ids, validation_factor_ids,
                     train_factor_versions, validation_factor_versions)
    if not all(value is not None for value in identity_args):
        raise ValueError("explicit factor identities and versions are required for both splits")
    if metric_instance == "predictive_dimension" or not metric_instance.strip():
        raise ValueError("an explicit non-default metric_instance is required")
    if all(value is not None for value in identity_args):
        train_keys = tuple(zip(train_factor_ids, train_factor_versions))
        valid_keys = tuple(zip(validation_factor_ids, validation_factor_versions))
        if len(train_keys) != train_dim.size or len(valid_keys) != validation_dim.size:
            raise ValueError("factor identities must match their summary lengths")
        if len(set(train_keys)) != len(train_keys) or len(set(valid_keys)) != len(valid_keys):
            raise ValueError("factor_id/factor_version identities must be unique")
        if set(train_keys) != set(valid_keys):
            missing_valid = sorted(set(train_keys) - set(valid_keys))
            missing_train = sorted(set(valid_keys) - set(train_keys))
            raise ValueError(
                f"train/validation factor identities differ; missing_validation={missing_valid}, "
                f"missing_train={missing_train}"
            )
        valid_positions = {key: i for i, key in enumerate(valid_keys)}
        order = np.asarray([valid_positions[key] for key in train_keys], dtype=np.int64)
        validation_dim = validation_dim[order]
        factor_ids = tuple(train_factor_ids)
        factor_versions = tuple(train_factor_versions)
    n = train_dim.size

    def _aligned_optional(name, values, *, validation=False):
        if values is None:
            return None
        result = _strict_summary(name, values)
        expected = len(order) if validation else n
        if result.size != expected:
            raise ValueError(f"{name} must contain one value per identified factor")
        return result[order] if validation else result

    train_rankic = _aligned_optional("train_rankic", train_rankic)
    validation_rankic = _aligned_optional("validation_rankic", validation_rankic, validation=True)
    train_icir = _aligned_optional("train_icir", train_icir)
    validation_icir = _aligned_optional("validation_icir", validation_icir, validation=True)
    train_sharpe = _aligned_optional("train_sharpe", train_sharpe)
    validation_sharpe = _aligned_optional("validation_sharpe", validation_sharpe, validation=True)
    train_shape = _aligned_optional("train_shape", train_shape)
    validation_shape = _aligned_optional("validation_shape", validation_shape, validation=True)
    instance_refs = dict(metric_instance_refs or {})
    supplied_pairs = {
        "rankic": (train_rankic, validation_rankic),
        "icir": (train_icir, validation_icir),
        "sharpe": (train_sharpe, validation_sharpe),
        "shape": (train_shape, validation_shape),
    }
    required_refs = {name for name, pair in supplied_pairs.items() if any(v is not None for v in pair)}
    if required_refs - set(instance_refs):
        raise ValueError(
            f"metric_instance_refs missing supplied evidence bindings: {sorted(required_refs - set(instance_refs))}"
        )

    retentions, reasons = compute_validation_retention_array(
        train_dim, validation_dim, policy
    )

    def _delta_pair(t: Optional[np.ndarray], v: Optional[np.ndarray]):
        if t is None or v is None:
            return tuple([None] * n)
        return compute_generalization_deltas(t, v)[0]

    rankic_delta = _delta_pair(train_rankic, validation_rankic)
    icir_delta = _delta_pair(train_icir, validation_icir)
    sharpe_delta = _delta_pair(train_sharpe, validation_sharpe)
    shape_delta = _delta_pair(train_shape, validation_shape)

    sign_consistent: List[Optional[bool]] = []
    for i in range(n):
        t = float(train_dim[i])
        v = float(validation_dim[i])
        if not (np.isfinite(t) and np.isfinite(v)):
            sign_consistent.append(None)
        else:
            sign_consistent.append(bool(np.sign(t) == np.sign(v)))

    grades = [
        RetentionGrade.from_value(r, policy) for r in retentions
    ]

    computed = [r for r, reason in zip(retentions, reasons) if r is not None]
    # A generalization result is factor-scoped.  Never average across the
    # candidate axis: adding an unrelated factor must not change an existing
    # factor's evidence.
    parameter_generalization = tuple(retentions)

    cis = tuple((None, None) for _ in range(n))
    ci_status = tuple("CI_NOT_IDENTIFIED_FROM_SUMMARY" for _ in range(n))
    n_train = tuple(0 for _ in range(n))
    n_valid = tuple(0 for _ in range(n))
    if (train_samples is None) != (validation_samples is None):
        raise ValueError("train_samples and validation_samples must be supplied together")
    if train_samples is not None:
        if not sample_unit or sample_unit not in {"time", "fold", "draw"}:
            raise ValueError("sample_unit must be one of time, fold, or draw")
        if sample_dependence not in {"independent", "paired"}:
            raise ValueError("sample_dependence must be independent or paired")
        ts = np.asarray(train_samples, dtype=np.float64)
        vs = np.asarray(validation_samples, dtype=np.float64)
        if ts.ndim != 2 or vs.ndim != 2:
            raise ValueError("sample evidence must have shape sample x factor")
        if ts.shape[1] != n or vs.shape[1] != len(order):
            raise ValueError("sample evidence factor axis does not match factor identities")
        vs = vs[:, order]
        if train_sample_ids is not None and len(train_sample_ids) != ts.shape[0]:
            raise ValueError("train_sample_ids length mismatch")
        if validation_sample_ids is not None and len(validation_sample_ids) != vs.shape[0]:
            raise ValueError("validation_sample_ids length mismatch")
        if train_sample_ids is None or validation_sample_ids is None:
            raise ValueError("sample IDs are required for auditable confidence evidence")
        if len(set(train_sample_ids)) != len(train_sample_ids):
            raise ValueError("train_sample_ids must be unique; aliases are not new evidence")
        if len(set(validation_sample_ids)) != len(validation_sample_ids):
            raise ValueError("validation_sample_ids must be unique; aliases are not new evidence")
        if sample_dependence == "paired" and tuple(train_sample_ids) != tuple(validation_sample_ids):
            raise ValueError("paired confidence evidence requires identical sample IDs")
        covariance_inputs = []
        ci_inputs = []
        for i in range(n):
            if sample_dependence == "independent":
                covariance_inputs.append(0.0)
                ci_inputs.append((ts[:, i], vs[:, i]))
                continue
            joint = np.isfinite(ts[:, i]) & np.isfinite(vs[:, i])
            paired_train, paired_valid = ts[joint, i], vs[joint, i]
            ci_inputs.append((paired_train, paired_valid))
            covariance_inputs.append(
                float(np.cov(ts[joint, i], vs[joint, i], ddof=1)[0, 1] / joint.sum())
                if joint.sum() >= 2 else 0.0
            )
        n_train = tuple(int(np.isfinite(left).sum()) for left, _ in ci_inputs)
        n_valid = tuple(int(np.isfinite(right).sum()) for _, right in ci_inputs)
        # Daily observations require an explicit serial-dependence estimator;
        # this delta-method producer does not silently pretend they are IID.
        if sample_unit == "time":
            raise ValueError(
                "time-sample confidence requires a declared temporal/HAC dependence model"
            )
        else:
            for i, (left, right) in enumerate(ci_inputs):
                if sample_unit == "fold":
                    mean_left = float(np.mean(left[np.isfinite(left)])) if np.isfinite(left).any() else np.nan
                    mean_right = float(np.mean(right[np.isfinite(right)])) if np.isfinite(right).any() else np.nan
                    if not (np.isclose(mean_left, train_dim[i], rtol=1e-10, atol=1e-12)
                            and np.isclose(mean_right, validation_dim[i], rtol=1e-10, atol=1e-12)):
                        raise ValueError("summary values disagree with supplied fold samples")
            if sample_unit == "draw":
                if sample_dependence != "paired":
                    raise ValueError("precomputed draw evidence must be paired by draw ID")
                ratio_cis = []
                for left, right in ci_inputs:
                    stable = np.isfinite(left) & np.isfinite(right) & (np.abs(left) >= policy.min_abs_train)
                    ratios = right[stable] / left[stable]
                    ratio_cis.append(
                        tuple(float(v) for v in np.quantile(ratios, [.025, .975]))
                        if ratios.size >= 2 else (None, None)
                    )
                cis = tuple(ratio_cis)
            else:
                cis = tuple(
                    _confidence_band(left, right, covariance_of_means=covariance_inputs[i])
                    for i, (left, right) in enumerate(ci_inputs)
                )
            ci_status = tuple(
                "VALID" if lo is not None and hi is not None
                else ("UNSTABLE_TRAIN_DENOMINATOR" if abs(float(np.nanmean(ci_inputs[i][0]))) < 1e-12
                      else "INSUFFICIENT_SAMPLES")
                for i, (lo, hi) in enumerate(cis)
            )
    ci = cis[0] if n == 1 and ci_status[0] == "VALID" else None

    return TrainVsValidationArtifact(
        train_predictive_dimension=tuple(float(v) for v in train_dim),
        validation_predictive_dimension=tuple(float(v) for v in validation_dim),
        validation_retention=retentions,
        retention_reason=reasons,
        train_validation_rankic_delta=rankic_delta,
        train_validation_icir_delta=icir_delta,
        train_validation_sharpe_delta=sharpe_delta,
        train_validation_shape_delta=shape_delta,
        parameter_generalization=parameter_generalization,
        retention_grade=tuple(str(g) for g in grades),
        confidence_interval_95=ci,
        confidence_intervals_95=cis,
        confidence_interval_status=ci_status,
        factor_ids=factor_ids,
        factor_versions=factor_versions,
        metric_instance=metric_instance,
        metric_instance_refs=tuple(sorted(instance_refs.items())),
        train_split_ref=train_split_ref,
        validation_split_ref=validation_split_ref,
        sample_unit=sample_unit,
        sample_dependence=sample_dependence,
        train_sample_ids=tuple(train_sample_ids or ()),
        validation_sample_ids=tuple(validation_sample_ids or ()),
        n_train_samples=n_train,
        n_validation_samples=n_valid,
        sign_consistent=tuple(sign_consistent),
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        metadata=metadata,
    )


_ARTIFACT_METRIC_FIELDS = {
    "train_predictive_dimension": "train_predictive_dimension",
    "validation_predictive_dimension": "validation_predictive_dimension",
    "validation_retention": "validation_retention",
    "train_validation_rankic_delta": "train_validation_rankic_delta",
    "train_validation_icir_delta": "train_validation_icir_delta",
    "train_validation_sharpe_delta": "train_validation_sharpe_delta",
    "train_validation_shape_delta": "train_validation_shape_delta",
    "parameter_generalization": "parameter_generalization",
}


def project_generalization_metric(
    artifact: TrainVsValidationArtifact,
    metric_id: str,
    *,
    expected_factor_ids: Tuple[str, ...],
    expected_factor_versions: Tuple[str, ...],
    producer_version: str,
):
    """Project one public scalar metric from typed comparison evidence."""
    if not isinstance(artifact, TrainVsValidationArtifact):
        raise TypeError("generalization_evidence must be TrainVsValidationArtifact")
    if metric_id not in _ARTIFACT_METRIC_FIELDS:
        raise ValueError(f"unsupported generalization metric {metric_id!r}")
    if tuple(artifact.factor_ids) != tuple(expected_factor_ids):
        raise ValueError("generalization factor identities do not match the evaluation request")
    if tuple(artifact.factor_versions) != tuple(expected_factor_versions):
        raise ValueError("generalization factor versions do not match the evaluation request")
    raw = getattr(artifact, _ARTIFACT_METRIC_FIELDS[metric_id])
    values = np.asarray([np.nan if value is None else value for value in raw], dtype=np.float64)
    if values.shape != (len(expected_factor_ids),):
        raise ValueError("generalization metric does not contain one value per factor")
    if metric_id == "train_predictive_dimension":
        counts = artifact.n_train_samples
    elif metric_id == "validation_predictive_dimension":
        counts = artifact.n_validation_samples
    else:
        counts = tuple(min(a, b) for a, b in zip(
            artifact.n_train_samples, artifact.n_validation_samples
        ))
    if not counts or not any(counts):
        counts = tuple(1 if np.isfinite(value) else 0 for value in values)
    ref_key = {
        "train_validation_rankic_delta": "rankic",
        "train_validation_icir_delta": "icir",
        "train_validation_sharpe_delta": "sharpe",
        "train_validation_shape_delta": "shape",
    }.get(metric_id)
    metric_instance = (
        dict(artifact.metric_instance_refs).get(ref_key)
        if ref_key is not None else artifact.metric_instance
    )
    if not metric_instance:
        raise ValueError(f"generalization metric {metric_id} lacks a bound metric instance")
    from quant_evaluator.contracts.axis_refs import FactorAxisRef
    from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
    return ScalarMetricArtifact(
        metric_id=metric_id,
        domain="generalization",
        values=values,
        factor_axis=FactorAxisRef(
            tuple(expected_factor_ids), tuple(expected_factor_versions)
        ),
        producer_version=producer_version,
        provenance={
            "observation_counts": tuple(int(value) for value in counts),
            "sample_unit": artifact.sample_unit or "split_summary",
            "factor_versions": artifact.factor_versions,
            "metric_instance": metric_instance,
            "train_split_ref": artifact.train_split_ref,
            "validation_split_ref": artifact.validation_split_ref,
            "confidence_intervals_95": artifact.confidence_intervals_95,
            "confidence_interval_status": artifact.confidence_interval_status,
        },
    )
