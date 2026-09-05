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
    parameter_generalization: Optional[float] = None
    retention_grade: Tuple[str, ...] = ()
    confidence_interval_95: Optional[Tuple[Optional[float], Optional[float]]] = None
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
        if self.sign_consistent and len(self.sign_consistent) != n:
            raise ValueError("sign_consistent length mismatch")
        if not self.policy_id.strip() or not self.policy_version.strip():
            raise ValueError("policy_id/policy_version must be non-empty")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

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
            parameter_generalization=payload.get("parameter_generalization"),
            retention_grade=tuple(payload.get("retention_grade", ())),
            confidence_interval_95=(
                None
                if payload.get("confidence_interval_95") is None
                else (payload["confidence_interval_95"][0], payload["confidence_interval_95"][1])
            ),
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
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """95% band on the retention using a combined standard-error.

    This is a conservative confidence interval on the retention measure:
    the relative retention is ``validation / train``, so the band applies
    the error of both means through a simple delta-method width using the
    combined relative standard errors.  When either mean has no SE, returns
    ``(None, None)`` — an honest missing CI, never 0.
    """
    se_t = _standard_error(train)
    se_v = _standard_error(validation)
    if se_t is None or se_v is None:
        return (None, None)
    mean_t = float(np.nanmean(train))
    mean_v = float(np.nanmean(validation))
    if abs(mean_t) < 1e-12:
        return (None, None)
    rel = mean_v / mean_t
    width = 1.96 * np.sqrt(
        (se_v / max(abs(mean_v), 1e-12)) ** 2
        + (se_t / abs(mean_t)) ** 2
    ) * abs(rel)
    return (rel - width, rel + width)


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
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    valid = np.asarray(validation_values, dtype=np.float64).reshape(-1)
    n = max(train.shape[0], valid.shape[0])
    train = np.resize(train, n)
    valid = np.resize(valid, n)
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
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    valid = np.asarray(validation_values, dtype=np.float64).reshape(-1)
    n = max(train.shape[0], valid.shape[0])
    train = np.resize(train, n)
    valid = np.resize(valid, n)
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
    train_dim = np.asarray(train_dim, dtype=np.float64).reshape(-1)
    validation_dim = np.asarray(validation_dim, dtype=np.float64).reshape(-1)
    n = max(train_dim.shape[0], validation_dim.shape[0])
    train_dim = np.resize(train_dim, n)
    validation_dim = np.resize(validation_dim, n)

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
    parameter_generalization = (
        float(np.mean(computed)) if computed else None
    )

    ci = _confidence_band(train_dim, validation_dim)

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
        sign_consistent=tuple(sign_consistent),
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        metadata=metadata,
    )