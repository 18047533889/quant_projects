"""Split contracts for SearchRunner.

The validated boundary here is structural plus (optionally) temporal:
purge/embargo/label-horizon leakage is rejected when the plan carries a
time axis. It still does not sandbox evaluator access to data or prove
PIT correctness; a trusted QE adapter remains responsible for enforcing
the plan during evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, List

import numpy as np


class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitPlan:
    """Immutable, externally supplied split boundary description.

    Optional temporal-leakage fields (backward compatible: existing plans
    without them validate exactly as before):

    - ``time_index``: positional ordering of samples along the time axis.
      ``None`` means the plan is purely positional and leakage constraints
      below are unexpressible (setting any of them then fails closed).
    - ``label_horizon``: number of trailing positions a training label
      reaches into the future.
    - ``label_bundle``: optional LabelBundle with real timestamps for
      label windows, taking precedence over label_horizon.
    - ``purge``: extra positions removed around the train/test boundary.
    - ``embargo``: positions after a test segment that train data must not
      occupy (coefficient/execution leakage).
    - ``validation_embargo``: embargo-like gap between train and validation
      segments (prevents train -> validation contamination).
    """

    split_id: str
    train_mask: Any
    validation_mask: Any
    test_mask: Any
    metadata: Dict[str, Any]
    time_index: Optional[Tuple] = None
    label_horizon: int = 0
    label_bundle: Optional["LabelBundle"] = None
    purge: int = 0
    embargo: int = 0
    validation_embargo: int = 0

    def __post_init__(self):
        validate_split_plan(self)


@dataclass(frozen=True)
class EvaluationProtocol:
    """Evaluator plus a validated split plan for SearchRunner safe mode.

    This is a contract boundary, not a data sandbox: the callback can still
    violate the plan unless its adapter enforces it.
    """

    split_plan: SplitPlan
    evaluator: Callable[[Any, int], Dict[str, Any]]

    def __post_init__(self):
        validate_split_plan(self.split_plan)
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, trial: Any, fidelity: int) -> Dict[str, Any]:
        return self.evaluator(trial, fidelity)


@dataclass
class SearchEvaluationResult:
    evaluation_id: str
    trial_id: str
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    diagnostics: Dict[str, Any]

    def selection_score(self, metric: str = "rank_ic") -> float:
        return self.validation_metrics[metric]


@dataclass(frozen=True)
class LabelBundle:
    """Bundle of label information for temporal leakage validation.

    Real timestamps take precedence over the integer ``label_horizon``
    fallback.  Scalar ``label_start_time`` / ``label_end_time`` remain for
    backward compatibility (a single label window); the aligned per-sample
    vectors ``label_start_times`` / ``label_end_times`` describe one label
    window per sample and drive interval-based leakage validation.

    Aligned vector fields (all 1-D, one element per sample):

    - ``sample_ids``: unique identity per sample (duplicate-free).
    - ``decision_times``: the time at which each sample's decision/signal is
      made.
    - ``label_start_times``: the start (inclusive) of each sample's label
      window.
    - ``label_end_times``: the end (inclusive) of each sample's label window.
    - ``label_availability_times``: the time at which each sample's label
      becomes knowable/available.

    When the vector fields are present they are mutually aligned and must
    align with the plan masks.  If provided, the timestamp fields take
    precedence over ``label_horizon`` for temporal leakage checks.
    """

    label_start_time: Any = None
    label_end_time: Any = None
    label_horizon: int = 0  # Fallback if timestamps not provided
    sample_ids: Any = None
    decision_times: Any = None
    label_start_times: Any = None
    label_end_times: Any = None
    label_availability_times: Any = None

    def has_timestamps(self) -> bool:
        """True when aligned per-sample timestamp vectors are present."""
        return not (
            self.label_start_times is None
            and self.label_end_times is None
            and self.decision_times is None
            and self.label_availability_times is None
            and self.sample_ids is None
        )


def _freeze_execution_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            str(key): _freeze_execution_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_execution_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_execution_value(item) for item in value), key=repr))
    return value


def _plain_execution_value(value: Any) -> Any:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {str(key): _plain_execution_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_execution_value(item) for item in value]
    return value


@dataclass(frozen=True)
class SelectedExecutionSpec:
    """Deeply frozen mathematical object authorized for sealed evaluation."""

    trial_id: str
    mutation_id: str
    parent_factor_ids: Tuple[str, ...]
    selection_evaluation_ref: str
    sealed_split_hash: str
    metadata: Any
    spec_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("trial_id", "mutation_id", "selection_evaluation_ref", "sealed_split_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "parent_factor_ids", tuple(self.parent_factor_ids))
        object.__setattr__(self, "metadata", _freeze_execution_value(self.metadata))
        payload = self.to_dict(include_hash=False)
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        if self.spec_hash and self.spec_hash != actual:
            raise ValueError("SelectedExecutionSpec hash does not match its content")
        object.__setattr__(self, "spec_hash", actual)

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        result = {
            "trial_id": self.trial_id,
            "mutation_id": self.mutation_id,
            "parent_factor_ids": list(self.parent_factor_ids),
            "selection_evaluation_ref": self.selection_evaluation_ref,
            "sealed_split_hash": self.sealed_split_hash,
            "metadata": _plain_execution_value(self.metadata),
        }
        if include_hash:
            result["spec_hash"] = self.spec_hash
        return result

    def validate_certification_complete(self) -> None:
        execution = self.metadata.get("execution_spec") if hasattr(self.metadata, "get") else None
        required = {
            "canonical_recipe", "effective_parameters", "fit_state_ref",
            "data_snapshot_ref", "operator_versions", "code_version",
            "orientation", "model_input_ref",
            "cost_model_ref",
            "required_test_metrics",
        }
        if not hasattr(execution, "keys"):
            raise ValueError("sealed certification requires metadata.execution_spec")
        missing = required.difference(execution.keys())
        if missing:
            raise ValueError(
                "sealed execution specification missing required fields: "
                + ", ".join(sorted(missing))
            )
        recipe = execution["canonical_recipe"]
        if not (
            (isinstance(recipe, str) and recipe.strip())
            or (hasattr(recipe, "keys") and len(recipe) > 0)
        ):
            raise ValueError("canonical_recipe must be a non-empty canonical DSL/ref")
        if not hasattr(execution["effective_parameters"], "keys"):
            raise ValueError("effective_parameters must be a frozen mapping")
        for name in (
            "fit_state_ref", "data_snapshot_ref", "code_version",
            "model_input_ref", "cost_model_ref",
        ):
            value = execution[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty immutable reference")
        versions = execution["operator_versions"]
        if not hasattr(versions, "keys") or not versions:
            raise ValueError("operator_versions must be a non-empty frozen mapping")
        if execution["orientation"] not in (-1, 1):
            raise ValueError("orientation must be exactly -1 or 1")
        metrics = execution["required_test_metrics"]
        if (
            isinstance(metrics, (str, bytes))
            or not isinstance(metrics, tuple)
            or not metrics
            or any(not isinstance(name, str) or not name.strip() for name in metrics)
        ):
            raise ValueError("required_test_metrics must be a non-empty frozen sequence")

    @property
    def required_test_metrics(self) -> Tuple[str, ...]:
        self.validate_certification_complete()
        return tuple(self.metadata["execution_spec"]["required_test_metrics"])

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SelectedExecutionSpec":
        return cls(**dict(value))


@dataclass(frozen=True)
class SealedTestEvaluationOutcome:
    """Typed evidence returned by the certified sealed-test evaluator."""

    metrics: Mapping[str, float]
    execution_spec_hash: str
    dataset_identity: str
    split_id: str
    factor_identity: str
    time_identity: str
    cost_identity: str
    test_evidence_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise TypeError("metrics must be a non-empty mapping")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        for name in (
            "execution_spec_hash", "dataset_identity", "split_id",
            "factor_identity", "time_identity", "cost_identity",
            "test_evidence_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SealedTestHandle:
    """Immutable authority for one test-split evaluation of a frozen winner."""

    search_session_id: str
    trial_id: str
    split_id: str
    evaluation_ref: str
    frozen_at: datetime
    execution_spec_hash: str = ""

    def __post_init__(self):
        for name in (
            "search_session_id", "trial_id", "split_id", "evaluation_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frozen_at, datetime):
            raise TypeError("frozen_at must be a datetime")


@dataclass(frozen=True)
class SealedTestResult:
    """Result produced by consuming a sealed handle through ``SearchSession``."""

    trial_id: str
    test_metrics: Mapping[str, float]
    frozen_at: datetime
    search_session_id: str
    split_id: str
    evaluation_ref: str
    selection_evaluation_ref: str = ""
    execution_spec_hash: str = ""

    def __post_init__(self):
        if not isinstance(self.test_metrics, Mapping):
            raise TypeError("test_metrics must be a mapping")
        object.__setattr__(
            self, "test_metrics", MappingProxyType(dict(self.test_metrics))
        )
        for name in (
            "trial_id", "search_session_id", "split_id", "evaluation_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frozen_at, datetime):
            raise TypeError("frozen_at must be a datetime")


def validate_split_plan(split_plan: SplitPlan) -> Dict[str, Any]:
    """Validate non-empty, equal-length, pairwise-disjoint boolean masks."""
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    if not isinstance(split_plan.split_id, str) or not split_plan.split_id.strip():
        raise ValueError("split_id must be a non-empty string")
    if not isinstance(split_plan.metadata, dict):
        raise TypeError("metadata must be a dict")
    masks = (split_plan.train_mask, split_plan.validation_mask, split_plan.test_mask)
    lengths = []
    for name, mask in zip(("train", "validation", "test"), masks):
        if mask is None or not hasattr(mask, "__len__"):
            raise ValueError(f"{name}_mask must be a non-empty sized boundary")
        try:
            length = len(mask)
        except TypeError as exc:
            raise ValueError(f"{name}_mask must be a non-empty sized boundary") from exc
        if length == 0:
            raise ValueError(f"{name}_mask must not be empty")
        # Require exact built-in bool values; bool-like scalar types are not
        # valid mask elements at this contract boundary.
        if any(type(value) is not bool for value in mask):
            raise ValueError(f"{name}_mask must contain boolean boundaries")
        lengths.append(length)
    if len(set(lengths)) != 1:
        raise ValueError("split masks must have equal lengths")
    try:
        if any(
            (type(a) is bool and type(b) is bool and a and b)
            for i, left in enumerate(masks)
            for right in masks[i + 1:]
            for a, b in zip(left, right)
        ):
            raise ValueError("train, validation, and test masks must be disjoint")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "disjoint" in str(exc):
            raise
        raise ValueError("split masks must contain boolean boundaries") from exc
    _validate_temporal_leakage(split_plan, lengths[0])
    return {"split_id": split_plan.split_id, "n_samples": lengths[0], "validated": True}


def _coerce_timestamp_vector(values: Any, name: str, n: int) -> np.ndarray:
    """Coerce an aligned 1-D timing field to a UTC-normalized datetime64 array.

    Rejects (ValueError): missing/None, non-1D shapes, misaligned lengths,
    NaT/None elements, and timezone ambiguity (mixed tz-aware / tz-naive, or
    tz-aware values carrying a non-UTC offset that is not resolved).
    """
    if values is None:
        raise ValueError(f"LabelBundle.{name} is required when timestamps are present")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"LabelBundle.{name} must be a 1-D sequence, got a string")
    try:
        length = len(values)
    except TypeError as exc:
        raise ValueError(
            f"LabelBundle.{name} must be a sized 1-D sequence"
        ) from exc
    if length != n:
        raise ValueError(
            f"LabelBundle.{name} length {length} must match the mask length {n}"
        )
    try:
        import pandas as pd

        series = pd.to_datetime(list(values))
    except ImportError:  # pragma: no cover - pandas is a project dependency
        raise
    except ValueError as exc:
        message = str(exc)
        if "mix tz-aware" in message or "tz-aware" in message or "tz-naive" in message:
            raise ValueError(
                f"LabelBundle.{name} has a timezone ambiguity: mixed tz-aware and "
                f"tz-naive values are not allowed ({message})"
            ) from exc
        raise ValueError(
            f"LabelBundle.{name} must contain valid timestamps: {message}"
        ) from exc
    try:
        if isinstance(series, pd.DatetimeIndex):
            tz = series.tz
        else:
            tz = getattr(series.dt, "tz", None)
    except AttributeError:
        tz = None
    if tz is not None:
        try:
            arr = np.asarray(series.tz_convert("UTC").tz_localize(None))
        except Exception as exc:
            raise ValueError(
                f"LabelBundle.{name} must have an unambiguous UTC-resolvable "
                f"timezone (ambiguous tz or unknown offset: {exc})"
            ) from exc
    else:
        arr = np.asarray(series)
    if arr.dtype.kind not in ("M", "m"):
        raise ValueError(f"LabelBundle.{name} must be a datetime sequence")
    if np.isnat(arr).any():
        raise ValueError(f"LabelBundle.{name} must not contain NaT/None timestamps")
    return arr


def _validate_label_bundle_intervals(split_plan: SplitPlan, label_bundle: LabelBundle) -> None:
    """Vectorized timestamp-based interval leakage validation.

    The label window for a sample is ``[label_start_time, label_end_time]``.
    A sample's label window must not overlap any sample outside its own
    segment whose decision/availability time falls inside that window -- i.e.
    the window must be empty of "future" (train/validation) signal points and
    of test availability points.

    Rejects (ValueError, typed): NaT, timezone ambiguity, misaligned lengths,
    end < start, duplicate sample identity, unknown availability.
    """
    n_samples = len(split_plan.train_mask)
    bundle = label_bundle

    starts = _coerce_timestamp_vector(bundle.label_start_times, "label_start_times", n_samples)
    ends = _coerce_timestamp_vector(bundle.label_end_times, "label_end_times", n_samples)
    decisions = (
        _coerce_timestamp_vector(bundle.decision_times, "decision_times", n_samples)
        if bundle.decision_times is not None
        else None
    )
    availability = (
        _coerce_timestamp_vector(
            bundle.label_availability_times, "label_availability_times", n_samples
        )
        if bundle.label_availability_times is not None
        else None
    )
    if availability is None:
        if decisions is None:
            raise ValueError(
                "LabelBundle availability is unknown: supply label_availability_times "
                "or decision_times when timestamp vectors are present"
            )
        availability = decisions
    if decisions is None:
        decisions = starts.copy()
    if len(starts) != len(ends) or len(ends) != len(decisions) or len(decisions) != len(availability):
        raise ValueError(
            "LabelBundle timestamp vectors must be mutually aligned (equal length)"
        )

    # end < start is a malformed label window.
    if np.any(ends < starts):
        bad = int(np.flatnonzero(ends < starts)[0])
        raise ValueError(
            f"LabelBundle label window at sample {bad} has end < start "
            f"(start={starts[bad]!r}, end={ends[bad]!r})"
        )

    # Duplicate sample identity is ambiguous and rejected.
    sample_ids = bundle.sample_ids
    if sample_ids is not None:
        try:
            ids = list(sample_ids)
        except TypeError as exc:
            raise ValueError("LabelBundle.sample_ids must be a 1-D sequence") from exc
        if len(ids) != n_samples:
            raise ValueError(
                f"LabelBundle.sample_ids length {len(ids)} must match the mask length {n_samples}"
            )
        seen = set()
        dup = next((value for value in ids if value in seen or seen.add(value)), None)
        if dup is not None:
            raise ValueError(f"LabelBundle.sample_ids must be unique; duplicate {dup!r}")

    masks = (
        ("train", np.asarray(split_plan.train_mask, dtype=bool)),
        ("validation", np.asarray(split_plan.validation_mask, dtype=bool)),
        ("test", np.asarray(split_plan.test_mask, dtype=bool)),
    )

    train_avail = availability[masks[0][1]]
    val_avail = availability[masks[1][1]]

    train_starts = starts[masks[0][1]]
    train_ends = ends[masks[0][1]]
    val_starts = starts[masks[1][1]]
    val_ends = ends[masks[1][1]]
    test_starts = starts[masks[2][1]]
    test_ends = ends[masks[2][1]]

    def _interval_sets_overlap(left_starts, left_ends, right_starts, right_ends) -> bool:
        """Return whether two sets of closed label intervals intersect."""
        if left_starts.size == 0 or right_starts.size == 0:
            return False
        order = np.argsort(right_starts)
        ordered_starts = right_starts[order]
        prefix_max_end = np.maximum.accumulate(right_ends[order])
        positions = np.searchsorted(ordered_starts, left_ends, side="right") - 1
        eligible = positions >= 0
        if not np.any(eligible):
            return False
        return bool(np.any(prefix_max_end[positions[eligible]] >= left_starts[eligible]))

    # Purge is about the label intervals themselves.  A maturity timestamp is
    # an information boundary; it cannot stand in for an interval endpoint.
    n_train_val = _interval_sets_overlap(
        train_starts, train_ends, val_starts, val_ends
    )
    n_train_test = _interval_sets_overlap(
        train_starts, train_ends, test_starts, test_ends
    )
    n_val_test = _interval_sets_overlap(
        val_starts, val_ends, test_starts, test_ends
    )

    if n_train_val:
        raise ValueError(
            "LabelBundle leakage: a train label interval overlaps validation label intervals"
        )
    if n_train_test:
        raise ValueError(
            "LabelBundle leakage: a train label interval overlaps test label intervals"
        )
    if n_val_test:
        raise ValueError(
            "LabelBundle leakage: a validation label interval overlaps test label intervals"
        )

    protocol = str(split_plan.metadata.get("protocol", "")).strip().lower()
    if protocol in {"forward_validation", "walk_forward", "strict_forward"}:
        val_decisions = decisions[masks[1][1]]
        test_decisions = decisions[masks[2][1]]
        if train_avail.size and val_decisions.size:
            fit_cutoff = np.min(val_decisions)
            if np.any(train_avail > fit_cutoff):
                raise ValueError(
                    "LabelBundle leakage: a training label is not mature at the "
                    "forward-validation fit cutoff"
                )
        if val_avail.size and test_decisions.size:
            test_cutoff = np.min(test_decisions)
            if np.any(val_avail > test_cutoff):
                raise ValueError(
                    "LabelBundle leakage: a validation label is not mature at "
                    "the sealed-test decision boundary"
                )


def _validate_temporal_leakage(split_plan: SplitPlan, n_samples: int) -> None:
    """Fail-closed purge/embargo/label-horizon leakage check.

    Only meaningful when the plan carries a time axis: without one the
    masks are purely positional and the constraints are unexpressible,
    so requesting them without ``time_index`` is rejected outright.
    """
    label_horizon = split_plan.label_horizon
    purge = split_plan.purge
    embargo = split_plan.embargo
    validation_embargo = split_plan.validation_embargo
    label_bundle = split_plan.label_bundle

    # Validate integer fields
    for name, value in (("label_horizon", label_horizon), ("purge", purge), ("embargo", embargo), ("validation_embargo", validation_embargo)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    # If no temporal constraints requested, skip -- UNLESS a LabelBundle with
    # real timestamps is present.  The bundle's label windows carry their own
    # temporal semantics and MUST be validated even when every plan-level
    # integer field is zero (FO-P0-01 regression: the previous code
    # early-returned before consulting the bundle, silently admitting
    # forward-label leakage into the test/validation segments).
    if (
        label_horizon == 0
        and purge == 0
        and embargo == 0
        and validation_embargo == 0
        and (label_bundle is None or not label_bundle.has_timestamps())
    ):
        return

    # Timestamp intervals and explicit positional purge/embargo are cumulative
    # constraints.  Real timestamps replace only the label-horizon fallback.
    has_timestamp_bundle = label_bundle is not None and label_bundle.has_timestamps()
    if has_timestamp_bundle:
        _validate_label_bundle_intervals(split_plan, label_bundle)

    time_index = split_plan.time_index
    positional_requested = purge or embargo or validation_embargo
    if not positional_requested and has_timestamp_bundle:
        return
    if time_index is None:
        raise ValueError(
            "purge/embargo/label_horizon/validation_embargo require a time_index; purely "
            "positional masks cannot express temporal leakage constraints"
        )
    if (
        not hasattr(time_index, "__len__")
        or isinstance(time_index, (str, bytes))
        or len(time_index) != n_samples
    ):
        raise ValueError(
            "time_index must be a sequence aligned with the masks "
            f"({n_samples} positions)"
        )
    # Map each mask position onto its rank on the time axis.
    try:
        order = sorted(range(n_samples), key=lambda pos: time_index[pos])
    except TypeError as exc:
        raise ValueError("time_index values must be mutually comparable") from exc
    rank_of = {pos: rank for rank, pos in enumerate(order)}

    train_positions = [
        i for i, flag in enumerate(split_plan.train_mask) if flag
    ]
    validation_positions = [
        i for i, flag in enumerate(split_plan.validation_mask) if flag
    ]
    test_positions = [i for i, flag in enumerate(split_plan.test_mask) if flag]

    # Determine effective label_horizon: use label_bundle if provided
    effective_label_horizon = label_horizon
    if label_bundle is not None and not has_timestamp_bundle:
        # LabelBundle provides real timestamps; use its label_horizon as fallback
        effective_label_horizon = label_bundle.label_horizon

    # (a) Train -> Test leakage: A test position whose label window reaches back into a
    # train position's forward-label span is leakage: the
    # train label at position p spans positions
    # p..p+label_horizon, so any test position within
    # purge + label_horizon positions AFTER a train position
    # (on the time axis) is rejected.
    lookback = purge + effective_label_horizon
    if lookback > 0:
        train_ranks = sorted(rank_of[p] for p in train_positions)
        for test_pos in test_positions:
            test_rank = rank_of[test_pos]
            for train_rank in train_ranks:
                # Use < instead of <= to avoid false positive when test position
                # is exactly at the boundary (purge should be enough gap)
                if train_rank < test_rank < train_rank + lookback:
                    raise ValueError(
                        "test position falls within purge + label_horizon "
                        f"({lookback}) positions after a train "
                        "position: forward-label overlap into the test segment"
                    )

    # (b) Train -> Validation leakage: A validation position whose label window reaches back into a
    # train position's forward-label span is leakage. This is the missing validation
    # of Train label interval ∩ Validation period.
    validation_lookback = validation_embargo + effective_label_horizon
    if validation_lookback > 0:
        train_ranks = sorted(rank_of[p] for p in train_positions)
        for val_pos in validation_positions:
            val_rank = rank_of[val_pos]
            for train_rank in train_ranks:
                if train_rank < val_rank <= train_rank + validation_lookback:
                    raise ValueError(
                        "validation position falls within validation_embargo + label_horizon "
                        f"({validation_lookback}) positions after a train "
                        "position: forward-label overlap into the validation segment"
                    )

    # (c) Validation -> Test leakage: A test position whose label window reaches back into a
    # validation position's forward-label span is leakage. This is the missing validation
    # of Validation label interval ∩ Test period.
    val_test_lookback = purge + effective_label_horizon
    if val_test_lookback > 0:
        val_ranks = sorted(rank_of[p] for p in validation_positions)
        for test_pos in test_positions:
            test_rank = rank_of[test_pos]
            for val_rank in val_ranks:
                if val_rank < test_rank <= val_rank + val_test_lookback:
                    raise ValueError(
                        "test position falls within purge + label_horizon "
                        f"({val_test_lookback}) positions after a validation "
                        "position: forward-label overlap into the test segment"
                    )

    # (d) Embargo: train positions within `embargo` positions AFTER a
    # test position are rejected.
    if embargo > 0:
        test_ranks = sorted(rank_of[p] for p in test_positions)
        for train_pos in train_positions:
            train_rank = rank_of[train_pos]
            for test_rank in test_ranks:
                if test_rank < train_rank <= test_rank + embargo:
                    raise ValueError(
                        "train position falls within embargo "
                        f"({embargo}) positions after a test position"
                    )


def create_split_aware_evaluation_fn(qe_adapter: Any, split_plan: SplitPlan) -> EvaluationProtocol:
    """Build the safe boundary around an adapter's evaluator."""
    if not callable(getattr(qe_adapter, "evaluate", None)):
        raise TypeError("qe_adapter must expose a callable evaluate method")
    return EvaluationProtocol(split_plan, qe_adapter.evaluate)


__all__ = [
    "SplitType", "SplitPlan", "LabelBundle", "EvaluationProtocol", "SearchEvaluationResult",
    "SelectedExecutionSpec",
    "SealedTestHandle", "SealedTestResult", "validate_split_plan",
    "create_split_aware_evaluation_fn",
]
