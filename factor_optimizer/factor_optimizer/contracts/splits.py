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
from typing import Any, Callable, Dict, Optional, Tuple, List

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


@dataclass(frozen=True)
class SealedTestHandle:
    """Immutable authority for one test-split evaluation of a frozen winner."""

    search_session_id: str
    trial_id: str
    split_id: str
    evaluation_ref: str
    frozen_at: datetime

    def __post_init__(self):
        for name in ("search_session_id", "trial_id", "split_id", "evaluation_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frozen_at, datetime):
            raise TypeError("frozen_at must be a datetime")


@dataclass(frozen=True)
class SealedTestResult:
    """Result produced by consuming a sealed handle through ``SearchSession``."""

    trial_id: str
    test_metrics: Dict[str, float]
    frozen_at: datetime
    search_session_id: str
    split_id: str
    evaluation_ref: str

    def __post_init__(self):
        if not isinstance(self.test_metrics, dict):
            raise TypeError("test_metrics must be a dict")
        for name in ("trial_id", "search_session_id", "split_id", "evaluation_ref"):
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

    # Classify each sample by its segment.  A sample's own availability point
    # is a legal boundary, so each segment's availability set excludes its own
    # label windows.
    train_avail = availability[masks[0][1]]
    val_avail = availability[masks[1][1]]
    test_avail = availability[masks[2][1]]

    # Compute the union of other-segment availability points that must not
    # fall inside any label window of this segment.  Using a sorted union
    # plus binary search keeps this vectorized (O(N log N)), never O(N^2).
    def _count_foreign_inside(starts_arr, ends_arr, foreign: np.ndarray) -> int:
        if foreign.size == 0:
            return 0
        foreign = np.sort(foreign)
        left = np.searchsorted(foreign, starts_arr, side="left")
        right = np.searchsorted(foreign, ends_arr, side="right") - 1
        hits = right >= left
        return int(hits.sum())

    train_starts = starts[masks[0][1]]
    train_ends = ends[masks[0][1]]
    val_starts = starts[masks[1][1]]
    val_ends = ends[masks[1][1]]
    test_starts = starts[masks[2][1]]
    test_ends = ends[masks[2][1]]

    # Train label interval ∩ Validation availability must be empty.
    n_train_val = _count_foreign_inside(train_starts, train_ends, val_avail)
    # Train label interval ∩ Test availability must be empty.
    n_train_test = _count_foreign_inside(train_starts, train_ends, test_avail)
    # Validation label interval ∩ Test availability must be empty.
    n_val_test = _count_foreign_inside(val_starts, val_ends, test_avail)

    if n_train_val:
        raise ValueError(
            "LabelBundle leakage: a train label interval overlaps validation "
            "availability (train labels leak into the validation segment)"
        )
    if n_train_test:
        raise ValueError(
            "LabelBundle leakage: a train label interval overlaps test "
            "availability (train labels leak into the test segment)"
        )
    if n_val_test:
        raise ValueError(
            "LabelBundle leakage: a validation label interval overlaps test "
            "availability (validation labels leak into the test segment)"
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

    # Timestamp-based interval validation takes PRECEDENCE over the integer
    # horizon fallback whenever the bundle carries real timestamps.
    if label_bundle is not None and label_bundle.has_timestamps():
        _validate_label_bundle_intervals(split_plan, label_bundle)
        return

    time_index = split_plan.time_index
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
    if label_bundle is not None:
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
    "SealedTestHandle", "SealedTestResult", "validate_split_plan",
    "create_split_aware_evaluation_fn",
]
