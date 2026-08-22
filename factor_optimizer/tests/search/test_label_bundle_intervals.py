"""LabelBundle timestamp-based interval leakage validation tests.

Covers FO-P0-01: the FO ``validate_split_plan`` early-returned when every
plan-level integer field (label_horizon/purge/embargo/validation_embargo) was
zero, BEFORE consulting the LabelBundle.  A bundle carrying real timestamps
therefore could not express forward-label leakage -- train/validation label
intervals silently overlapping test (or validation) availability passed
validation.  This file pins the closed gap:

- aligned LabelBundle timestamp vectors take precedence over the integer
  horizon fallback;
- timestamp interval intersections are validated vectorized (numpy), never
  O(N^2);
- a bundle with real timestamps is NEVER skipped by the plan-level zero
  early-return;
- malformed vectors (NaT, tz ambiguity, misaligned lengths, end < start,
  duplicate sample ids, unknown availability) are rejected with typed
  ValueError.
"""

import time

import numpy as np
import pandas as pd
import pytest

from factor_optimizer.contracts.splits import (
    LabelBundle,
    SplitPlan,
    validate_split_plan,
)

T0 = np.datetime64("2026-01-01", "ns")


def _plan(train, validation, test, **kwargs):
    return SplitPlan("split", train, validation, test, {}, **kwargs)


def _bundle(starts, ends, decisions=None, availability=None, sample_ids=None):
    kwargs = dict(
        label_start_times=list(starts),
        label_end_times=list(ends),
        decision_times=list(decisions) if decisions is not None else None,
        label_availability_times=(
            list(availability) if availability is not None else None
        ),
        sample_ids=list(sample_ids) if sample_ids is not None else None,
    )
    return LabelBundle(**kwargs)


def _dt(day, hour=0):
    return np.datetime64(f"2026-01-{day:02d}T{hour:02d}:00:00", "ns")


def _base_masks(n=8, train=(0, 1, 2, 3), val=(4, 5), test=(6, 7)):
    masks = [False] * n
    train_m = [i in train for i in range(n)]
    val_m = [i in val for i in range(n)]
    test_m = [i in test for i in range(n)]
    return train_m, val_m, test_m


# ---------------------------------------------------------------------------
# Overlap rejection
# ---------------------------------------------------------------------------

def test_train_label_interval_overlapping_validation_availability_rejected():
    # Train samples 0..3 decide on day 1; their labels reach day 6.  Validation
    # samples 4..5 decide on day 5 -- inside the train label interval.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends = [_dt(6)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="train label interval overlaps validation"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_train_label_interval_overlapping_test_availability_rejected():
    # Train decides day 1; its label window [1,8] contains the test
    # availability day 7.  Validation availability (day 9) is OUTSIDE the
    # train window, so this isolates the train-vs-TEST rejection.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    ends = [_dt(8)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="train label interval overlaps test"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_validation_label_interval_overlapping_test_availability_rejected():
    # Validation labels (decide day 5) reach day 7; test availability at day 7
    # falls inside the validation label interval.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(7)] * 2 + [_dt(7)] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="validation label interval overlaps test"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_clean_non_overlapping_timestamps_accepted():
    # Train decides day 1, labels [1,2]; validation day 4 labels [4,5]; test
    # day 7 labels [7,8].  No interval crosses any other segment's availability.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(4)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(8)] * 2
    decisions = starts[:]
    result = validate_split_plan(
        _plan(
            train_m, val_m, test_m,
            label_bundle=_bundle(starts, ends, decisions=decisions),
        )
    )
    assert result["validated"] is True
    assert result["n_samples"] == n


def test_zero_plan_level_fields_with_overlapping_timestamps_still_rejected():
    # THE P0 regression: every plan-level integer field is zero, but the
    # bundle's real timestamps overlap.  The old early-return skipped this
    # entirely; it must now be rejected.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    ends = [_dt(8)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2  # train labels reach day 8
    decisions = starts[:]
    with pytest.raises(ValueError, match="train label interval overlaps test"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_zero_plan_level_fields_train_interval_overlaps_test_availability():
    # The exact P0 vector, isolated: validation availability (day 9) is outside
    # the train label window [1,8]; only the test availability (day 7) lies
    # inside it.  With every plan-level integer zero, the OLD code skipped this
    # entirely; the timestamp interval rule must now reject it.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    ends = [_dt(8)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="train label interval overlaps test"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_timestamps_take_precedence_over_integer_horizon():
    # label_horizon is ignored once the bundle carries real timestamps: the
    # integer arithmetic on the same masks would be legal, but the timestamp
    # intervals overlap and must be rejected.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    ends = [_dt(8)] * 4 + [_dt(9)] * 2 + [_dt(7)] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="train label interval overlaps test"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_horizon=0,
                purge=0,
                embargo=0,
                validation_embargo=0,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


# ---------------------------------------------------------------------------
# Malformed vector rejection (typed ValueError)
# ---------------------------------------------------------------------------

def test_nat_timestamp_rejected():
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    starts[0] = pd.NaT
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    decisions = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    with pytest.raises(ValueError, match="NaT"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_misaligned_lengths_rejected():
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2  # one short
    decisions = starts[:]
    with pytest.raises(ValueError, match="length"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends[:-1], decisions=decisions),
            )
        )


def test_end_before_start_rejected():
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends[0] = starts[0] - np.timedelta64(1, "h")  # end < start for sample 0
    decisions = starts[:]
    with pytest.raises(ValueError, match="end < start"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_timezone_ambiguity_rejected():
    # Mixed tz-aware and tz-naive timestamps are ambiguous.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [pd.Timestamp("2026-01-01", tz="Asia/Shanghai")] * 4
    starts += [pd.Timestamp("2026-01-05")] * 2  # naive
    starts += [pd.Timestamp("2026-01-07")] * 2  # naive
    ends = [pd.Timestamp("2026-01-02", tz="Asia/Shanghai")] * 4
    ends += [pd.Timestamp("2026-01-05")] * 2
    ends += [pd.Timestamp("2026-01-07")] * 2
    decisions = starts[:]
    with pytest.raises(ValueError, match="tz|timezone|unambiguous"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions),
            )
        )


def test_duplicate_sample_ids_rejected():
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(4)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(8)] * 2
    decisions = starts[:]
    sample_ids = list(range(n))
    sample_ids[3] = sample_ids[0]  # duplicate
    with pytest.raises(ValueError, match="unique"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends, decisions=decisions, sample_ids=sample_ids),
            )
        )


def test_unknown_availability_rejected():
    # An availability time that is not an actual decision time is
    # "unknown".  With no decision_times and no label_availability_times
    # supplied, the bundle cannot establish when any label becomes knowable
    # -- the strict validation fails closed.
    n = 8
    train_m, val_m, test_m = _base_masks(n)
    starts = [_dt(1)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    ends = [_dt(2)] * 4 + [_dt(5)] * 2 + [_dt(7)] * 2
    with pytest.raises(ValueError, match="availability is unknown"):
        validate_split_plan(
            _plan(
                train_m, val_m, test_m,
                label_bundle=_bundle(starts, ends),
            )
        )


# ---------------------------------------------------------------------------
# Vectorized / large-scale
# ---------------------------------------------------------------------------

def test_large_vectorized_case_completes_quickly():
    # 10k samples: train 0..3999, validation 4000..6999, test 7000..9999.
    # Clean non-overlapping windows: decide day 1..10000 hours; each label
    # window is [t, t+1h] which never reaches the next segment's earliest
    # availability.
    n = 10_000
    train_m = [i < 4000 for i in range(n)]
    val_m = [4000 <= i < 7000 for i in range(n)]
    test_m = [i >= 7000 for i in range(n)]
    starts = [np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(i, "h") for i in range(n)]
    ends = [np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(i, "h") + np.timedelta64(30, "m") for i in range(n)]
    decisions = starts[:]
    begin = time.monotonic()
    result = validate_split_plan(
        _plan(
            train_m, val_m, test_m,
            label_bundle=_bundle(starts, ends, decisions=decisions),
        )
    )
    elapsed = time.monotonic() - begin
    assert result["validated"] is True
    # O(N log N) searchsorted: 10k samples must complete far under 5s.
    assert elapsed < 5.0, f"vectorized validation took {elapsed:.2f}s -- O(N^2)?"