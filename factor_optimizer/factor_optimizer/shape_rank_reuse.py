"""Invocation-local reuse of FE's average-rank feature for shape repairs."""
from __future__ import annotations

import hashlib
import math
import numbers

import numpy as np
import pandas as pd


RANK_CONTRACT = "fe.cs_rank.average.percentile.v1"
DEFAULT_MAX_BYTES = 128 * 1024 * 1024
MIN_REUSE_ROWS_BY_PLAN_COUNT = ((500_000, 14), (1_000_000, 6))


def should_admit_u_shape_rank_reuse(actual_train_rows: int,
                                    distinct_eligible_plan_count: int) -> bool:
    """Evidence-based TRAIN admission; all other cases stay on FE execution."""
    if (type(actual_train_rows) is not int or actual_train_rows < 0
            or type(distinct_eligible_plan_count) is not int
            or distinct_eligible_plan_count < 0):
        raise ValueError("TRAIN rows and eligible plan count must be nonnegative integers")
    return any(actual_train_rows >= rows and distinct_eligible_plan_count >= plans
               for rows, plans in MIN_REUSE_ROWS_BY_PLAN_COUNT)


def count_distinct_eligible_u_shape_plans(plans) -> int:
    """Count distinct eligible identities, ignoring wrappers and duplicates."""
    identities = set()
    for plan in plans:
        if not is_eligible_u_shape_plan(plan):
            continue
        try:
            identities.add(plan.identity)
        except (TypeError, ValueError, OverflowError):
            continue
    return len(identities)


def is_eligible_u_shape_plan(plan) -> bool:
    """Whether this exact, unwrapped ValueRepairPlan supports the shortcut."""
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    if (type(plan) is not ValueRepairPlan or type(plan.family) is not str
            or plan.family not in {"U_SHAPE_REPAIR", "INVERTED_U_REPAIR"}
            or plan.transform != "rank_shape"):
        return False
    try:
        params = dict(plan.parameters)
        if set(params) != {"center", "power", "asymmetric", "inverted"}:
            return False
        center, power = params["center"], params["power"]
        return bool(
            isinstance(center, numbers.Real) and not isinstance(center, (bool, np.bool_))
            and math.isfinite(float(center)) and 0.0 <= float(center) <= 1.0
            and isinstance(power, numbers.Real) and not isinstance(power, (bool, np.bool_))
            and math.isfinite(float(power)) and float(power) > 0.0
            and type(params["asymmetric"]) is bool
            and type(params["inverted"]) is bool
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _frame_key(frame: pd.DataFrame, *, training_context_ref: str,
               rank_contract: str = RANK_CONTRACT) -> str | None:
    """Fingerprint the complete rank input, preserving order and caller index."""
    try:
        relevant = frame[["date", "asset_id", "value"]]
        h = hashlib.sha256()
        h.update(repr((id(frame), type(frame.index), frame.index.names,
                       tuple((name, str(dtype)) for name, dtype in relevant.dtypes.items()),
                       training_context_ref, rank_contract)).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(relevant, index=True).to_numpy(dtype=np.uint64).tobytes())
        values = relevant["value"].to_numpy(dtype=np.float64, copy=True)
        h.update(np.isfinite(values).tobytes())
        h.update(values.tobytes())
        return h.hexdigest()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


class RankFeatureCache:
    """One-entry bounded cache scoped by the caller to one TRAIN invocation."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self._key = None
        self._rank = None
        self.hits = 0
        self.misses = 0
        self.rank_calls = 0
        self.bypasses = 0
        self.evictions = 0

    @property
    def retained_bytes(self) -> int:
        return 0 if self._rank is None else int(self._rank.nbytes)

    def _bypass(self) -> None:
        self.bypasses += 1
        if self._rank is not None:
            self.evictions += 1
        self._key = None
        self._rank = None

    def average_rank(self, frame: pd.DataFrame, *,
                     training_context_ref: str) -> pd.Series | None:
        # A disabled/over-budget cache must not fingerprint or rank, including
        # the empty-frame case where the retained-size arithmetic is zero.
        if self.max_bytes == 0 or len(frame) * np.dtype(np.float64).itemsize > self.max_bytes:
            self.misses += 1
            self._bypass()
            return None
        key = _frame_key(frame, training_context_ref=training_context_ref)
        if key is None:
            self.misses += 1
            self._bypass()
            return None
        if key == self._key and self._rank is not None:
            self.hits += 1
            return pd.Series(self._rank, index=frame.index, name="value", copy=False)

        self.misses += 1
        self.rank_calls += 1
        from factor_optimizer.adapters.repair_execution import _execute_fe_cs_rank
        computed = _execute_fe_cs_rank(frame)
        rank = computed.to_numpy(dtype=np.float64, copy=True)
        if rank.nbytes > self.max_bytes:
            self._bypass()
            return None
        if self._rank is not None:
            self.evictions += 1
        # A bytes-owned ndarray cannot be made writable again via
        # Series.to_numpy().setflags(write=True).
        self._rank = np.frombuffer(rank.tobytes(), dtype=np.float64)
        self._key = key
        return pd.Series(self._rank, index=frame.index, name="value", copy=False)


def apply_u_shape_from_rank(plan, frame: pd.DataFrame,
                            cache: RankFeatureCache, *,
                            allow_research: bool = False) -> pd.Series | None:
    """Return the exact rank-shape formula for an unwrapped U-family plan.

    A None result means the caller should use the plan's ordinary execute path.
    Baseline/wrapper and malformed plans fail closed to that path.
    """
    from factor_optimizer.adapters.repair_execution import _validate_frame

    if allow_research is not True:
        raise ValueError("this adapter is research-only; explicit allow_research=True required")
    if not is_eligible_u_shape_plan(plan):
        return None
    frame = _validate_frame(frame)
    params = dict(plan.parameters)
    center, power = params["center"], params["power"]
    rank = cache.average_rank(frame, training_context_ref=plan.training_context_ref)
    if rank is None:
        return None
    distance = (rank - center).abs()
    if params["asymmetric"]:
        left = rank < center
        if center > 0:
            distance.loc[left] = distance.loc[left] / center
        if center < 1:
            distance.loc[~left] = distance.loc[~left] / (1.0 - center)
    shaped = distance.pow(power)
    if params["inverted"]:
        shaped = -shaped
    return shaped.rename("value")
