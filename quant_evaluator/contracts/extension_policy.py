"""Frozen resampling policies and primitive interface contracts (plan §8.3).

This module defines the *policy* layer of the extension primitives:

- :class:`ResamplingPolicy` — frozen moving-block policy; every field enters
  the metric-instance hash (plan §6.1 "新指标默认参数").
- :class:`ResamplingPlan` — the frozen (starts, lengths) plan generated ONCE
  by NumPy PCG64 and shared by CPU/GPU (plan §6.3).  The plan is independent
  of factor ids, factor count, tile width and thread count; changing
  ``output_length``, the source time interval or ``seed`` necessarily changes
  the plan hash.
- :class:`SharedSortArtifact` — one-sort-many-consumers artifact (plan §6.2).
- Small typed result wrappers so business code never unpacks "the 3rd result"
  positionally (plan §8.3).

NOTE: an unrelated legacy :class:`quant_evaluator.contracts.resampling.ResamplingPlan`
(original-clock policy plan, no numeric starts) already exists.  This plan is
the plan §6.3 numeric (starts[B,J], lengths[J]) contract used by the extension
families; import them with care.

The primitive *function* signatures (``make_resampling_plan``, ``prefix_moments``,
``sample_moments``, ``hac_lrv``, ``path_block_summaries``, ``sample_path_mdd``,
``upper_tail_mean``, ``shared_sort``) are implemented in
``quant_evaluator.metrics.extension.primitives`` (CPU reference) with the exact
shapes fixed by plan §8.3:  ``workspace_bytes`` only decides tiling and never
changes formulas, samples or the RNG sequence; returned tuples use the typed
wrappers below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Tuple

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError

__all__ = [
    "ResamplingPolicy",
    "ResamplingPlan",
    "SharedSortArtifact",
    "PrefixMoments",
    "MomentResult",
    "PathSummaryBlock",
    "PathMddResult",
    "TailMeanResult",
    "HacLrvResult",
    "SORT_PRODUCTS",
    "default_hac_max_lag",
]

_SORT_PRODUCTS = ("rank", "distinct_count", "quantile")


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or int(value) < minimum:
        raise InvalidContractError(
            f"{name} must be an integer >= {minimum} (bool rejected), got {value!r}"
        )
    return int(value)


@dataclass(frozen=True)
class ResamplingPolicy:
    """Frozen moving-block bootstrap policy (plan §6.3 / §8.3)."""

    block_length: int
    repetitions: int
    confidence_level: float
    seed: int
    sample_length: int
    output_length: int
    method: str = "moving_block"
    rng: str = "numpy.PCG64"

    def __post_init__(self) -> None:
        _integer(self.block_length, "block_length", 1)
        _integer(self.repetitions, "repetitions", 2)
        _integer(self.seed, "seed", 0)
        _integer(self.sample_length, "sample_length", 1)
        _integer(self.output_length, "output_length", 1)
        if self.sample_length < self.block_length:
            raise InvalidContractError(
                f"sample_length ({self.sample_length}) must be >= block_length ({self.block_length})"
            )
        if isinstance(self.confidence_level, (bool, np.bool_)) \
                or not isinstance(self.confidence_level, (int, float, np.integer, np.floating)) \
                or not np.isfinite(self.confidence_level) \
                or not 0.0 < float(self.confidence_level) < 1.0:
            raise InvalidContractError(
                "confidence_level must be finite and strictly between zero and one"
            )
        if self.method != "moving_block":
            raise InvalidContractError(f"unsupported resampling method {self.method!r}")
        if self.rng != "numpy.PCG64":
            raise InvalidContractError(f"unsupported rng {self.rng!r}")

    @property
    def policy_hash(self) -> str:
        return stable_content_hex(
            tag="ResamplingPolicy.v1",
            fields={
                "method": self.method,
                "rng": self.rng,
                "block_length": int(self.block_length),
                "repetitions": int(self.repetitions),
                "confidence_level": float(self.confidence_level),
                "seed": int(self.seed),
                "sample_length": int(self.sample_length),
                "output_length": int(self.output_length),
            },
        )


@dataclass(frozen=True)
class ResamplingPlan:
    """Frozen numeric block plan: starts (B,J) + lengths (J,), generated once.

    Plan §6.3: ``J = ceil(output_length / block_length)``; the trailing short
    block keeps its start drawn from the FULL-block start set (identical to
    "draw complete blocks then truncate").  Plans are consumed by the whole
    factor family; workers never re-seed.
    """

    starts: np.ndarray            # (B,J) int64, uniform integers in [0, sample_length-block_length]
    lengths: np.ndarray           # (J,) int64
    source_time_ref: str
    policy: ResamplingPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.source_time_ref, str) or not self.source_time_ref.strip():
            raise InvalidContractError("source_time_ref must be a non-empty string")
        if not isinstance(self.policy, ResamplingPolicy):
            raise InvalidContractError("policy must be a ResamplingPolicy")
        starts = self.starts
        lengths = self.lengths
        if not isinstance(starts, np.ndarray) or starts.dtype.kind not in "iu" or starts.ndim != 2:
            raise InvalidContractError("starts must be a rank-2 integer ndarray (B,J)")
        if not isinstance(lengths, np.ndarray) or lengths.dtype.kind not in "iu" or lengths.ndim != 1:
            raise InvalidContractError("lengths must be a rank-1 integer ndarray (J,)")
        policy = self.policy
        block = int(policy.block_length)
        j_expected = -(-int(policy.output_length) // block)
        if starts.shape != (int(policy.repetitions), j_expected):
            raise InvalidContractError(
                f"starts shape {starts.shape} must be "
                f"(B,J)=({int(policy.repetitions)},{j_expected})"
            )
        if lengths.shape != (j_expected,):
            raise InvalidContractError(f"lengths shape {lengths.shape} must be (J,)=({j_expected},)")
        expected_lengths = np.full(j_expected, block, dtype=np.int64)
        expected_lengths[-1] = int(policy.output_length) - (j_expected - 1) * block
        if not np.array_equal(np.asarray(lengths, dtype=np.int64), expected_lengths):
            raise InvalidContractError("lengths must be [block_length, ..., output_length-(J-1)*block]")
        upper = int(policy.sample_length) - block
        if np.any(np.asarray(starts) < 0) or np.any(np.asarray(starts) > upper):
            raise InvalidContractError(
                f"starts must lie in [0, sample_length-block_length]={([0, upper])}"
            )

    @property
    def policy_hash(self) -> str:
        return self.policy.policy_hash

    @property
    def plan_hash(self) -> str:
        return stable_content_hex(
            tag="ResamplingPlan.v1",
            fields={
                "policy_hash": self.policy_hash,
                "source_time_ref": self.source_time_ref,
                "starts": np.asarray(self.starts, dtype=np.int64),
                "lengths": np.asarray(self.lengths, dtype=np.int64),
            },
        )


@dataclass(frozen=True)
class SharedSortArtifact:
    """One-sort-many-consumers artifact (plan §6.2).

    ``R = T * F_tile`` rows of cross-sections of width N.  Invalid entries are
    sorted to the tail behind a +inf sentinel and never share runs with legal
    finite values.  For a finite-value run ``[a, b)`` the 1-based average rank
    is ``(a + b + 1) / 2``.  ``run_start``/``run_end_exclusive`` are produced
    only when a consumer needs them; summary-only flows keep no full host
    ranking.
    """

    input_ref: str
    axis_ref: str
    mask_ref: str
    orientation_ref: str
    dtype: str
    layout: str
    order: np.ndarray                 # (R,N) integer sort permutation
    sorted_values: np.ndarray         # (R,N) values in sorted order (input dtype)
    finite_count: np.ndarray          # (R,) valid (masked+finite) counts
    requested_products: Tuple[str, ...]
    run_start: np.ndarray | None      # (R,N) optional
    run_end_exclusive: np.ndarray | None  # (R,N) optional
    workspace_bytes: int
    device_id: str

    def __post_init__(self) -> None:
        order = self.order
        values = self.sorted_values
        finite = self.finite_count
        if not isinstance(order, np.ndarray) or order.dtype.kind not in "iu" or order.ndim != 2:
            raise InvalidContractError("order must be a rank-2 integer ndarray (R,N)")
        r, n = order.shape
        if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape != (r, n):
            raise InvalidContractError(f"sorted_values shape must be (R,N)=({r},{n})")
        if not isinstance(finite, np.ndarray) or finite.dtype.kind not in "iu" or finite.shape != (r,):
            raise InvalidContractError(f"finite_count shape must be (R,)=({r},)")
        if finite.max(initial=0) > n:
            raise InvalidContractError("finite_count cannot exceed the cross-section width")
        if (self.run_start is None) != (self.run_end_exclusive is None):
            raise InvalidContractError("run_start and run_end_exclusive must be provided together")
        if self.run_start is not None:
            for name, array in (("run_start", self.run_start), ("run_end_exclusive", self.run_end_exclusive)):
                if not isinstance(array, np.ndarray) or array.dtype.kind not in "iu" or array.shape != (r, n):
                    raise InvalidContractError(f"{name} shape must be (R,N)=({r},{n})")
        if not self.requested_products:
            raise InvalidContractError("requested_products must be a non-empty tuple")
        for product in self.requested_products:
            if product not in _SORT_PRODUCTS:
                raise InvalidContractError(
                    f"unknown requested product {product!r}; allowed: {_SORT_PRODUCTS}"
                )
        _integer(self.workspace_bytes, "workspace_bytes", 0)


class PrefixMoments(NamedTuple):
    """prefix_moments result: center (F,), p1/p2 (T+1,F) with a leading zero row."""

    center: np.ndarray
    p1: np.ndarray
    p2: np.ndarray


class MomentResult(NamedTuple):
    """sample_moments result: per-bootstrap mean/var plus a per-column valid mask."""

    mean: np.ndarray   # (B,F) FP64
    var: np.ndarray    # (B,F) FP64, ddof=1 sample variance
    valid: np.ndarray  # (B,F) bool; False marks a numerically unstable column


class PathSummaryBlock(NamedTuple):
    """path_block_summaries result: (start, length_slot, F, 4) lookup + per-column mask.

    The four trailing components are the ordered log-path summary
    ``(total, high, low, log_drawdown)`` of plan §6.5; ``length_slot`` indexes
    the sorted unique block lengths.
    """

    summary_lookup: np.ndarray  # (S, L, F, 4) FP64; unreachable cells are NaN
    length_slots: np.ndarray    # (L,) sorted unique block lengths
    valid: np.ndarray           # (F,) bool


class PathMddResult(NamedTuple):
    """sample_path_mdd result: percentage max drawdown per (bootstrap, factor)."""

    mdd: np.ndarray    # (B,F) FP64, = -expm1(-log_drawdown)
    valid: np.ndarray  # (F,) bool


class TailMeanResult(NamedTuple):
    """upper_tail_mean result: value, demanded tail mass and validity per column."""

    value: np.ndarray  # (F,) FP64
    mass: np.ndarray   # (F,) FP64, demanded tail mass m
    valid: np.ndarray  # (F,) bool


class HacLrvResult(NamedTuple):
    """hac_lrv result: long-run variance Ω per column plus validity."""

    omega2: np.ndarray  # (F,) FP64, Ω = γ0 + 2 Σ (1-l/(L+1)) γl  (plan §6.7)
    valid: np.ndarray   # (F,) bool


def default_hac_max_lag(n_time: int, declared_min_lag: int | None = None) -> int:
    """``nw_rule_v1`` automatic HAC bandwidth (plan §6.1): ``floor(4*(T/100)^(2/9))``.

    Not tuned on the significance of the current returns.  When the caller
    declares a minimum lag (e.g. overlapping labels with horizon H declaring
    ``H-1``), the rule never returns less than it.
    """
    t = _integer(n_time, "n_time", 1)
    rule = int(4.0 * (t / 100.0) ** (2.0 / 9.0))
    if declared_min_lag is None:
        return rule
    declared = _integer(declared_min_lag, "declared_min_lag", 0)
    return max(rule, declared)
