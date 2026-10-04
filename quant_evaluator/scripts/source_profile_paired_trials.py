"""Pure helpers for bounded randomized, paired source API trials.

This module only constructs a counterbalanced run order and aggregates caller-
validated end-to-end CPU/CUDA receipts. It does not execute workloads, build
qualification records, or authorize automatic routing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
import re

TIMING_SCOPE = "evaluate_factor_source_batch_wall_v1"
MAX_PAIRED_TRIALS = 16
_CONTEXT_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_randomized_paired_schedule(
    *, seed: int, pairs: int = 3,
) -> tuple[tuple[str, str], ...]:
    """Return a seeded randomized CPU/CUDA order with counterbalanced pairs.

    ``cuda_strict`` is the source API backend token. The two orientations have
    equal counts for an even schedule and differ by one for an odd schedule.
    No post-generation repair is performed.
    """
    if type(seed) is not int:
        raise ValueError("seed must be a built-in integer")
    if type(pairs) is not int or not 3 <= pairs <= MAX_PAIRED_TRIALS:
        raise ValueError("pairs must be an integer from 3 through 16")

    rng = random.Random(seed)
    orientations = [("cpu", "cuda_strict"), ("cuda_strict", "cpu")]
    if pairs % 2:
        majority = rng.choice(orientations)
        minority = orientations[1] if majority == orientations[0] else orientations[0]
        schedule = [majority] * (pairs // 2 + 1) + [minority] * (pairs // 2)
    else:
        schedule = orientations * (pairs // 2)
    rng.shuffle(schedule)
    return tuple(schedule)


def _positive_seconds(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return seconds


def _positive_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    total = low + high
    # Keep the normal midpoint for subnormals; avoid overflow for huge inputs.
    return total / 2 if math.isfinite(total) else low + (high - low) / 2


def aggregate_paired_trials(
    trials: Sequence[Mapping], *, minimum_pairs: int = 3,
) -> dict:
    """Aggregate bounded end-to-end paired receipts without granting qualification.

    Each trial binds a randomized API execution order, one exact source context,
    complete API wall timings, per-arm independent-oracle outcomes, and CPU/CUDA
    parity. Any oracle or parity failure makes the aggregate fail closed.
    """
    if (type(minimum_pairs) is not int
            or not 3 <= minimum_pairs <= MAX_PAIRED_TRIALS):
        raise ValueError("minimum_pairs must be an integer from 3 through 16")
    if (not isinstance(trials, Sequence)
            or isinstance(trials, (str, bytes, bytearray))):
        raise ValueError("trials must be a sequence of trial mappings")
    trial_count = len(trials)
    if not minimum_pairs <= trial_count <= MAX_PAIRED_TRIALS:
        raise ValueError("trial count must meet minimum_pairs and stay within the bounded pair range")

    context = None
    cpu_times: list[float] = []
    cuda_times: list[float] = []
    pair_wins = {"cpu": 0, "cuda": 0, "tie": 0}
    pair_winners: list[str] = []
    orders: list[tuple[str, str]] = []
    correctness_pass = True

    for index in range(trial_count):
        trial = trials[index]
        if not isinstance(trial, Mapping):
            raise ValueError(f"trial {index} must be a mapping")
        order = trial.get("execution_order")
        if (type(order) not in (tuple, list) or len(order) != 2
                or any(type(token) is not str for token in order)
                or set(order) != {"cpu", "cuda_strict"}):
            raise ValueError(f"trial {index} execution_order must contain cpu and cuda_strict once")
        orders.append(tuple(order))

        fingerprint = trial.get("context_fingerprint")
        if (type(fingerprint) is not str
                or _CONTEXT_SHA256.fullmatch(fingerprint) is None):
            raise ValueError(f"trial {index} context_fingerprint must be a nonempty SHA-256")
        if context is None:
            context = fingerprint
        elif fingerprint != context:
            raise ValueError("all trials must share one exact context fingerprint")

        if trial.get("timing_scope") != TIMING_SCOPE:
            raise ValueError(f"trial {index} timing_scope must be the complete source API wall")
        cpu_seconds = _positive_seconds(trial.get("cpu_seconds"), "cpu_seconds")
        cuda_seconds = _positive_seconds(trial.get("cuda_seconds"), "cuda_seconds")
        cpu_times.append(cpu_seconds)
        cuda_times.append(cuda_seconds)

        for field in ("cpu_oracle_pass", "cuda_oracle_pass", "parity_pass"):
            if type(trial.get(field)) is not bool:
                raise ValueError(f"trial {index} {field} must be a boolean")
            correctness_pass = correctness_pass and trial[field]

        pair_winner = ("cpu" if cpu_seconds < cuda_seconds else
                       "cuda" if cuda_seconds < cpu_seconds else "tie")
        pair_winners.append(pair_winner)
        pair_wins[pair_winner] += 1

    cpu_median = _positive_median(cpu_times)
    cuda_median = _positive_median(cuda_times)
    median_winner = ("cpu" if cpu_median < cuda_median else
                     "cuda" if cuda_median < cpu_median else None)
    majority_winner = next(
        (backend for backend in ("cpu", "cuda")
         if pair_wins[backend] > trial_count / 2), None)
    winner = (majority_winner
              if correctness_pass and majority_winner == median_winner else None)

    return {
        "schema": "source_profile_paired_trials.aggregate.v1",
        "pair_count": trial_count,
        "context_fingerprint": context,
        "timing_scope": TIMING_SCOPE,
        "execution_orders": [list(order) for order in orders],
        "pair_winners": pair_winners,
        "pair_wins": pair_wins,
        "median_seconds": {"cpu": cpu_median, "cuda": cuda_median},
        "aggregate_pass": correctness_pass,
        "winner": winner,
    }


__all__ = (
    "MAX_PAIRED_TRIALS", "TIMING_SCOPE", "aggregate_paired_trials",
    "build_randomized_paired_schedule",
)
