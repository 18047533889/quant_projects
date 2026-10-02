"""Bounded native Polars feasibility probe for layered decay.

Default mode uses a small in-memory panel. Pass --full-panel to reproduce the
480 x 3,000 QE+recurrence timing; that mode is intentionally opt-in because its
wide expression plan peaks around a few GiB. No files or datasets are written.
This is feasibility evidence only, not production/admission certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import polars as pl

from factor_optimizer.adapters.layered_decay import _assign_daily_quantiles
from factor_preprocess.transforms.layered_decay import layered_decay


ROOT = Path(__file__).resolve().parents[2]
N_LAYERS = 20
ORDINARY_RTOL = 1e-12
ORDINARY_ATOL = 1e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_polars(values: np.ndarray, bins: np.ndarray, half_lives, present=None):
    """40 native adjusted-EWM streams; returns rows in caller order.

    Input layer IDs are precomputed. Each source observation stays in its
    origin layer. A non-finite/missing layer or a gap in an asset's date
    sequence starts a fresh segment. A row's output consumes only its prior
    source row.
    """
    n_dates, n_assets = values.shape
    original_idx = np.arange(n_dates * n_assets, dtype=np.int64)
    dates = np.repeat(np.arange(n_dates, dtype=np.int32), n_assets)
    assets = np.tile(np.arange(n_assets, dtype=np.int32), n_dates)
    x = values.ravel()
    q = bins.ravel()
    if present is not None:
        keep = np.asarray(present, dtype=bool).ravel()
        original_idx, dates, assets, x, q = (
            col[keep] for col in (original_idx, dates, assets, x, q)
        )

    frame = pl.DataFrame(
        {"idx": original_idx, "date": dates, "asset": assets, "x": x, "q": q}
    ).sort(["asset", "date"])
    frame = frame.with_columns(
        (pl.col("x").is_finite() & (pl.col("q") >= 0)).alias("valid"),
        pl.col("date").diff().over("asset").alias("date_diff"),
    )
    frame = frame.with_columns(
        (
            (
                ~pl.col("valid")
                | pl.col("date_diff").is_null()
                | (pl.col("date_diff") != 1)
            )
            .cast(pl.UInt32)
            .cum_sum()
            .over("asset")
        ).alias("seg")
    )
    frame = frame.with_columns(
        pl.col("valid").cast(pl.UInt32).cum_sum().over(["asset", "seg"]).alias("age")
    )
    frame = frame.with_columns(
        pl.col("x").shift(1).over("asset").alias("xlag"),
        pl.col("q").shift(1).over("asset").alias("qlag"),
        pl.col("valid").shift(1).over("asset").alias("vlag"),
        pl.col("seg").shift(1).over("asset").alias("slag"),
        pl.col("age").shift(1).over("asset").alias("alag"),
    )
    frame = frame.with_columns(
        (pl.col("vlag").fill_null(False) & (pl.col("date_diff") == 1)).alias("lagok")
    )

    alpha = np.asarray(
        [-math.expm1(-math.log(2.0) / float(h)) for h in half_lives],
        dtype=np.float64,
    )
    if len(alpha) != N_LAYERS:
        raise ValueError("the feasibility probe requires twenty layer half-lives")
    rho = 1.0 - alpha

    inputs = []
    for layer in range(N_LAYERS):
        inputs.extend(
            [
                (
                    pl.when(pl.col("qlag") == layer)
                    .then(pl.col("xlag") / N_LAYERS)
                    .otherwise(0.0)
                    .alias(f"value_{layer}")
                ),
                (pl.when(pl.col("qlag") == layer).then(1.0).otherwise(0.0)
                 .alias(f"indicator_{layer}")),
            ]
        )
    frame = frame.with_columns(inputs)

    streams = []
    for layer, a in enumerate(alpha):
        streams.extend(
            [
                pl.col(f"value_{layer}")
                .ewm_mean(alpha=float(a), adjust=True, min_samples=1)
                .over(["asset", "slag"])
                .alias(f"value_ewm_{layer}"),
                pl.col(f"indicator_{layer}")
                .ewm_mean(alpha=float(a), adjust=True, min_samples=1)
                .over(["asset", "slag"])
                .alias(f"indicator_ewm_{layer}"),
            ]
        )
    frame = frame.with_columns(streams)

    numerator_terms = []
    denominator_terms = []
    for layer, r in enumerate(rho):
        qmass = 1.0 - pl.lit(float(r)).pow(pl.col("alag").cast(pl.Float64))
        numerator_terms.append(pl.col(f"value_ewm_{layer}") * qmass)
        denominator_terms.append(pl.col(f"indicator_ewm_{layer}") * qmass)
    numerator = pl.sum_horizontal(numerator_terms)
    denominator = pl.sum_horizontal(denominator_terms)
    return (
        frame.select(
            "idx",
            pl.when(pl.col("lagok") & (denominator > 0))
            .then(numerator / denominator * N_LAYERS)
            .otherwise(None)
            .alias("value"),
        )
        .sort("idx")["value"]
        .to_numpy()
    )


def _measure_case(values: np.ndarray, repetitions: int) -> dict:
    half_lives = tuple(np.linspace(1.0, 60.0, N_LAYERS))

    def published_fp():
        bins = _assign_daily_quantiles(values)
        return layered_decay(values, bins, half_lives, allow_research=True).ravel()

    def native():
        bins = _assign_daily_quantiles(values)
        return native_polars(values, bins, half_lives)

    expected = published_fp()
    actual = native()
    finite = np.isfinite(expected)
    if not np.array_equal(finite, np.isfinite(actual)):
        raise AssertionError("finite/NaN masks differ")
    if not np.allclose(
        expected[finite], actual[finite], rtol=ORDINARY_RTOL, atol=ORDINARY_ATOL
    ):
        raise AssertionError("native output differs beyond recorded ordinary tolerance")

    timings = {"published_fp_qe_plus_recurrence_s": [], "native_polars_qe_sort_recurrence_restore_s": []}
    for repetition in range(repetitions):
        calls = (
            (("published_fp_qe_plus_recurrence_s", published_fp),
             ("native_polars_qe_sort_recurrence_restore_s", native))
            if repetition % 2 == 0
            else (("native_polars_qe_sort_recurrence_restore_s", native),
                  ("published_fp_qe_plus_recurrence_s", published_fp))
        )
        for name, call in calls:
            start = time.perf_counter()
            output = call()
            timings[name].append(time.perf_counter() - start)
            if not np.array_equal(np.isfinite(output), finite) or not np.allclose(
                expected[finite], output[finite], rtol=ORDINARY_RTOL, atol=ORDINARY_ATOL
            ):
                raise AssertionError(f"{name} drifted on repetition {repetition}")

    fp_median = statistics.median(timings["published_fp_qe_plus_recurrence_s"])
    native_median = statistics.median(
        timings["native_polars_qe_sort_recurrence_restore_s"]
    )
    return {
        "shape": list(values.shape),
        "rows": int(values.size),
        "finite_outputs": int(finite.sum()),
        "timings_s": timings,
        "median_published_fp_s": fp_median,
        "median_native_polars_s": native_median,
        "published_over_native_speed_ratio": fp_median / native_median,
        "max_abs_difference": float(np.max(np.abs(expected[finite] - actual[finite]))),
        "max_relative_difference": float(
            np.max(
                np.abs(expected[finite] - actual[finite])
                / np.maximum(np.abs(expected[finite]), np.finfo(float).tiny)
            )
        ),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def _correctness_probes(seed: int) -> dict:
    half_life_sets = (tuple(range(1, 21)), tuple(np.linspace(1.0, 60.0, 20)))

    # The recorded seed and the CLI-derived seed are both retained: cancellation
    # near zero may fail relative-only comparison despite tiny absolute error.
    gap_tie_results = {}
    for probe_seed in sorted({92, seed}):
        rng = np.random.default_rng(probe_seed)
        values = rng.integers(-4, 5, size=(64, 80)).astype(float)
        present = rng.random(values.shape) > 0.035
        values[~present] = np.nan
        bins = _assign_daily_quantiles(values)
        bins[~present] = -1
        lives = half_life_sets[1]
        expected = layered_decay(values, bins, lives, allow_research=True).ravel()
        actual = native_polars(values, bins, lives, present=present)
        expected = expected[np.flatnonzero(present.ravel())]
        finite = np.isfinite(expected)
        if not np.array_equal(finite, np.isfinite(actual)):
            raise AssertionError(f"gap/tie mask mismatch for seed={probe_seed}")
        nonzero = finite & (expected != 0)
        difference = np.abs(expected[finite] - actual[finite])
        max_abs = float(np.max(difference))
        max_rel = float(
            np.max(np.abs(expected[nonzero] - actual[nonzero]) / np.abs(expected[nonzero]))
        )
        ordinary_match = bool(
            np.allclose(expected[finite], actual[finite], rtol=1e-12, atol=1e-12)
        )
        if not ordinary_match:
            raise AssertionError(f"gap/tie exceeded ordinary tolerance for seed={probe_seed}")
        gap_tie_results[str(probe_seed)] = {
            "shape": [64, 80],
            "present_rows": int(present.sum()),
            "finite_outputs": int(finite.sum()),
            "mask_equal": True,
            "max_abs_difference": max_abs,
            "max_relative_nonzero_difference": max_rel,
            "strict_relative_only_pass": bool(max_rel <= 1e-12),
            "ordinary_allclose_rtol_1e-12_atol_1e-12": ordinary_match,
        }

    tiny = []
    for magnitude in (1e-300, 1e-310, 1e-320):
        for lives in half_life_sets:
            small_values = np.full((64, 4), magnitude)
            layer_ids = np.fromfunction(
                lambda t, a: (t * 7 + a * 5) % N_LAYERS,
                small_values.shape,
                dtype=int,
            ).astype(np.int32)
            reference = layered_decay(
                small_values, layer_ids, lives, allow_research=True
            ).ravel()
            candidate = native_polars(small_values, layer_ids, lives)
            finite = np.isfinite(reference)
            nonzero = finite & (reference != 0)
            tiny.append(
                {
                    "magnitude": magnitude,
                    "half_life_min": float(min(lives)),
                    "half_life_max": float(max(lives)),
                    "reference_nonzero_outputs": int(nonzero.sum()),
                    "native_zero_outputs": int(np.sum(candidate[finite] == 0)),
                    "native_finite": bool(np.isfinite(candidate[finite]).all()),
                    "max_relative_error_atol_zero": float(
                        np.max(
                            np.abs(candidate[nonzero] - reference[nonzero])
                            / np.abs(reference[nonzero])
                        )
                    ),
                }
            )

    maximum = np.finfo(float).max
    extreme_layers = np.fromfunction(
        lambda t, a: (t * 7 + a * 5) % N_LAYERS,
        (80, 4),
        dtype=int,
    ).astype(np.int32)
    extreme_checks = {}
    for label, magnitude in (("tested_upper_envelope", maximum / 10.0),
                             ("float64_max_outside_envelope", maximum)):
        extreme_values = np.full((80, 4), magnitude)
        extreme_values[1::2] *= -1.0
        reference = layered_decay(
            extreme_values, extreme_layers, half_life_sets[1], allow_research=True
        ).ravel()
        candidate = native_polars(extreme_values, extreme_layers, half_life_sets[1])
        finite = np.isfinite(reference)
        extreme_checks[label] = {
            "input_abs_max": float(magnitude),
            "native_finite_outputs": int(np.isfinite(candidate[finite]).sum()),
            "reference_finite_outputs": int(finite.sum()),
            "nonfinite_native_outputs": int(np.sum(~np.isfinite(candidate[finite]))),
            "rtol_1e-12_atol_1e-12_match": bool(
                np.allclose(
                    reference[finite], candidate[finite], rtol=1e-12, atol=1e-12
                )
            ),
        }
    return {
        "gap_ties_by_seed": gap_tie_results,
        "tiny_values": tiny,
        "float64_extreme": extreme_checks,
        "half_life_sets": [[1, 20], [1.0, 60.0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-panel", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8087)
    args = parser.parse_args()
    if args.repetitions != 3:
        raise ValueError("use exactly three alternating timing repetitions")
    if args.full_panel:
        shape = (480, 3000)
    else:
        shape = (64, 80)
    values = np.random.default_rng(args.seed).standard_normal(shape)

    source_paths = (
        "factor_preprocess/factor_preprocess/transforms/layered_decay.py",
        "factor_preprocess/factor_preprocess/transforms/layered_decay_state.py",
        "factor_optimizer/factor_optimizer/adapters/layered_decay.py",
    )
    report = {
        "benchmark": "native-layered-decay-feasibility",
        "classification": "exploratory feasibility only; not production/admission evidence",
        "repo_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {"python": __import__("platform").python_version(),
                        "numpy": np.__version__, "polars": pl.__version__},
        "source_sha256": {
            path: _sha256(ROOT / path) for path in source_paths
        },
        "method": {
            "repetitions": 3,
            "timing_order": "FP,Polars; Polars,FP; FP,Polars",
            "input": "synthetic in-memory panel; precomputed QE 20-layer IDs",
            "native": "20 adjusted EWM value streams + 20 adjusted EWM indicator streams",
            "scaling": "fixed causal input division by 20; qmass=1-rho**age; sum layer numerators/masses; restore original row order",
            "ordinary_tolerance": {"rtol": ORDINARY_RTOL, "atol": ORDINARY_ATOL},
            "no_writes": True,
        },
        "correctness_probes": _correctness_probes(args.seed + 5),
        "e2e": _measure_case(values, args.repetitions),
        "numeric_envelope": {
            "tested_safe_probe": "finite nonzero magnitudes from 1e-300 through float64_max/10 on discrete probes; h in [1,60]",
            "fallback": "use the published FP scale-normalized recurrence when nonzero inputs may be below 1e-300 or above float64_max/10; float64_max produced a non-finite native output",
            "caveat": "The fixed /20 path loses relative precision in the subnormal/tiny domain and can overflow at float64 maximum. Do not derive a segment maximum from future rows.",
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
