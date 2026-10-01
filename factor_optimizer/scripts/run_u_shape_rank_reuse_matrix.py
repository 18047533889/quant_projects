"""Run a serial, alternating cached/uncached U-rank benchmark matrix.

Each child process writes one compact JSON record. This command is intended
for a quiet, explicitly authorized benchmark window; it does not run tests or
remove or overwrite existing evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS = PROJECT_ROOT / "factor_optimizer/scripts/benchmark_u_shape_rank_reuse.py"
BENCHMARK_DOCS = PROJECT_ROOT / "factor_optimizer/docs/benchmarks"
DEFAULT_OUTPUT_DIR = BENCHMARK_DOCS / "u_shape_rank_reuse_20261001_runs"
DEFAULT_SENSITIVITY_OUTPUT_DIR = BENCHMARK_DOCS / "u_shape_rank_reuse_candidate_sensitivity_20261001_runs"
MAX_JSON_RECORD_BYTES = 64 * 1024
CASES = (
    (200, 640, 76_800),
    (167, 5_000, 500_000),
    (334, 5_000, 1_000_000),
    (500, 5_000, 1_500_000),
)
PAIR_ORDERS = (
    ("cached", "uncached"),
    ("uncached", "cached"),
    ("cached", "uncached"),
)
SENSITIVITY_CASES = ((167, 5_000, 500_000), (334, 5_000, 1_000_000))
SENSITIVITY_PROFILES = (
    ("u_only_max8", ("U_SHAPE_REPAIR",), 8),
    ("inverted_u_only_max6", ("INVERTED_U_REPAIR",), 6),
    ("u_and_inverted_max14", ("U_SHAPE_REPAIR", "INVERTED_U_REPAIR"), 14),
)


def build_plan(output_dir: Path) -> list[dict]:
    """Return the exact 24 invocations without creating files or directories."""
    plan = []
    for time_points, assets, expected_train_rows in CASES:
        for pair_number, modes in enumerate(PAIR_ORDERS, start=1):
            for mode in modes:
                output_path = output_dir / (
                    f"u_shape_rank_reuse_{time_points}x{assets}_"
                    f"train{expected_train_rows}_pair{pair_number}_{mode}.json"
                )
                plan.append({
                    "time_points": time_points,
                    "assets": assets,
                    "expected_train_rows": expected_train_rows,
                    "pair": pair_number,
                    "mode": mode,
                    "output_path": output_path,
                })
    return plan


def build_sensitivity_plan(output_dir: Path) -> list[dict]:
    """Return a non-mutating 36-run plan for the supported candidate profiles."""
    plan = []
    for profile, families, maximum_candidates in SENSITIVITY_PROFILES:
        for time_points, assets, expected_train_rows in SENSITIVITY_CASES:
            for pair_number, modes in enumerate(PAIR_ORDERS, start=1):
                for mode in modes:
                    output_path = output_dir / (
                        f"{profile}_{time_points}x{assets}_train{expected_train_rows}_"
                        f"pair{pair_number}_{mode}.json"
                    )
                    plan.append({
                        "profile": profile, "families": families,
                        "maximum_candidates": maximum_candidates,
                        "time_points": time_points, "assets": assets,
                        "expected_train_rows": expected_train_rows,
                        "pair": pair_number, "mode": mode,
                        "output_path": output_path,
                    })
    return plan


def _prepare_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    docs_root = BENCHMARK_DOCS.resolve()
    if docs_root not in resolved.parents:
        raise ValueError(f"output directory must be inside {docs_root}")
    if resolved.exists():
        if any(resolved.iterdir()):
            raise FileExistsError(f"refusing non-empty evidence directory: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _invoke(spec: dict, panel_seed: int) -> dict:
    output_path = spec["output_path"]
    command = [
        sys.executable,
        str(HARNESS),
        "--time-points", str(spec["time_points"]),
        "--assets", str(spec["assets"]),
        "--mode", spec["mode"],
        "--seed", str(panel_seed),
        "--json-out", str(output_path),
    ]
    if "families" in spec:
        command.extend(["--families", ",".join(spec["families"])])
        command.extend(["--maximum-candidates", str(spec["maximum_candidates"])])
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark child failed ({result.returncode}): {result.stderr[-2000:]}"
        )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark child returned invalid JSON: {exc}") from exc
    if not output_path.is_file():
        raise RuntimeError(f"child did not persist its JSON record: {output_path}")
    if output_path.stat().st_size > MAX_JSON_RECORD_BYTES:
        raise RuntimeError(f"run record exceeded {MAX_JSON_RECORD_BYTES} bytes")
    if payload.get("mode") != spec["mode"]:
        raise RuntimeError(f"unexpected child mode in record: {output_path}")
    if payload.get("panel_shape") != [spec["time_points"], spec["assets"]]:
        raise RuntimeError(f"unexpected panel shape in record: {output_path}")
    if payload.get("training_frame_rows") != spec["expected_train_rows"]:
        raise RuntimeError(
            f"TRAIN row mismatch at {output_path}: "
            f"expected {spec['expected_train_rows']}, "
            f"got {payload.get('training_frame_rows')}"
        )
    observations = payload.get("training_frame_observations", {})
    if (
        observations.get("consistent") is not True
        or observations.get("actual_rows") != spec["expected_train_rows"]
    ):
        raise RuntimeError(f"TRAIN frame observation mismatch at {output_path}")
    if "families" in spec:
        configured = payload.get("fixture", {}).get("batch_optimization_config", {})
        if configured.get("families") != list(spec["families"]):
            raise RuntimeError(f"family profile mismatch at {output_path}")
        if configured.get("maximum_candidates") != spec["maximum_candidates"]:
            raise RuntimeError(f"candidate budget mismatch at {output_path}")
        _validate_sensitivity_record(payload, spec, output_path)

    if spec["mode"] == "cached":
        observed = (
            observations.get("cached_fingerprint_rows", [])
            + observations.get("cached_fe_rank_rows", [])
        )
        if not observed or any(rows != spec["expected_train_rows"] for rows in observed):
            raise RuntimeError(f"cached TRAIN frame evidence mismatch at {output_path}")
    else:
        observed = observations.get("uncached_initial_train_execute_rows", [])
        if not observed or any(rows != spec["expected_train_rows"] for rows in observed):
            raise RuntimeError(f"uncached TRAIN frame evidence mismatch at {output_path}")
    cache_metrics = payload.get("rank_cache_metrics", {})
    if cache_metrics.get("instance_count") != 1:
        raise RuntimeError(f"expected one actual rank-cache instance: {output_path}")
    if not cache_metrics.get("class_identity_preserved"):
        raise RuntimeError(f"rank-cache factory changed class identity: {output_path}")
    return payload


def _validate_sensitivity_record(payload: dict, spec: dict, output_path) -> None:
    """Require a genuinely admitted search with observed rank-shape plans."""
    budget = payload.get("candidate_budget", {})
    if budget.get("status") != "admitted":
        raise RuntimeError(f"candidate budget was not admitted at {output_path}")
    required = budget.get("required")
    maximum = budget.get("maximum")
    if (type(required) is not int or type(maximum) is not int
            or required <= 0 or required > spec["maximum_candidates"]
            or maximum != spec["maximum_candidates"]):
        raise RuntimeError(f"invalid candidate budget evidence at {output_path}")
    if payload.get("candidate_count") != required:
        raise RuntimeError(f"candidate count/budget mismatch at {output_path}")
    identities = payload.get("observed_distinct_train_rank_plan_identities")
    if type(identities) is not int or identities <= 0 or identities > required:
        raise RuntimeError(f"invalid distinct TRAIN rank-plan count at {output_path}")


def _validate_sensitivity_pair(cached: dict, uncached: dict, output_path) -> int:
    """Require paired evidence and the same positive rank-plan workload."""
    if cached.get("candidate_evidence_sha256") != uncached.get("candidate_evidence_sha256"):
        raise RuntimeError(f"candidate evidence mismatch at {output_path}")
    cached_count = cached.get("observed_distinct_train_rank_plan_identities")
    uncached_count = uncached.get("observed_distinct_train_rank_plan_identities")
    if type(cached_count) is not int or cached_count <= 0 or cached_count != uncached_count:
        raise RuntimeError(f"cached/uncached distinct rank-plan mismatch at {output_path}")
    for payload in (cached, uncached):
        budget = payload.get("candidate_budget", {})
        if budget.get("status") != "admitted" or payload.get("candidate_count") != budget.get("required"):
            raise RuntimeError(f"sensitivity pair includes a non-admitted search at {output_path}")
    return cached_count


def run_matrix(output_dir: Path, *, panel_seed: int = 20261001) -> dict:
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError(
            "run with the project interpreter: .venv/bin/python "
            "factor_optimizer/scripts/run_u_shape_rank_reuse_matrix.py"
        )
    if not HARNESS.is_file():
        raise FileNotFoundError(HARNESS)
    durable_output_dir = _prepare_output_dir(output_dir)
    plan = build_plan(durable_output_dir)
    if len(plan) != 24 or len({item["output_path"] for item in plan}) != 24:
        raise AssertionError("matrix plan must contain 24 unique JSON output paths")

    records = []
    for spec in plan:
        payload = _invoke(spec, panel_seed)
        records.append((spec, payload))
        counters = payload["rank_cache_metrics"]["totals"]
        print(json.dumps({
            "event": "run_complete",
            "panel_shape": payload["panel_shape"],
            "train_rows": payload["training_frame_rows"],
            "pair": spec["pair"],
            "mode": spec["mode"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "peak_rss_mib": payload["process_peak_rss_mib"],
            "cache_hits": counters["hits"],
            "cache_misses": counters["misses"],
            "record": str(spec["output_path"]),
        }, sort_keys=True, separators=(",", ":")))

    panels = []
    for time_points, assets, train_rows in CASES:
        selected = [
            (spec, payload) for spec, payload in records
            if spec["time_points"] == time_points and spec["assets"] == assets
        ]
        by_pair = {}
        for spec, payload in selected:
            by_pair.setdefault(spec["pair"], {})[spec["mode"]] = payload
        pair_parity = {
            pair: modes["cached"]["candidate_evidence_sha256"]
            == modes["uncached"]["candidate_evidence_sha256"]
            for pair, modes in sorted(by_pair.items())
        }
        if len(pair_parity) != 3 or not all(pair_parity.values()):
            raise RuntimeError(
                f"candidate evidence mismatch for {time_points}x{assets}; "
                f"records remain at {durable_output_dir}"
            )
        per_mode = {}
        for mode in ("cached", "uncached"):
            mode_records = [payload for spec, payload in selected if spec["mode"] == mode]
            per_mode[mode] = {
                "median_elapsed_seconds": statistics.median(
                    item["elapsed_seconds"] for item in mode_records
                ),
                "median_peak_rss_mib": statistics.median(
                    item["process_peak_rss_mib"] for item in mode_records
                ),
                "cache_hit_counts": [
                    item["rank_cache_metrics"]["totals"]["hits"]
                    for item in mode_records
                ],
                "cache_miss_counts": [
                    item["rank_cache_metrics"]["totals"]["misses"]
                    for item in mode_records
                ],
                "cache_rank_call_counts": [
                    item["rank_cache_metrics"]["totals"]["rank_calls"]
                    for item in mode_records
                ],
            }
        panels.append({
            "panel_shape": [time_points, assets],
            "input_cells": time_points * assets,
            "train_rows": train_rows,
            "pair_count": len(by_pair),
            "evidence_parity_by_pair": pair_parity,
            "per_mode": per_mode,
        })
    summary = {
        "panel_seed": panel_seed,
        "optimizer_seed": 73,
        "run_count": len(records),
        "output_dir": str(durable_output_dir),
        "panels": panels,
    }
    print(json.dumps({"event": "matrix_complete", **summary},
                     sort_keys=True, separators=(",", ":")))
    return summary


def run_sensitivity_matrix(output_dir: Path, *, panel_seed: int = 20261001) -> dict:
    """Run admitted, paired family/count sensitivity cases serially."""
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError("run with the project interpreter: .venv/bin/python")
    durable_output_dir = _prepare_output_dir(output_dir)
    plan = build_sensitivity_plan(durable_output_dir)
    if len(plan) != 36 or len({item["output_path"] for item in plan}) != 36:
        raise AssertionError("sensitivity plan must contain 36 unique records")
    records = []
    for spec in plan:
        payload = _invoke(spec, panel_seed)
        records.append((spec, payload))
        print(json.dumps({"event": "sensitivity_run_complete", "profile": spec["profile"],
                          "train_rows": payload["training_frame_rows"], "pair": spec["pair"],
                          "mode": spec["mode"], "elapsed_seconds": payload["elapsed_seconds"],
                          "distinct_train_rank_plan_identities": payload[
                              "observed_distinct_train_rank_plan_identities"],
                          "record": str(spec["output_path"])},
                         sort_keys=True, separators=(",", ":")))
    profiles = []
    for profile, families, maximum_candidates in SENSITIVITY_PROFILES:
        for time_points, assets, train_rows in SENSITIVITY_CASES:
            selected = [(spec, payload) for spec, payload in records
                        if spec["profile"] == profile and spec["time_points"] == time_points]
            by_pair = {}
            for spec, payload in selected:
                by_pair.setdefault(spec["pair"], {})[spec["mode"]] = payload
            if len(by_pair) != 3 or any(set(modes) != {"cached", "uncached"}
                                        for modes in by_pair.values()):
                raise RuntimeError(f"incomplete sensitivity pairs for {profile}, {train_rows}; "
                                   f"records remain at {durable_output_dir}")
            parity = {}
            plan_counts = {}
            for pair, modes in sorted(by_pair.items()):
                plan_counts[pair] = _validate_sensitivity_pair(
                    modes["cached"], modes["uncached"],
                    f"{profile}, {train_rows}, pair {pair}")
                parity[pair] = True
            if len(set(plan_counts.values())) != 1:
                raise RuntimeError(f"distinct plan count changed across pairs for {profile}, "
                                   f"{train_rows}; records remain at {durable_output_dir}")
            per_mode = {}
            for mode in ("cached", "uncached"):
                mode_records = [payload for _, payload in selected
                                if payload["mode"] == mode]
                per_mode[mode] = {
                    "median_elapsed_seconds": statistics.median(
                        item["elapsed_seconds"] for item in mode_records),
                    "median_peak_rss_mib": statistics.median(
                        item["process_peak_rss_mib"] for item in mode_records),
                }
            profiles.append({"profile": profile, "families": list(families),
                             "maximum_candidates": maximum_candidates,
                             "panel_shape": [time_points, assets], "train_rows": train_rows,
                             "evidence_parity_by_pair": parity,
                             "distinct_train_rank_plan_count": next(iter(plan_counts.values())),
                             "candidate_count": selected[0][1]["candidate_count"],
                             "per_mode": per_mode,
                             "cached_elapsed_change_pct": 100 * (
                                 per_mode["cached"]["median_elapsed_seconds"] /
                                 per_mode["uncached"]["median_elapsed_seconds"] - 1),
                             "cached_peak_rss_change_mib": (
                                 per_mode["cached"]["median_peak_rss_mib"] -
                                 per_mode["uncached"]["median_peak_rss_mib"])})
    summary = {"panel_seed": panel_seed, "optimizer_seed": 73,
               "run_count": len(records), "output_dir": str(durable_output_dir),
               "profiles": profiles}
    print(json.dumps({"event": "sensitivity_matrix_complete", **summary},
                     sort_keys=True, separators=(",", ":")))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20261001, dest="panel_seed")
    parser.add_argument("--sensitivity", action="store_true",
                        help="run the isolated 36-record admitted rank-plan sensitivity profile")
    args = parser.parse_args()
    if args.sensitivity:
        output_dir = (args.output_dir if args.output_dir != DEFAULT_OUTPUT_DIR
                      else DEFAULT_SENSITIVITY_OUTPUT_DIR)
        run_sensitivity_matrix(output_dir, panel_seed=args.panel_seed)
    else:
        run_matrix(args.output_dir, panel_seed=args.panel_seed)


if __name__ == "__main__":
    main()
