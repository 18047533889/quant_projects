#!/usr/bin/env python3
"""Run a bounded, numerical QE scale staircase through the public V5 API."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
from quant_evaluator.jobs.handler import QEJobHandler
from quant_evaluator.jobs.refs import QEJobShard, shard_factor_refs, write_manifest
from quant_platform.app.contracts import JobSpec


T = 4
N = 20
BATCH_SIZE = 256
TOTALS = (1_000, 10_000, 100_000)


@dataclass
class TierResult:
    factors_requested: int
    metric_instances: int
    factor_instance_attempted: int
    factor_instance_completed: int
    factor_instance_failed: int
    failure_rate: float
    shards: int
    failed_shards: int
    batch_limit: int
    elapsed_seconds: float
    factors_per_second: float
    factor_instances_per_second: float
    rss_start_mib: float
    rss_peak_mib: float
    rss_peak_delta_mib: float
    output_files: int
    output_bytes: int


def rss_peak_mib() -> float:
    # Linux ru_maxrss is KiB; this script is intentionally server-c/Linux only.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def command_text(args: list[str]) -> str:
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def hardware() -> dict:
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("MemTotal:", "SwapTotal:")):
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": command_text(["sh", "-c", "lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1"]),
        "logical_cpus": os.cpu_count(),
        "memory": meminfo,
        "numpy": np.__version__,
        "thread_env": {name: os.environ.get(name) for name in
                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "POLARS_MAX_THREADS")},
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def labels() -> LabelBundle:
    rng = np.random.default_rng(20260907)
    return LabelBundle(
        target_id="return_h1",
        values=np.ascontiguousarray(rng.normal(size=(T, N))),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
        asset_axis=AxisRef("asset", "str", N,
                           np.asarray([f"asset_{i:02d}" for i in range(N)])),
        source_ref="generated:seed=20260907",
    )


def make_batch(refs: tuple[str, ...]) -> FactorBatch:
    first = int(refs[0].rsplit(":", 1)[1])
    rng = np.random.default_rng(20260907 + first)
    values = np.ascontiguousarray(rng.normal(size=(T, N, len(refs))))
    return FactorBatch(
        tuple(refs), AxisRef("time", "int", T, np.arange(T)),
        AxisRef("asset", "str", N,
                np.asarray([f"asset_{i:02d}" for i in range(N)])), values,
    )


def run_tier(total: int, root: Path, instances: tuple[MetricInstance, ...],
             label_bundle: LabelBundle) -> TierResult:
    tier_root = root / f"f{total}"
    manifests = tier_root / "manifests"
    sink = tier_root / "results"
    factor_shards = shard_factor_refs((f"factor:{i}" for i in range(total)),
                                      shard_size=BATCH_SIZE)
    handler = QEJobHandler(
        factor_batch_resolver=make_batch,
        label_resolver=lambda _: label_bundle,
        scenario_resolver=lambda _: EvaluationScenario(label_bundle),
    )
    started_rss = rss_peak_mib()
    started = time.perf_counter()
    completed = 0
    failed = 0
    failed_shards = 0
    output_refs: list[str] = []
    for shard_index, refs in enumerate(factor_shards):
        shard = QEJobShard(
            refs, "label:h1", tuple(item.to_dict() for item in instances),
            (("scale", "scenario:scale"),), str(sink), shard_index,
        )
        manifest = write_manifest(manifests, shard)
        spec = JobSpec(
            "qe_metric_instance_shard", f"v5-scale-{total}-{shard_index}",
            input_artifact_refs=(manifest,),
        )
        try:
            result = handler(spec)
            output_refs.extend(result.output_artifact_refs)
            completed += len(refs) * len(instances)
        except Exception:
            failed += len(refs) * len(instances)
            failed_shards += 1
        finally:
            # The resolver-created FactorBatch and evaluate bundle are no longer
            # referenced after the handler returns. Force collection per shard.
            gc.collect()
    elapsed = time.perf_counter() - started
    attempted = total * len(instances)
    unique_outputs = sorted(set(output_refs))
    output_bytes = sum(Path(ref).stat().st_size for ref in unique_outputs)
    peak = rss_peak_mib()
    return TierResult(
        factors_requested=total,
        metric_instances=len(instances),
        factor_instance_attempted=attempted,
        factor_instance_completed=completed,
        factor_instance_failed=failed,
        failure_rate=failed / attempted,
        shards=len(factor_shards),
        failed_shards=failed_shards,
        batch_limit=BATCH_SIZE,
        elapsed_seconds=elapsed,
        factors_per_second=total / elapsed,
        factor_instances_per_second=completed / elapsed,
        rss_start_mib=started_rss,
        rss_peak_mib=peak,
        rss_peak_delta_mib=max(0.0, peak - started_rss),
        output_files=len(unique_outputs),
        output_bytes=output_bytes,
    )


def markdown(payload: dict) -> str:
    lines = [
        "# V5 QE numerical staircase",
        "",
        f"Run (UTC): `{payload['run_utc']}`",
        "",
        "> Scope warning: this is a small-panel bounded-memory validation with "
        "T=4 and N=20. It is not a production-throughput benchmark and must not "
        "be extrapolated to production panel sizes.",
        "",
        "Every shard used the public `EvaluationRequest`/`evaluate` path through "
        "`QEJobHandler`, with atomic result-sink writes, then released per-shard objects.",
        "",
        "| Factors | Shards | factor×instance done | Failed | Failure rate | Seconds | factor×instance/s | Peak RSS MiB | Peak delta MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["tiers"]:
        lines.append(
            f"| {row['factors_requested']:,} | {row['shards']} | "
            f"{row['factor_instance_completed']:,} | {row['factor_instance_failed']:,} | "
            f"{row['failure_rate']:.6%} | {row['elapsed_seconds']:.3f} | "
            f"{row['factor_instances_per_second']:.1f} | {row['rss_peak_mib']:.1f} | "
            f"{row['rss_peak_delta_mib']:.1f} |"
        )
    lines.extend(["", "## Configuration", "", "```json",
                  json.dumps(payload["configuration"], indent=2, sort_keys=True),
                  "```", "", "## Hardware", "", "```json",
                  json.dumps(payload["hardware"], indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default="evidence/v5")
    args = parser.parse_args()
    evidence = Path(args.evidence_dir).resolve()
    run_root = evidence / "numeric_staircase_artifacts"
    instance_specs = (
        MetricInstance("rank_ic", parameters={"min_assets": 10}, scenario_id="scale", horizon=1,
                       price_convention="vwap_to_vwap"),
        MetricInstance("coverage", scenario_id="scale", horizon=1,
                       price_convention="vwap_to_vwap"),
    )
    label_bundle = labels()
    tiers = [asdict(run_tier(total, run_root, instance_specs, label_bundle)) for total in TOTALS]
    payload = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "totals": list(TOTALS), "T": T, "N": N, "batch_max": BATCH_SIZE,
            "instances": [item.to_dict() for item in instance_specs],
            "path": "public EvaluationRequest/evaluate via QEJobHandler atomic sink",
            "production_throughput_claim": False,
        },
        "hardware": hardware(),
        "tiers": tiers,
    }
    atomic_json(evidence / "v5_numeric_staircase.json", payload)
    atomic_text(evidence / "v5_numeric_staircase.md", markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(row["factor_instance_failed"] == 0 for row in tiers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
