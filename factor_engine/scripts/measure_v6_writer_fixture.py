"""Isolated synthetic writer measurement, never real-factor acceptance."""
import argparse
import gc
import json
from pathlib import Path
import resource
import threading
import time

import numpy as np
import pandas as pd
import psutil

from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
from factor_engine.runtime.durable_artifact_sink import write_verified_factor_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error("root must already exist")
    index = pd.MultiIndex.from_product([
        pd.date_range("2020-01-01", periods=1000),
        [f"SYNTHETIC-{i:03d}" for i in range(100)]], names=["timestamp", "instrument"])
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop_sampling = threading.Event()

    def sample_rss():
        while not stop_sampling.wait(0.002):
            samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.monotonic()
    artifacts = []
    for ordinal in range(8):
        value = pd.Series(np.sin(np.arange(len(index), dtype=float) / 100 + ordinal),
                          index=index, name=f"fixture-{ordinal}")
        receipt = write_verified_factor_artifact(
            root, "synthetic-writer-only", ordinal, value.name, value,
            policy=DefaultExecutionPolicy(), budget_bytes=32 * 1024**2)
        manifest = json.loads(Path(receipt["path"]).read_text())
        artifacts.append({"ordinal": ordinal, "receipt": receipt,
                          "conversions": [c["conversions"] for c in manifest["chunks"]]})
        del value, manifest
        gc.collect()
    elapsed = time.monotonic() - started
    samples.append(process.memory_info().rss)
    stop_sampling.set()
    sampler.join()
    report = {
        "status": "MEASURED_SYNTHETIC_ONLY", "rows_per_factor": len(index),
        "factors": len(artifacts), "seconds": elapsed,
        "verified_cells_per_second": len(index) * len(artifacts) / elapsed,
        "process_baseline_rss_bytes": baseline,
        "process_final_rss_bytes": process.memory_info().rss,
        "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_basis": "Linux ru_maxrss, includes imports and data construction; not per-buffer copies",
        "interval_sampled_peak_rss_bytes": max(samples),
        "interval_sampled_growth_bytes": max(samples) - baseline,
        "interval_samples": len(samples),
        "interval_basis": "2ms sampled current RSS from pre-loop baseline through final GC; may miss short peaks; includes runtime initialization, inputs and retained allocator state; not isolated writer scratch",
        "workspace_admission_bytes": 32 * 1024**2,
        "boundaries": ["writer-only", "synthetic values", "sequential", "no backend benchmark",
                       "not hard RSS bound proof", "not 100k factors", "no production publication"],
        "artifacts": artifacts,
    }
    (root / "measurement.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}))


if __name__ == "__main__":
    main()
