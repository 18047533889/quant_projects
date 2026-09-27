"""Bounded synthetic A/B for the exact CPU Spearman row-tile refactor.

Run once per process, alternating ``old new new old`` to compare time/RSS.
The baseline source is pinned to the committed pre-change implementation.
No COS data is read and no files are written.
"""

import hashlib
import json
import resource
import subprocess
import sys
import time

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

BASELINE_REF = "9f16b217f6739063278c18771762204557a98abc"


def main():
    mode = sys.argv[1]
    rng = np.random.default_rng(20260928)
    t, n, f = 600, 2000, 12
    values = rng.normal(size=(t, n, f))
    values[rng.random(values.shape) < 0.08] = np.nan
    label_values = rng.normal(size=(t, n))
    label_values[rng.random(label_values.shape) < 0.04] = np.nan
    factors = FactorBatch(tuple(f"f{i}" for i in range(f)),
                          AxisRef("time", "int64", t, np.arange(t, dtype=np.int64)),
                          AxisRef("asset", "str", n,
                                  np.asarray([f"A{i}" for i in range(n)])), values,
                          validity=np.isfinite(values))
    labels = LabelBundle("return", label_values, 1,
                         decision_time=tuple(range(t)),
                         label_start_time=tuple(range(t)),
                         label_end_time=tuple(range(1, t + 1)),
                         validity=np.isfinite(label_values))
    if mode == "old":
        source = subprocess.check_output(
            ["git", "show", f"{BASELINE_REF}:quant_evaluator/metrics/ic.py"], text=True)
        namespace = {"__name__": "quant_evaluator.metrics.ic_baseline"}
        exec(compile(source, "<git:HEAD:ic.py>", "exec"), namespace)
        compute = namespace["compute_daily_ic"]
    elif mode == "new":
        from quant_evaluator.metrics.ic import compute_daily_ic
        compute = compute_daily_ic
    else:
        raise ValueError(mode)
    start = time.perf_counter()
    ic, counts = compute(factors, labels, method="spearman", min_assets=20)
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256(ic.tobytes() + counts.tobytes()).hexdigest()
    print(json.dumps({"mode": mode, "shape": [t, n, f],
                      "baseline_ref": BASELINE_REF,
                      "seconds": elapsed, "peak_rss_kib":
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "output_sha256": digest}))


if __name__ == "__main__":
    main()
