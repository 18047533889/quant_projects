import json
import sys
from pathlib import Path

from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler


old = AdaptiveBatchScheduler._transfer_cse_memory_ownership


def probe(self, ctx, task_id, lease):
    store = getattr(ctx, "shared_buffers", None)
    target = (
        store.capture_lease_target(task_id.split(":", 1)[1])
        if store is not None and task_id.startswith("cse:")
        else None
    )
    if task_id == "cse:740f5e830f25622e3aedf60b76ffa0140419018002285eba468a4185cebd198e":
        value = getattr(getattr(target, "_entry", None), "value", None)
        broker_lease = getattr(lease, "_broker_lease", lease)
        print(json.dumps({
            "event": "cse_transfer_probe",
            "task_id": task_id,
            "value_type": type(value).__module__ + "." + type(value).__qualname__,
            "value_repr": repr(value)[:120],
            "bytes": getattr(target, "bytes", None),
            "physical_ownership_supported": getattr(
                target, "physical_ownership_supported", None
            ),
            "lease_type": type(lease).__module__ + "." + type(lease).__qualname__,
            "admissible_peak_bytes": getattr(
                getattr(broker_lease, "_task", None), "admissible_peak_bytes", None
            ),
        }), flush=True)
    return old(self, ctx, task_id, lease)


AdaptiveBatchScheduler._transfer_cse_memory_ownership = probe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factor_catalog_20260915"))
from smoke_catalog import main

sys.argv = [
    "smoke_catalog.py",
    "--input", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/factor_catalog_review_r14a.williams_daily_smoke.input.jsonl.gz",
    "--output", "/home/sunhaiwei/quant_projects/evidence/factor_catalog_20260916/r14-williams-cse-diagnostic.output.jsonl.gz",
    "--limit", "20",
    "--batch-size", "20",
    "--batch-only",
    "--backend", "pandas",
    "--max-rss-mib", "1800",
    "--timeout-seconds", "20",
    "--deadline-mode", "external-watchdog",
    "--daily-only",
]
main()
