"""Force the smoke recovery path without modifying engine production behavior."""
import sys
from factor_engine.runtime.engine import FactorEngine
import smoke_catalog

def fail_batch(self, *args, **kwargs):
    raise RuntimeError("diagnostic forced batch abort to isolate single-run recovery")

FactorEngine.run_many = fail_batch
sys.argv = [
    "smoke_catalog.py", "--offset", "2186", "--limit", "10",
    "--prepare-chunk", "10", "--batch-size", "10", "--daily-only",
    "--timeout-seconds", "30", "--rss-rotate-mib", "1500",
    "--output", "forced-abort2186-recovery.jsonl.gz",
]
smoke_catalog.main()
