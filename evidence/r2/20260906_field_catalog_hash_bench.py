import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from factor_engine.fields import FIELD_REGISTRY as r

def uncached():
    return hashlib.sha256(json.dumps(r.export_catalog(), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, default=str).encode()).hexdigest()

expected=uncached()
measurements={}
for name, fn in [('full_export_baseline',uncached),('content_fingerprint_cache',r.catalog_hash)]:
    assert fn()==expected
    start=time.perf_counter()
    for _ in range(300): assert fn()==expected
    measurements[name]=time.perf_counter()-start
result={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'timestamp':datetime.now(timezone.utc).isoformat(),'fields':len(r.fields()),
    'iterations':300,'seconds':measurements,'same_hash':True,
    'speedup':measurements['full_export_baseline']/measurements['content_fingerprint_cache'],
    'limitation':'single-process catalog hash microbenchmark, not end-to-end run_many throughput'}
Path('evidence/r2/20260906_field_catalog_hash_benchmark.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
