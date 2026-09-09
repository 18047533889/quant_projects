"""Real isolated-process L1 stress receipt; no production cache touched."""
import json
import resource
import time
import hashlib
from pathlib import Path
import numpy as np
import quant_evaluator.runtime.cache_v2 as cache_module
from quant_evaluator.runtime.cache_v2 import MemoryCacheLayer, ZlibCompressor

cache = MemoryCacheLayer(65536, ZlibCompressor(), workspace_budget_bytes=65536)
source = Path(cache_module.__file__)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
start = time.monotonic()
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
value = np.arange(8., dtype=float)
for _ in range(1_000_000):
    assert cache.put('one', value)
    cache.clear()
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert len(cache._compression_stats) == 1000
assert cache.get_stats()['current_size_bytes'] == 0
original_dumps = cache_module.pickle.dumps
def forbidden_pickle(*args, **kwargs):
    raise AssertionError('oversized payload reached serialization')
cache_module.pickle.dumps = forbidden_pickle
try:
    assert not cache.put('compressible', np.zeros(200000))
    assert not cache.put('incompressible', np.random.default_rng(99).normal(size=200000))
finally:
    cache_module.pickle.dumps = original_dumps
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
print(json.dumps({'iterations': 1_000_000, 'seconds': time.monotonic()-start,
    'ru_maxrss_before_kib': before, 'ru_maxrss_after_kib': after,
    'scope': 'Linux isolated process, payload/workspace budget excludes caller arrays and interpreter',
    'stats': cache.get_stats(), 'large_payloads_rejected_before_pickle': True,
    'source_sha256': source_hash, 'source_path': str(source)}, indent=2))
