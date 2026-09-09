"""Run separate pytest processes against one already-frozen V8 source manifest."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v8/final_regression'
MANIFEST = ROOT / 'evidence/v8/final_source/source_manifest.json'


def check_sources():
    records = json.loads(MANIFEST.read_text())['files']
    changed = [r['path'] for r in records
               if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() != r['sha256']]
    if changed:
        raise RuntimeError(f'Frozen source changed: {changed}')
    return len(records)


def lane(suites):
    results = {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               POLARS_MAX_THREADS='2', PYTHONPATH='factor_optimizer:factor_preprocess:.')
    for name, args in suites:
        local_env = dict(env)
        if name == 'vectorbt':
            local_env['PYTHONPATH'] = '/tmp/vectorbt_qs_declared_deps_v6:' + env['PYTHONPATH']
        command = [sys.executable, '-m', 'pytest', *args, '-q',
                   '--junitxml=' + str(OUT / f'{name}.xml')]
        print(f'START {name}', flush=True)
        with (OUT/f'{name}.log').open('w') as log:
            result = subprocess.run(command,cwd=ROOT,env=local_env,stdout=log,
                                    stderr=subprocess.STDOUT,timeout=900)
        results[name] = dict(command=command,returncode=result.returncode,
                             log=f'final_regression/{name}.log')
        print(f'FINISH {name}: {result.returncode}', flush=True)
    return results


def main():
    count = check_sources()
    OUT.mkdir(exist_ok=False)
    lanes = [
        [('qe',['quant_evaluator/tests']),
         ('fa_main',['factor_assets/tests','-k','not test_concurrent_writers_same_hash_are_idempotent']),
         ('fa_concurrency',['factor_assets/tests/test_persistent_seen_index.py::TestPersistentSeenIndex::test_concurrent_writers_same_hash_are_idempotent'])],
        [('fo',['factor_optimizer/tests']),('fp',['factor_preprocess/tests']),
         ('platform',['quant_platform/tests'])],
        [('modeljobs',['tests/modeling','jobs/tests']),('vectorbt',['vectorbt_qs/tests'])],
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lane,lanes):
            results.update(result)
    after = check_sources()
    payload = dict(source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                   source_files_before=count,source_files_after=after,source_changed=False,
                   suites=results)
    (OUT/'execution.json').write_text(json.dumps(payload,indent=2)+'\n')
    assert all(row['returncode'] == 0 for row in results.values()), results
    print('ALL SUITES PASSED; FROZEN SOURCE UNCHANGED',flush=True)


if __name__ == '__main__':
    main()
