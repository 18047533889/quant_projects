"""Run isolated suite processes on the current checkout and verify source stability."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve()
PREFIXES={'quant_evaluator','factor_assets','factor_optimizer','factor_preprocess',
          'quant_platform','factor_engine','data_access','vectorbt_qs','modeling','jobs','tests','scripts','loop'}

def sources():
    raw=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
    paths=sorted(set(p for p in raw.decode().split('\0') if p and p.split('/')[0] in PREFIXES
                     and Path(p).suffix in {'.py','.toml','.yaml','.yml'}))
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if (ROOT/p).is_file()}

def lane(suites):
    results={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',POLARS_MAX_THREADS='2',
             PYTHONPATH=str(ROOT/'factor_optimizer')+':'+str(ROOT/'factor_preprocess')+':'+str(ROOT))
    for name,args in suites:
        local_env=dict(env)
        if name=='vectorbt': local_env['PYTHONPATH']='/tmp/vectorbt_qs_declared_deps_v6:'+env['PYTHONPATH']
        command=[sys.executable,'-m','pytest',*args,'-q','--timeout=300','--junitxml='+str(OUT/(name+'.xml'))]
        print('START '+name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:
            try:
                run=subprocess.run(command,cwd=ROOT,env=local_env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                returncode=run.returncode
            except subprocess.TimeoutExpired:
                returncode=124
        results[name]={'returncode':returncode,'command':command}
        print(f'FINISH {name}: {returncode}',flush=True)
    return results

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    before=sources()
    (OUT/'source_hashes.json').write_text(json.dumps(before,indent=2)+'\n')
    lanes=[
      [('qe',['quant_evaluator/tests']),
       ('fa_main',['factor_assets/tests','-k','not test_concurrent_writers_same_hash_are_idempotent']),
       ('fa_concurrency',['factor_assets/tests/test_persistent_seen_index.py::TestPersistentSeenIndex::test_concurrent_writers_same_hash_are_idempotent'])],
      [('fo',['factor_optimizer/tests']),('fp',['factor_preprocess/tests']),('platform',['quant_platform/tests'])],
      [('modeljobs',['tests/modeling','jobs/tests']),('vectorbt',['vectorbt_qs/tests']),
       ('integration_edges',['test_validate_unified_ledger.py',
         'factor_engine/tests/backend/test_polars_ts_rolling_pairwise.py',
         'factor_engine/tests/operators/test_filter_layer.py','factor_engine/tests/operators/test_same_clock_lag.py',
         'factor_engine/tests/operators/test_v9_butterworth_frequency.py',
         'factor_engine/tests/operators/test_v9_local_linear_endpoint.py',
         'data_access/tests/unit/test_r30_change_lineage_2026_08.py',
         'data_access/tests/unit/test_r32_change_impact_p0.py'])],
    ]
    results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lane,lanes): results.update(result)
    after=sources()
    counts={}
    for name,result in results.items():
        if not (OUT/(name+'.xml')).exists():
            counts[name]={'passed':0,'failed':None,'skipped_or_xfailed':0,'junit_available':False}
            continue
        cases=list(ET.parse(OUT/(name+'.xml')).getroot().iter('testcase'))
        bad=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
        skip=sum(c.find('skipped') is not None for c in cases)
        counts[name]={'passed':len(cases)-bad-skip,'failed':bad,'skipped_or_xfailed':skip}
    payload={'checkout':str(ROOT),'head_before_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
             'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
             'source_changed':before!=after,'source_files':len(before),'suites':results,'counts':counts}
    (OUT/'execution.json').write_text(json.dumps(payload,indent=2)+'\n')
    assert before==after,'source changed during validation'
    assert all(v['returncode']==0 for v in results.values()),results
    print(json.dumps(counts),flush=True)
    print('ALL MAIN CHECKOUT SUITES PASSED; SOURCE UNCHANGED',flush=True)

if __name__=='__main__': main()
