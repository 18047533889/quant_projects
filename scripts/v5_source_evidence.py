#!/usr/bin/env python3
"""Capture exact checkout evidence without committing or changing source files."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/v5/final_source'

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prefixes=('quant_evaluator','factor_assets','factor_optimizer','factor_preprocess',
              'quant_platform','vectorbt_qs','modeling','jobs','tests/modeling','scripts')
    names=git('ls-files','-z','--cached','--others','--exclude-standard','--',*prefixes).decode().split('\0')
    records=[]
    for name in sorted(set(names)):
        path=ROOT/name
        if not name or not path.is_file() or path.suffix not in {'.py','.toml','.yaml','.yml','.json'}:
            continue
        raw=path.read_bytes()
        records.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                        'git_blob':hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()})
    imports={}
    for name in ('quant_evaluator','factor_assets','factor_optimizer','factor_preprocess',
                 'quant_platform','vectorbt_qs','modeling','jobs'):
        module=importlib.import_module(name)
        imports[name]={'file':module.__file__,'paths':list(getattr(module,'__path__',()))}
    artifacts={}
    for name,raw in [('current_vs_HEAD.patch',git('diff','--binary','HEAD')),
                     ('status.txt',git('status','--short'))]:
        (OUT/name).write_bytes(raw)
        artifacts[name]=hashlib.sha256(raw).hexdigest()
    for name in ('quant_v5_baseline_diff_20260907.patch','quant_v5_baseline_status_20260907.txt'):
        raw=(Path('/tmp')/name).read_bytes()
        (OUT/name).write_bytes(raw)
        artifacts[name]=hashlib.sha256(raw).hexdigest()
    payload={'schema':'v5-source-evidence.v1','captured_at':datetime.now(timezone.utc).isoformat(),
        'root':str(ROOT),'head':git('rev-parse','HEAD').decode().strip(),
        'branch':git('branch','--show-current').decode().strip(),'imports':imports,
        'files':records,'artifact_sha256':artifacts,
        'scope_note':'Current patch includes preserved pre-existing V3 work; untracked sources are indexed by hash, not contained in git diff. Baseline patch/status retained; no clean V5-only patch is claimed.'}
    raw=json.dumps(payload,indent=2,sort_keys=True).encode()
    (OUT/'source_manifest.json').write_bytes(raw)
    print(json.dumps({'files':len(records),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
                      'output':str(OUT),'head':payload['head']},indent=2))

if __name__=='__main__': main()
