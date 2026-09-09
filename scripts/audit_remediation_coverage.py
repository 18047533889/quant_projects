"""Read-only content audit of historical remediation against current main.

Different content is not classified as lost: semantic merges require review.
Generated build trees, caches and historical raw data are not source candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CODE={'data_access','factor_assets','factor_engine','factor_optimizer','factor_preprocess',
      'quant_evaluator','quant_platform','vectorbt_qs','modeling','jobs','tests','scripts','loop','docs'}
EXT={'.py','.toml','.yaml','.yml','.json','.md','.txt','.sh','.csv','.in'}
SKIP={'.git','.venv','venv','build','dist','__pycache__','.pytest_cache','.mypy_cache',
      '.ruff_cache','.tox','node_modules','wheelhouse'}
ROOT_FILES={'build_unified_ledger.py','validate_unified_ledger.py','test_validate_unified_ledger.py'}

def git(root,*args,check=True):
    r=subprocess.run(['git','-C',str(root),*args],capture_output=True)
    if check and r.returncode: raise RuntimeError(f'git {args[0]} failed ({r.returncode})')
    return r.stdout if r.returncode==0 else None

def names(raw): return [p for p in raw.decode().split('\0') if p]
def digest(raw): return hashlib.sha256(raw).hexdigest() if raw is not None else None
def eligible(p):
    parts=Path(p).parts
    return ((parts[0] in CODE and Path(p).suffix in EXT) or p in ROOT_FILES) and not any(
        s in SKIP or s.endswith('.egg-info') for s in parts)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    target=Path(__file__).resolve().parents[1]
    source=args.source.resolve()
    source_head=git(source,'rev-parse','HEAD').decode().strip()
    target_head=git(target,'rev-parse','HEAD').decode().strip()
    changed=set(names(git(source,'diff','HEAD','--name-only','-z')))
    untracked=set(names(git(source,'ls-files','--others','--exclude-standard','-z')))
    ignored=set(names(git(source,'ls-files','--others','--ignored','--exclude-standard','-z','--','*.py','*.toml')))
    candidates=changed|{p for p in untracked|ignored if eligible(p)}
    rows=[]
    for p in sorted(candidates):
        path=source/p
        old=path.read_bytes() if path.is_file() else None
        new=(target/p).read_bytes() if (target/p).is_file() else None
        committed=git(target,'show',target_head+':'+p,check=False)
        base=git(source,'show',source_head+':'+p,check=False)
        if old is None:
            status='DELETION_PRESENT' if new is None and committed is None else 'REVIEW_DELETION'
        elif committed==old:
            status='EXACT_IN_MAIN_COMMIT'
        elif new==old:
            status='EXACT_IN_WORKTREE_ONLY'
        elif committed is None:
            status='MISSING_FROM_MAIN_COMMIT'
        elif base==committed and base!=old:
            status='MAIN_STILL_AT_SOURCE_BASE'
        else:
            status='DIFFERENT_REQUIRES_SEMANTIC_REVIEW'
        rows.append({'path':p,'status':status,'origin':'tracked_change' if p in changed else 'ignored_source' if p in ignored else 'untracked_source',
                     'source_sha256':digest(old),'main_commit_sha256':digest(committed),
                     'main_working_sha256':digest(new),'source_base_sha256':digest(base)})
    payload={'source':str(source),'source_head':source_head,'main':str(target),'main_head':target_head,
             'main_branch':git(target,'branch','--show-current').decode().strip(),
             'counts':{status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})},
             'excluded_untracked_or_ignored_count':len((untracked|ignored)-candidates),'files':rows,
             'caution':'Exact inclusion is byte-level proof. Differences need semantic review; ignored build products and raw evidence are excluded, never overwritten onto newer main.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload['counts']))
    for r in rows:
        if r['status']!='EXACT_IN_MAIN_COMMIT': print(r['status']+' '+r['path'])

if __name__=='__main__': main()
