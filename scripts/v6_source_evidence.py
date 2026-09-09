#!/usr/bin/env python3
"""Freeze V6 source identities without overwriting the historical V5 snapshot."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v6/final_source'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prefixes = ('quant_evaluator', 'factor_assets', 'factor_optimizer', 'factor_preprocess',
                'quant_platform', 'vectorbt_qs', 'modeling', 'jobs', 'tests/modeling', 'scripts')
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', *prefixes, '.gitignore').decode().split('\0')
    baseline_path = ROOT / 'evidence/v5/final_source/source_manifest.json'
    baseline_bytes = baseline_path.read_bytes()
    baseline = {row['path']: row['sha256'] for row in json.loads(baseline_bytes)['files']}
    records = []
    for name in sorted(set(names)):
        path = ROOT / name
        config_file = path.name == '.gitignore' or path.name.startswith('requirements') or name == 'quant_evaluator/docs/METRIC_CAPABILITY_MATRIX.csv'
        if not name or not path.is_file() or (path.suffix not in {'.py', '.toml', '.yaml', '.yml', '.json'} and not config_file):
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        records.append({'path': name, 'bytes': len(raw), 'sha256': digest,
                        'git_blob': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest(),
                        'v5_sha256': baseline.get(name), 'changed_since_v5': baseline.get(name) != digest})
    imports = {}
    for name in ('quant_evaluator', 'factor_assets', 'factor_optimizer', 'factor_preprocess',
                 'quant_platform', 'vectorbt_qs', 'modeling', 'jobs'):
        module = importlib.import_module(name)
        imports[name] = {'file': module.__file__, 'paths': list(getattr(module, '__path__', ()))}
    artifacts = {}
    for name, raw in [('current_vs_HEAD.patch', git('diff', '--binary', 'HEAD')),
                      ('status.txt', git('status', '--short'))]:
        (OUT / name).write_bytes(raw)
        artifacts[name] = hashlib.sha256(raw).hexdigest()
    payload = {'schema': 'v6-source-evidence.v1', 'captured_at': datetime.now(timezone.utc).isoformat(),
               'root': str(ROOT), 'head': git('rev-parse', 'HEAD').decode().strip(),
               'branch': git('branch', '--show-current').decode().strip(), 'imports': imports,
               'files': records, 'artifact_sha256': artifacts,
               'v5_manifest_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
               'scope_note': 'Full diff preserves inherited V3/V5 work; it is not a V6-only patch. '
                             'Untracked source files are indexed separately. No production changes.'}
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    (OUT / 'source_manifest.json').write_bytes(raw)
    print(json.dumps({'files': len(records), 'changed_since_v5': sum(r['changed_since_v5'] for r in records),
                      'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
