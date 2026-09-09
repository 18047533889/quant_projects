#!/usr/bin/env python3
"""Freeze V7 source identities without overwriting the historical V6 snapshot."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v8/baseline_source'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    prefixes = ('quant_evaluator', 'factor_assets', 'factor_optimizer', 'factor_preprocess',
                'quant_platform', 'factor_engine', 'vectorbt_qs', 'modeling', 'jobs', 'tests/modeling', 'scripts')
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', *prefixes, '.gitignore').decode().split('\0')
    baseline_path = ROOT / 'evidence/v7/final_source/source_manifest.json'
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
                        'v6_sha256': baseline.get(name),
                        'comparison_to_v6': ('not_previously_indexed' if name not in baseline else
                                             'unchanged' if baseline[name] == digest else 'changed'),
                        'changed_since_v6': name in baseline and baseline[name] != digest})
    imports = {}
    for name in ('quant_evaluator', 'factor_assets', 'factor_optimizer', 'factor_preprocess',
                 'quant_platform', 'factor_engine', 'vectorbt_qs', 'modeling', 'jobs'):
        module = importlib.import_module(name)
        imports[name] = {'file': module.__file__, 'paths': list(getattr(module, '__path__', ()))}
    artifacts = {}
    for name, raw in [('current_vs_HEAD.patch', git('diff', '--binary', 'HEAD')),
                      ('status.txt', git('status', '--short'))]:
        (OUT / name).write_bytes(raw)
        artifacts[name] = hashlib.sha256(raw).hexdigest()
    payload = {'schema': 'v7-source-evidence.v1', 'captured_at': datetime.now(timezone.utc).isoformat(),
               'root': str(ROOT), 'head': git('rev-parse', 'HEAD').decode().strip(),
               'branch': git('branch', '--show-current').decode().strip(), 'imports': imports,
               'files': records, 'artifact_sha256': artifacts,
               'v6_manifest_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
               'scope_note': 'Full diff preserves inherited V3/V6 work; it is not a V7-only patch. '
                             'Untracked source files are indexed separately. No production changes. '
                             'Factor-engine dependency source is also indexed in V7; absence from V6 '
                             'manifest is not evidence that an entire file was newly changed.'}
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    (OUT / 'source_manifest.json').write_bytes(raw)
    archive = OUT / 'source_snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for record in records:
            path = ROOT / record['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError('Source changed while archiving: ' + record['path'])
            bundle.add(path, arcname=record['path'], recursive=False)
    (OUT / 'source_snapshot.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '\n')
    print(json.dumps({'files': len(records), 'changed_since_v6': sum(r['changed_since_v6'] for r in records),
                      'not_previously_indexed': sum(r['comparison_to_v6'] == 'not_previously_indexed' for r in records),
                      'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
