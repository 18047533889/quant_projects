"""Current registry evidence, not mathematical certification; no production writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def source_record(target):
    try:
        path = inspect.getsourcefile(target)
        if path is None:
            return {"status": "SOURCE_UNAVAILABLE"}
        source = Path(path).resolve()
        return {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "symbol": getattr(target, '__qualname__', type(target).__qualname__)}
    except (TypeError, OSError):
        return {"status": "SOURCE_UNAVAILABLE"}


def declared_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': str(value)}
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): declared_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((declared_value(v) for v in value),
                      key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [declared_value(v) for v in value]
    if isinstance(value, type):
        return value.__module__ + '.' + value.__qualname__
    if is_dataclass(value):
        return {f.name: declared_value(getattr(value, f.name)) for f in fields(value)}
    return {'unserialized_type': type(value).__module__ + '.' + type(value).__qualname__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical-index', required=True)
    parser.add_argument('--previous-bindings', help='Optional previous binding JSONL, streamed for change accounting')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry as registry, _impl_source_hash, _contract_hash
    from factor_engine.backend.operator_semantic_version import semantic_version

    load_all()
    current = set(registry.list_canonical())
    records = []
    for canonical in sorted(current):
        aliases = sorted(k for k, v in registry._aliases.items() if v == canonical)
        bindings = []
        for backend, operator in sorted(registry._operators.get(canonical, {}).items()):
            methods = {}
            for name in ('calculate', '_calculate_series', '_fn'):
                method = getattr(operator, name, None)
                if callable(method):
                    methods[name] = source_record(method)
            bindings.append({
                'backend': backend, 'implementation_hash': _impl_source_hash(operator),
                'class': source_record(type(operator)), 'methods': methods,
                'contract_hash': _contract_hash(operator),
                'declared_metadata': declared_value(getattr(operator, 'metadata', None)),
                'verdict': 'NOT_RUN', 'execution_proof': 'NOT_RUN',
                'boundary': 'runtime winner binding only; source hashes are not numerical evidence',
            })
        records.append({'canonical': canonical, 'aliases': aliases,
                        'semantic_version': semantic_version(canonical), 'bindings': bindings,
                        'mathematical_verdict': 'NOT_RUN',
                        'override_sources': {
                            backend: list(chain) for (name, backend), chain in registry._override_chain.items()
                            if name == canonical},
                        'catalog_only': not bool(bindings)})
    assert len(records) == len(current) and {r['canonical'] for r in records} == current
    with open(args.historical_index, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'canonical' not in reader.fieldnames:
            raise ValueError('historical index lacks canonical column')
        historical = {row['canonical'] for row in reader}
    versions = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'polars', 'duckdb', 'pyarrow', 'arch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'NOT_INSTALLED'
    summary = {
        'schema_version': 'factor_engine.v7.current_binding_inventory.v1',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'created_at': datetime.now(timezone.utc).isoformat(), 'load_profile': 'load_all default',
        'current_canonicals': len(current), 'historical_canonicals': len(historical),
        'bindings': sum(len(r['bindings']) for r in records),
        'aliases': len(registry._aliases), 'added': sorted(current-historical),
        'removed_or_alias_only': sorted(historical-current),
        'alias_only': sorted(name for name in historical-current if name in registry._aliases),
        'versions': versions, 'all_current_canonicals_have_rows': True,
        'mathematical_certification': 'NOT_CLAIMED',
        'candidate_boundary': 'override lineage exported; unregistered candidates need separate source review',
    }
    if args.previous_bindings:
        previous = {}
        with open(args.previous_bindings, encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                canonical = row['canonical']
                if canonical in previous:
                    raise ValueError('duplicate canonical in previous bindings')
                previous[canonical] = {
                    b['backend']: b['implementation_hash'] for b in row['bindings']
                }
        changes = []
        for record in records:
            old = previous.get(record['canonical'])
            if old is None:
                continue
            before = old
            after = {b['backend']: b['implementation_hash'] for b in record['bindings']}
            for backend in sorted(set(before) | set(after)):
                if before.get(backend) != after.get(backend):
                    changes.append({'canonical': record['canonical'], 'backend': backend,
                                    'before': before.get(backend), 'after': after.get(backend),
                                    'prior_evidence': 'STALE_REQUIRES_REVALIDATION'})
        summary['implementation_changes_since_previous'] = len(changes)
        with open(args.previous_bindings, 'rb') as stream:
            summary['previous_bindings_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        (output/'implementation_changes.json').write_text(
            json.dumps(changes, sort_keys=True, indent=2, allow_nan=False))
    with (output/'bindings.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2))
    (output/'overwrite_history.json').write_text(json.dumps(list(registry._overwrite_log), default=str, indent=2))
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
