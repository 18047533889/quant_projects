"""Reconcile closed evidence into a new review checkpoint; never overwrite."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from export_review import export_review, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='append', default=[], type=Path)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix('.manifest.json')
    validation_path = args.output.with_suffix('.validation.json')
    for target in (args.output, manifest_path, validation_path):
        if target.exists():
            raise FileExistsError(target)
    previous = json.loads(args.previous.read_text())
    smoke = list(dict.fromkeys([Path(p) for p in previous['smoke_evidence']] + args.smoke))
    compiled = [Path(p) for p in previous['compile_evidence']]
    # Fully consume gzip streams before export: no incomplete-prefix promotion.
    evidence = []
    for path in smoke + compiled:
        count = sum(1 for _ in records(path))
        evidence.append({'path': str(path), 'records': count,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    counts = export_review(args.source, args.audit, args.output, smoke, compiled)
    ids = set()
    row_count = 0
    with gzip.open(args.output, 'rt', encoding='utf-8-sig', newline='') as stream:
        for source, row in itertools.zip_longest(records(args.source), csv.DictReader(stream)):
            if source is None or row is None:
                raise ValueError('review/source row count mismatch')
            if (str(source['source_row']), source['id'], source['formula']) != (
                    row['source_row'], row['id'], row['original_formula']):
                raise ValueError('review/source identity or formula mismatch')
            if row['id']:
                if row['id'] in ids:
                    raise ValueError('duplicate ID')
                ids.add(row['id'])
            row_count += 1
    validation = {'rows': row_count, 'unique_factor_ids': len(ids),
                  'bytes': args.output.stat().st_size,
                  'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  'row_order_identity_original_formula_gzip_footer': 'PASS'}
    manifest = {'counts': counts, 'source': str(args.source), 'audit': str(args.audit),
                'smoke_evidence': [str(p) for p in smoke],
                'compile_evidence': [str(p) for p in compiled],
                'evidence_fingerprints': evidence,
                'scope': 'checkpoint_not_final_all_factor_execution',
                'validation_scope': 'historical_formula_matched_not_current_code_certification'}
    for path, data in ((manifest_path, manifest), (validation_path, validation)):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'counts': counts, 'validation': validation}, ensure_ascii=False))


if __name__ == '__main__':
    main()
