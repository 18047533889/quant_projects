#!/usr/bin/env python3
"""Extract operator surface definitions and certification evidence."""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def extract_operators_from_surface():
    """Extract all operator canonicals from operator_surface.py."""
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        DAILY_FACTOR_MIGRATED,
        UNSAFE_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
        LEGACY_ONLY_CANONICALS,
        INTERNAL_ONLY_CANONICALS,
        classify_canonical,
    )

    # Collect all operators
    all_operators = set()
    all_operators.update(DAILY_CANONICALS)
    all_operators.update(EXTENDED_ONLY_CANONICALS)
    all_operators.update(DAILY_FACTOR_MIGRATED)
    all_operators.update(UNSAFE_CANONICALS)
    all_operators.update(RESEARCH_ONLY_CANONICALS)
    all_operators.update(LEGACY_ONLY_CANONICALS)
    all_operators.update(INTERNAL_ONLY_CANONICALS)

    # Map each operator to its surface
    operator_surface_map = {}
    for op in all_operators:
        surface = classify_canonical(op)
        operator_surface_map[op] = surface

    return operator_surface_map

def parse_evidence_file(evidence_path):
    """Parse a certification evidence JSON file."""
    print(f"Parsing {evidence_path}...", file=sys.stderr)

    with open(evidence_path, 'r') as f:
        evidence_data = json.load(f)

    operator_evidence = {}

    # Extract operators from the 'operators' key
    operators = evidence_data.get('operators', {})

    # Extract backend parity lists
    polars_parity = set(evidence_data.get('polars_reference_parity', []))
    polars_edge = set(evidence_data.get('polars_edge_verified', []))
    duckdb_parity = set(evidence_data.get('duckdb_reference_parity', []))
    duckdb_edge = set(evidence_data.get('duckdb_edge_verified', []))
    no_fallback = set(evidence_data.get('no_fallback_verified', []))

    for canonical, data in operators.items():
        evidence = {
            'canonical': canonical,
            'backends': {},
            'certification_level': None,
            'pit_safe': None,
            'checkpointable': None,
            'parameter_count': 0,
            'has_evidence': True,
            'semantic_version': data.get('semantic_version'),
        }

        # Determine backend support from implementation hashes and parity lists
        has_pandas = bool(data.get('implementation_hash_pandas'))
        has_polars = bool(data.get('implementation_hash_polars'))
        has_duckdb = bool(data.get('implementation_hash_duckdb'))

        evidence['backends']['pandas'] = {
            'available': has_pandas,
            'production_certified': has_pandas,  # Pandas is reference
            'reference': True,
        }

        evidence['backends']['polars'] = {
            'available': has_polars,
            'production_certified': canonical in polars_parity,
            'reference': canonical in polars_parity,
            'edge_verified': canonical in polars_edge,
        }

        evidence['backends']['duckdb'] = {
            'available': has_duckdb,
            'production_certified': canonical in duckdb_parity,
            'reference': canonical in duckdb_parity,
            'edge_verified': canonical in duckdb_edge,
        }

        # Check for certified parameter domain
        if 'certified_parameter_domain' in data:
            param_domain = data['certified_parameter_domain']
            if isinstance(param_domain, dict):
                evidence['parameter_count'] = len(param_domain)

        operator_evidence[canonical] = evidence

    return operator_evidence

def main():
    """Generate comprehensive operator mapping."""
    print("Extracting operators from surface definitions...", file=sys.stderr)
    operator_surface_map = extract_operators_from_surface()

    print(f"Found {len(operator_surface_map)} unique operators", file=sys.stderr)

    # Parse evidence files
    evidence_dir = project_root / "evidence"
    primitive_evidence = {}
    factor_evidence = {}

    primitive_path = evidence_dir / "primitive_verified.json"
    factor_path = evidence_dir / "factor_operator_verified.json"

    if primitive_path.exists():
        primitive_evidence = parse_evidence_file(primitive_path)
        print(f"Loaded evidence for {len(primitive_evidence)} primitive operators", file=sys.stderr)

    if factor_path.exists():
        factor_evidence = parse_evidence_file(factor_path)
        print(f"Loaded evidence for {len(factor_evidence)} factor operators", file=sys.stderr)

    # Combine evidence
    all_evidence = {**primitive_evidence, **factor_evidence}

    # Create comprehensive mapping
    comprehensive_map = {}

    for canonical, surface in sorted(operator_surface_map.items()):
        entry = {
            'canonical': canonical,
            'surface': surface,
            'has_evidence': canonical in all_evidence,
        }

        if canonical in all_evidence:
            evidence = all_evidence[canonical]
            entry.update({
                'backends': evidence['backends'],
                'certification_level': evidence['certification_level'],
                'pit_safe': evidence['pit_safe'],
                'checkpointable': evidence['checkpointable'],
                'parameter_count': evidence['parameter_count'],
            })
        else:
            entry.update({
                'backends': {},
                'certification_level': None,
                'pit_safe': None,
                'checkpointable': None,
                'parameter_count': 0,
            })

        comprehensive_map[canonical] = entry

    # Generate statistics
    stats = {
        'total_operators': len(comprehensive_map),
        'by_surface': defaultdict(int),
        'with_evidence': 0,
        'without_evidence': 0,
        'by_backend': {
            'pandas': {'available': 0, 'certified': 0},
            'polars': {'available': 0, 'certified': 0},
            'duckdb': {'available': 0, 'certified': 0},
        },
        'pit_safe_count': 0,
        'checkpointable_count': 0,
    }

    for canonical, entry in comprehensive_map.items():
        stats['by_surface'][entry['surface']] += 1

        if entry['has_evidence']:
            stats['with_evidence'] += 1
        else:
            stats['without_evidence'] += 1

        if entry['pit_safe']:
            stats['pit_safe_count'] += 1

        if entry['checkpointable']:
            stats['checkpointable_count'] += 1

        for backend in ['pandas', 'polars', 'duckdb']:
            if backend in entry['backends']:
                if entry['backends'][backend]['available']:
                    stats['by_backend'][backend]['available'] += 1
                if entry['backends'][backend]['production_certified']:
                    stats['by_backend'][backend]['certified'] += 1

    # Output results
    output = {
        'statistics': stats,
        'operators': comprehensive_map,
    }

    print(json.dumps(output, indent=2))

    # Print summary to stderr
    print("\n" + "="*60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Total operators: {stats['total_operators']}", file=sys.stderr)
    print(f"\nBy surface:", file=sys.stderr)
    for surface, count in sorted(stats['by_surface'].items()):
        print(f"  {surface:15s}: {count:4d}", file=sys.stderr)
    print(f"\nEvidence coverage:", file=sys.stderr)
    print(f"  With evidence:    {stats['with_evidence']:4d}", file=sys.stderr)
    print(f"  Without evidence: {stats['without_evidence']:4d}", file=sys.stderr)
    print(f"\nBackend support:", file=sys.stderr)
    for backend in ['pandas', 'polars', 'duckdb']:
        print(f"  {backend}:", file=sys.stderr)
        print(f"    Available: {stats['by_backend'][backend]['available']:4d}", file=sys.stderr)
        print(f"    Certified: {stats['by_backend'][backend]['certified']:4d}", file=sys.stderr)
    print(f"\nOperator properties:", file=sys.stderr)
    print(f"  PIT safe:        {stats['pit_safe_count']:4d}", file=sys.stderr)
    print(f"  Checkpointable:  {stats['checkpointable_count']:4d}", file=sys.stderr)
    print("="*60, file=sys.stderr)

if __name__ == '__main__':
    main()
