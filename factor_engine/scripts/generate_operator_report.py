#!/usr/bin/env python3
"""Generate operator contract report files."""

import json
import sys
from pathlib import Path

def load_operator_data():
    """Load the extracted operator mapping."""
    with open('/tmp/operator_mapping_clean.json', 'r') as f:
        return json.load(f)

def generate_summary_report(data):
    """Generate a comprehensive summary report."""
    stats = data['statistics']
    operators = data['operators']

    report = []
    report.append("=" * 80)
    report.append("OPERATOR COLD-START CONTRACT SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append(f"Total Operators: {stats['total_operators']}")
    report.append("")

    # By surface
    report.append("Operators by Surface:")
    for surface, count in sorted(stats['by_surface'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_operators'] * 100
        report.append(f"  {surface:15s}: {count:4d} ({pct:5.1f}%)")
    report.append("")

    # Evidence coverage
    report.append("Evidence Coverage:")
    report.append(f"  With evidence:    {stats['with_evidence']:4d} ({stats['with_evidence']/stats['total_operators']*100:5.1f}%)")
    report.append(f"  Without evidence: {stats['without_evidence']:4d} ({stats['without_evidence']/stats['total_operators']*100:5.1f}%)")
    report.append("")

    # Operators without evidence
    no_evidence = sorted([name for name, op in operators.items() if not op['has_evidence']])
    report.append(f"Operators without evidence ({len(no_evidence)}):")
    for i in range(0, len(no_evidence), 5):
        chunk = no_evidence[i:i+5]
        report.append("  " + ", ".join(chunk))
    report.append("")

    # Backend coverage by surface
    report.append("Backend Coverage by Surface:")
    for surface in ['daily', 'extended', 'research']:
        surface_ops = [op for op in operators.values() if op['surface'] == surface]
        if not surface_ops:
            continue

        pandas_count = sum(1 for op in surface_ops if op['backends'].get('pandas', {}).get('available', False))
        polars_count = sum(1 for op in surface_ops if op['backends'].get('polars', {}).get('available', False))
        duckdb_count = sum(1 for op in surface_ops if op['backends'].get('duckdb', {}).get('available', False))

        report.append(f"  {surface} ({len(surface_ops)} operators):")
        report.append(f"    pandas: {pandas_count:4d}")
        report.append(f"    polars: {polars_count:4d}")
        report.append(f"    duckdb: {duckdb_count:4d}")
    report.append("")

    # Daily operators with full 3-backend support
    daily_ops = [op for op in operators.values() if op['surface'] == 'daily']
    full_backend = [
        op['canonical'] for op in daily_ops
        if op['backends'].get('pandas', {}).get('available', False)
        and op['backends'].get('polars', {}).get('available', False)
        and op['backends'].get('duckdb', {}).get('available', False)
    ]
    report.append(f"Daily operators with full 3-backend support: {len(full_backend)}")
    report.append("")

    # Daily operators with partial backend support
    partial_backend = [
        op['canonical'] for op in daily_ops
        if op['has_evidence']
        and not (
            op['backends'].get('pandas', {}).get('available', False)
            and op['backends'].get('polars', {}).get('available', False)
            and op['backends'].get('duckdb', {}).get('available', False)
        )
    ]
    report.append(f"Daily operators with evidence but incomplete backend support: {len(partial_backend)}")
    if len(partial_backend) <= 50:
        for i in range(0, len(partial_backend), 5):
            chunk = partial_backend[i:i+5]
            report.append("  " + ", ".join(chunk))
    else:
        report.append(f"  (showing first 50)")
        for i in range(0, min(50, len(partial_backend)), 5):
            chunk = partial_backend[i:i+5]
            report.append("  " + ", ".join(chunk))
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)

def generate_csv_surface_mapping(data):
    """Generate CSV mapping operators to surfaces."""
    operators = data['operators']

    lines = ["canonical,surface,has_evidence"]
    for canonical, op in sorted(operators.items()):
        lines.append(f"{canonical},{op['surface']},{op['has_evidence']}")

    return "\n".join(lines)

def generate_csv_backend_capability(data):
    """Generate CSV with backend capabilities."""
    operators = data['operators']

    lines = ["canonical,surface,pandas_available,polars_available,duckdb_available,pandas_certified,polars_certified,duckdb_certified"]

    for canonical, op in sorted(operators.items()):
        pandas = op['backends'].get('pandas', {})
        polars = op['backends'].get('polars', {})
        duckdb = op['backends'].get('duckdb', {})

        lines.append(
            f"{canonical},"
            f"{op['surface']},"
            f"{pandas.get('available', False)},"
            f"{polars.get('available', False)},"
            f"{duckdb.get('available', False)},"
            f"{pandas.get('production_certified', False)},"
            f"{polars.get('production_certified', False)},"
            f"{duckdb.get('production_certified', False)}"
        )

    return "\n".join(lines)

def generate_json_contract_stub(data):
    """Generate JSON stub for operator contracts."""
    operators = data['operators']

    contracts = {}
    for canonical, op in sorted(operators.items()):
        contract = {
            "canonical": canonical,
            "surface": op['surface'],
            "has_evidence": op['has_evidence'],
            "backends": {
                "pandas": op['backends'].get('pandas', {}).get('available', False),
                "polars": op['backends'].get('polars', {}).get('available', False),
                "duckdb": op['backends'].get('duckdb', {}).get('available', False),
            },
            "certification": {
                "level": op.get('certification_level'),
                "pit_safe": op.get('pit_safe'),
                "checkpointable": op.get('checkpointable'),
            },
            "parameters": {
                "count": op.get('parameter_count', 0),
            }
        }
        contracts[canonical] = contract

    return json.dumps(contracts, indent=2)

def main():
    """Generate all report files."""
    print("Loading operator data...", file=sys.stderr)
    data = load_operator_data()

    output_dir = Path('/home/shw/quant_projects/factor_engine/operator_contracts')
    output_dir.mkdir(exist_ok=True)

    # Generate summary report
    print("Generating summary report...", file=sys.stderr)
    summary = generate_summary_report(data)
    with open(output_dir / 'operator_summary.txt', 'w') as f:
        f.write(summary)
    print(summary)

    # Generate CSV files
    print("\nGenerating CSV files...", file=sys.stderr)

    surface_csv = generate_csv_surface_mapping(data)
    with open(output_dir / 'operator_surface_mapping.csv', 'w') as f:
        f.write(surface_csv)
    print(f"  - operator_surface_mapping.csv ({len(surface_csv.splitlines())} lines)")

    backend_csv = generate_csv_backend_capability(data)
    with open(output_dir / 'operator_backend_capability.csv', 'w') as f:
        f.write(backend_csv)
    print(f"  - operator_backend_capability.csv ({len(backend_csv.splitlines())} lines)")

    # Generate JSON contract stub
    print("\nGenerating JSON contract stub...", file=sys.stderr)
    contract_json = generate_json_contract_stub(data)
    with open(output_dir / 'operator_contracts.json', 'w') as f:
        f.write(contract_json)
    print(f"  - operator_contracts.json")

    # Also save the full raw data
    with open(output_dir / 'operator_data_full.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  - operator_data_full.json")

    print(f"\nAll files saved to: {output_dir}/")

if __name__ == '__main__':
    main()
