#!/usr/bin/env python3
"""Backend Parity Checker - Systematic verification across pandas/polars/duckdb/q.

This script identifies backend support gaps and parity issues for high-priority operators.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

# High-priority operators by category
HIGH_PRIORITY_OPERATORS = {
    "rolling_window": [
        "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max",
        "ts_median", "ts_rank", "ts_var", "ts_corr", "ts_cov"
    ],
    "technical_indicators": [
        "EMA", "SMA", "RSI", "MACD_line", "MACD_signal",
        "bollinger_upper", "bollinger_lower", "bollinger_mid"
    ],
    "statistical": [
        "rank", "zscore", "correlation", "covariance",
        "skew", "kurt", "quantile"
    ],
    "cross_sectional": [
        "cs_rank", "cs_zscore", "cs_mean", "cs_std",
        "cs_demean", "group_mean", "group_rank", "group_std"
    ]
}

BACKENDS = ["pandas_numpy", "polars", "sql", "q"]


def analyze_test_coverage() -> Dict:
    """Analyze existing parity test coverage."""
    parity_dir = Path("/home/shw/quant_projects/factor_engine/tests/backend_parity")

    test_files = {
        "three_backend": "test_three_backend_parity.py",
        "ts_family": "test_ts_family_systematic_parity.py",
        "cs_family": "test_cs_family_systematic_parity.py",
        "group_family": "test_group_family_systematic_parity.py",
        "elementwise": "test_elementwise_family_systematic_parity.py",
        "edge_cases": "test_edge_cases_comprehensive_parity.py",
        "production_core": "test_production_core_triple_parity.py",
        "p0_edge_cases": "test_p0_edge_cases_triple_parity.py",
    }

    coverage = {}
    for name, filename in test_files.items():
        file_path = parity_dir / filename
        if file_path.exists():
            content = file_path.read_text()
            # Count test functions
            test_count = content.count("def test_")
            # Count operators mentioned
            operators_found = set()
            for category, ops in HIGH_PRIORITY_OPERATORS.items():
                for op in ops:
                    if f'"{op}"' in content or f"'{op}'" in content:
                        operators_found.add(op)

            coverage[name] = {
                "file": str(file_path),
                "exists": True,
                "lines": len(content.splitlines()),
                "test_count": test_count,
                "operators_covered": sorted(operators_found),
                "operator_count": len(operators_found)
            }
        else:
            coverage[name] = {"exists": False, "file": str(file_path)}

    return coverage


def check_backend_documentation() -> Dict:
    """Check backend coverage documentation."""
    docs = {}

    # Check BACKEND_COVERAGE.md
    backend_doc = Path("/home/shw/quant_projects/factor_engine/BACKEND_COVERAGE.md")
    if backend_doc.exists():
        content = backend_doc.read_text()
        docs["backend_coverage"] = {
            "exists": True,
            "pandas_mentioned": "Pandas" in content,
            "polars_mentioned": "Polars" in content,
            "duckdb_mentioned": "DuckDB" in content,
            "q_mentioned": "q" in content or "kdb+" in content,
            "lines": len(content.splitlines())
        }

    # Check backend parity coverage report
    coverage_report = Path("/home/shw/quant_projects/factor_engine/tests/backend_parity/COVERAGE_REPORT.md")
    if coverage_report.exists():
        content = coverage_report.read_text()
        docs["parity_coverage_report"] = {
            "exists": True,
            "lines": len(content.splitlines()),
            "has_summary": "Summary" in content,
            "total_tests": content.count("test_")
        }

    return docs


def analyze_q_backend_capability() -> Dict:
    """Analyze q/kdb+ backend capability."""
    q_backend_dir = Path("/home/shw/quant_projects/factor_engine/backend/q_backend")
    q_tests_dir = Path("/home/shw/quant_projects/factor_engine/tests/q_backend")

    q_info = {
        "implementation_exists": q_backend_dir.exists(),
        "test_suite_exists": q_tests_dir.exists(),
        "files": {}
    }

    if q_backend_dir.exists():
        q_files = list(q_backend_dir.glob("*.py"))
        q_info["files"] = {
            f.name: {
                "lines": len(f.read_text().splitlines()),
                "has_capability": "capability" in f.name.lower(),
                "has_compiler": "compiler" in f.name.lower(),
            }
            for f in q_files if not f.name.startswith("__")
        }

    if q_tests_dir.exists():
        test_files = list(q_tests_dir.glob("test_*.py"))
        q_info["test_files"] = [f.name for f in test_files]

    return q_info


def check_known_divergences() -> List[Dict]:
    """Check for documented backend divergences."""
    divergences = []

    # Check memory documents for known issues
    memory_dir = Path("/home/shw/.claude/projects/-home-shw/memory")
    if memory_dir.exists():
        # Look for backend parity memory
        for mem_file in memory_dir.glob("*.md"):
            if "backend" in mem_file.name.lower() or "parity" in mem_file.name.lower():
                content = mem_file.read_text()
                if any(word in content.lower() for word in ["divergence", "mismatch", "差异", "不一致"]):
                    divergences.append({
                        "file": str(mem_file),
                        "name": mem_file.stem,
                        "has_divergence_mention": True
                    })

    return divergences


def main():
    """Generate comprehensive backend parity report."""

    print("=" * 100)
    print("BACKEND PARITY CHECKER REPORT")
    print("=" * 100)
    print()

    # 1. Test Coverage Analysis
    print("1. EXISTING PARITY TEST COVERAGE")
    print("-" * 100)
    coverage = analyze_test_coverage()

    total_operators_covered = set()
    for name, info in coverage.items():
        if info.get("exists"):
            print(f"\n{name}:")
            print(f"  File: {info['file']}")
            print(f"  Lines: {info['lines']}")
            print(f"  Test functions: {info['test_count']}")
            print(f"  Priority operators covered: {info['operator_count']}")
            if info['operators_covered']:
                print(f"  Operators: {', '.join(info['operators_covered'][:5])}...")
                total_operators_covered.update(info['operators_covered'])

    all_priority_ops = set()
    for ops in HIGH_PRIORITY_OPERATORS.values():
        all_priority_ops.update(ops)

    print(f"\n  TOTAL UNIQUE OPERATORS COVERED: {len(total_operators_covered)}/{len(all_priority_ops)}")
    uncovered = all_priority_ops - total_operators_covered
    if uncovered:
        print(f"  Uncovered operators: {', '.join(sorted(uncovered))}")

    # 2. Backend Documentation
    print("\n\n2. BACKEND DOCUMENTATION")
    print("-" * 100)
    docs = check_backend_documentation()
    for doc_name, info in docs.items():
        print(f"\n{doc_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 3. Q Backend Analysis
    print("\n\n3. Q/KDB+ BACKEND CAPABILITY")
    print("-" * 100)
    q_info = analyze_q_backend_capability()
    print(f"Implementation exists: {q_info['implementation_exists']}")
    print(f"Test suite exists: {q_info['test_suite_exists']}")
    if q_info.get('files'):
        print(f"\nImplementation files ({len(q_info['files'])}):")
        for fname, finfo in q_info['files'].items():
            print(f"  {fname}: {finfo['lines']} lines")
    if q_info.get('test_files'):
        print(f"\nTest files: {', '.join(q_info['test_files'])}")

    # 4. Known Divergences
    print("\n\n4. KNOWN DIVERGENCES/ISSUES")
    print("-" * 100)
    divergences = check_known_divergences()
    if divergences:
        for div in divergences:
            print(f"  {div['name']}: {div['file']}")
    else:
        print("  No documented divergences found in memory files")

    # 5. Recommendations
    print("\n\n5. RECOMMENDATIONS")
    print("-" * 100)
    print("  Priority actions:")
    print("  1. Fix registry issue (ts_mean_abs_deviation alias error)")
    print("  2. Run existing parity tests to identify failures")
    print(f"  3. Add parity tests for {len(uncovered)} uncovered operators")
    print("  4. Verify q backend has equivalent test coverage")
    print("  5. Document any known divergences in backend/contracts")

    # 6. Summary Statistics
    print("\n\n6. SUMMARY STATISTICS")
    print("-" * 100)
    print(f"  Total parity test files: {sum(1 for c in coverage.values() if c.get('exists'))}")
    print(f"  Total parity test functions: {sum(c.get('test_count', 0) for c in coverage.values())}")
    print(f"  Total lines of parity tests: {sum(c.get('lines', 0) for c in coverage.values())}")
    print(f"  Priority operators covered: {len(total_operators_covered)}/{len(all_priority_ops)} ({100*len(total_operators_covered)/len(all_priority_ops):.1f}%)")
    print(f"  Q backend files: {len(q_info.get('files', {}))}")
    print(f"  Q backend tests: {len(q_info.get('test_files', []))}")

    print("\n" + "=" * 100)
    print("REPORT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
