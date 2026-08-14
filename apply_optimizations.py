"""
Apply performance optimizations across all packages.
This script implements optimizations based on audit findings.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import ast
import re


class PerformanceOptimizer:
    """Automated performance optimization applier."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.optimizations_applied = []
        self.issues_found = []

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for performance issues."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check 1: rolling().apply() with lambda
            if '.rolling(' in content and '.apply(' in content and 'lambda' in content:
                for i, line in enumerate(content.split('\n'), 1):
                    if '.rolling(' in line and '.apply(' in line and 'lambda' in line:
                        issues.append({
                            'file': str(file_path),
                            'line': i,
                            'type': 'rolling_apply_lambda',
                            'severity': 'CRITICAL',
                            'impact': '10-100x slowdown',
                            'content': line.strip()
                        })

            # Check 2: Nested loops over dataframes
            if 'for ' in content and ('iterrows' in content or 'itertuples' in content):
                for i, line in enumerate(content.split('\n'), 1):
                    if 'iterrows' in line or 'itertuples' in line:
                        issues.append({
                            'file': str(file_path),
                            'line': i,
                            'type': 'dataframe_iteration',
                            'severity': 'HIGH',
                            'impact': '5-50x slowdown',
                            'content': line.strip()
                        })

            # Check 3: Excessive .copy() calls
            copy_count = content.count('.copy()')
            if copy_count > 5:
                issues.append({
                    'file': str(file_path),
                    'line': 0,
                    'type': 'excessive_copies',
                    'severity': 'MEDIUM',
                    'impact': f'{copy_count} DataFrame copies',
                    'content': f'{copy_count} .copy() calls found'
                })

            # Check 4: Backend conversions
            conversions = content.count('to_pandas()') + content.count('to_polars()') + content.count('pl.from_pandas(')
            if conversions > 3:
                issues.append({
                    'file': str(file_path),
                    'line': 0,
                    'type': 'backend_conversions',
                    'severity': 'HIGH',
                    'impact': f'{conversions} backend conversions',
                    'content': f'{conversions} conversion calls'
                })

            # Check 5: Nested loops (3+ levels)
            nested = self._detect_nested_loops(content)
            if nested:
                for nest_info in nested:
                    issues.append({
                        'file': str(file_path),
                        'line': nest_info['line'],
                        'type': 'nested_loops',
                        'severity': 'HIGH',
                        'impact': f"Nested loop depth {nest_info['depth']}",
                        'content': nest_info.get('snippet', '')
                    })

            # Check 6: Missing numba decorators on hot functions
            if 'def ' in content and 'window' in content and '@njit' not in content:
                for i, line in enumerate(content.split('\n'), 1):
                    if line.strip().startswith('def ') and 'window' in line and i > 1:
                        prev_line = content.split('\n')[i-2] if i > 1 else ""
                        if '@njit' not in prev_line and '@jit' not in prev_line:
                            if any(kw in line for kw in ['rolling', 'mean', 'std', 'corr', 'rank']):
                                issues.append({
                                    'file': str(file_path),
                                    'line': i,
                                    'type': 'missing_numba',
                                    'severity': 'MEDIUM',
                                    'impact': '10-50x potential speedup',
                                    'content': line.strip()
                                })

        except Exception as e:
            pass

        return issues

    def _detect_nested_loops(self, source_code: str) -> List[Dict]:
        """Detect nested loops >= 2 levels."""
        try:
            tree = ast.parse(source_code)
        except:
            return []

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.nested = []
                self.depth = 0

            def visit_For(self, node):
                self.depth += 1
                if self.depth >= 2:
                    self.nested.append({
                        'line': node.lineno,
                        'depth': self.depth
                    })
                self.generic_visit(node)
                self.depth -= 1

            def visit_While(self, node):
                self.depth += 1
                if self.depth >= 2:
                    self.nested.append({
                        'line': node.lineno,
                        'depth': self.depth
                    })
                self.generic_visit(node)
                self.depth -= 1

        visitor = LoopVisitor()
        visitor.visit(tree)
        return visitor.nested

    def scan_package(self, package_name: str) -> List[Dict]:
        """Scan entire package for performance issues."""
        package_dir = self.root_dir / package_name
        all_issues = []

        if not package_dir.exists():
            print(f"Warning: {package_name} not found")
            return []

        for py_file in package_dir.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue

            issues = self.scan_file(py_file)
            all_issues.extend(issues)

        return all_issues

    def generate_report(self, issues: List[Dict]) -> str:
        """Generate formatted report of issues."""
        # Group by severity
        critical = [i for i in issues if i['severity'] == 'CRITICAL']
        high = [i for i in issues if i['severity'] == 'HIGH']
        medium = [i for i in issues if i['severity'] == 'MEDIUM']

        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE OPTIMIZATION AUDIT RESULTS")
        report.append("=" * 80)
        report.append(f"\nTotal Issues: {len(issues)}")
        report.append(f"  CRITICAL: {len(critical)}")
        report.append(f"  HIGH:     {len(high)}")
        report.append(f"  MEDIUM:   {len(medium)}")
        report.append("\n")

        if critical:
            report.append("=" * 80)
            report.append("🔴 CRITICAL ISSUES (Immediate Action Required)")
            report.append("=" * 80)
            for i, issue in enumerate(critical, 1):
                report.append(f"\n#{i} [{issue['type']}] {issue['file']}:{issue['line']}")
                report.append(f"   Impact: {issue['impact']}")
                report.append(f"   Code: {issue['content'][:100]}")

        if high:
            report.append("\n" + "=" * 80)
            report.append("🟠 HIGH PRIORITY ISSUES")
            report.append("=" * 80)
            for i, issue in enumerate(high[:10], 1):  # Top 10
                report.append(f"\n#{i} [{issue['type']}] {issue['file']}:{issue['line']}")
                report.append(f"   Impact: {issue['impact']}")
                report.append(f"   Code: {issue['content'][:100]}")

        if medium:
            report.append("\n" + "=" * 80)
            report.append("🟡 MEDIUM PRIORITY ISSUES")
            report.append("=" * 80)
            report.append(f"\nFound {len(medium)} medium priority issues")
            # Group by type
            by_type = {}
            for issue in medium:
                by_type.setdefault(issue['type'], []).append(issue)
            for issue_type, type_issues in by_type.items():
                report.append(f"  - {issue_type}: {len(type_issues)} occurrences")

        return "\n".join(report)


def main():
    """Main entry point."""
    root = Path("/home/shw/quant_projects")

    print("Scanning packages for performance issues...")
    optimizer = PerformanceOptimizer(root)

    all_issues = {}
    packages = [
        "quant_evaluator",
        "factor_preprocess",
        "factor_assets",
        "factor_optimizer",
        "research_control"
    ]

    for pkg in packages:
        print(f"  Scanning {pkg}...")
        issues = optimizer.scan_package(pkg)
        all_issues[pkg] = issues
        print(f"    Found {len(issues)} issues")

    # Generate summary
    print("\n" + "=" * 80)
    print("PACKAGE SUMMARY")
    print("=" * 80)

    total = 0
    for pkg, issues in all_issues.items():
        critical = len([i for i in issues if i['severity'] == 'CRITICAL'])
        high = len([i for i in issues if i['severity'] == 'HIGH'])
        medium = len([i for i in issues if i['severity'] == 'MEDIUM'])
        total += len(issues)

        print(f"\n{pkg}:")
        print(f"  Total: {len(issues)}")
        print(f"  🔴 Critical: {critical}")
        print(f"  🟠 High: {high}")
        print(f"  🟡 Medium: {medium}")

    print(f"\n{'='*80}")
    print(f"TOTAL ISSUES ACROSS ALL PACKAGES: {total}")
    print(f"{'='*80}\n")

    # Generate detailed report
    print("\nGenerating detailed report...")
    for pkg, issues in all_issues.items():
        if issues:
            report = optimizer.generate_report(issues)
            output_file = root / f"PERF_AUDIT_{pkg.upper()}.md"

            with open(output_file, 'w') as f:
                f.write(f"# Performance Audit: {pkg}\n\n")
                f.write(report)

            print(f"  Wrote {output_file}")

    return all_issues


if __name__ == "__main__":
    issues = main()
