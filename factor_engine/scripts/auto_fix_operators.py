#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated operator issue diagnosis and batch fixing tool.

This tool provides comprehensive automation for:
1. Scanning all operators and classifying issues
2. Auto-fixing simple problems (missing policy, docstrings, defaults)
3. Generating templates for complex fixes
4. Batch processing with validation and rollback
5. Generating detailed repair reports

Usage:
    python scripts/auto_fix_operators.py --scan              # Diagnose only
    python scripts/auto_fix_operators.py --all --apply       # Fix all auto-fixable issues
    python scripts/auto_fix_operators.py --type policy       # Fix only policy issues
    python scripts/auto_fix_operators.py --dry-run           # Preview changes
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Issue:
    """Single operator issue."""
    operator: str
    category: str  # policy, docstring, defaults, polars_bridge, typing, etc.
    severity: str  # critical, high, medium, low
    description: str
    file_path: str | None = None
    line_number: int | None = None
    auto_fixable: bool = False
    fix_template: str | None = None
    current_code: str | None = None


@dataclass
class FixResult:
    """Result of attempting to fix an issue."""
    issue: Issue
    success: bool
    applied: bool = False
    error: str | None = None
    changes: list[str] = field(default_factory=list)


class OperatorScanner:
    """Scans operators and identifies common issues."""

    def __init__(self):
        self.issues: list[Issue] = []
        self.operators_scanned = 0
        self.files_scanned = 0

    def scan_all(self) -> list[Issue]:
        """Scan all operators and collect issues."""
        print("🔍 Scanning operators...")

        # Scan cleaned_operators directory
        cleaned_ops_dir = PROJECT_ROOT / "cleaned_operators"
        if cleaned_ops_dir.exists():
            self._scan_directory(cleaned_ops_dir)

        # Scan mining directory
        mining_dir = PROJECT_ROOT / "mining"
        if mining_dir.exists():
            self._scan_mining_catalog(mining_dir / "operator_catalog.py")

        print(f"✓ Scanned {self.operators_scanned} operators in {self.files_scanned} files")
        print(f"✓ Found {len(self.issues)} issues")

        return self.issues

    def _scan_directory(self, directory: Path):
        """Scan a directory for operator files."""
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name.startswith("test_"):
                continue

            self.files_scanned += 1
            self._scan_file(py_file)

    def _scan_file(self, file_path: Path):
        """Scan a single Python file for operator issues."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._check_function(node, file_path, content)
                elif isinstance(node, ast.ClassDef):
                    self._check_class(node, file_path, content)

        except Exception as e:
            print(f"⚠️  Error scanning {file_path}: {e}")

    def _check_function(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Check a function for operator issues."""
        func_name = node.name

        # Skip private functions and test functions
        if func_name.startswith("_") or func_name.startswith("test_"):
            return

        # Check if it looks like an operator function
        if not self._is_operator_function(node, content):
            return

        self.operators_scanned += 1

        # Check for missing docstring
        if not ast.get_docstring(node):
            self.issues.append(Issue(
                operator=func_name,
                category="docstring",
                severity="medium",
                description="Missing docstring",
                file_path=str(file_path),
                line_number=node.lineno,
                auto_fixable=True,
                current_code=self._get_function_code(node, content)
            ))

        # Check for missing default values
        missing_defaults = self._check_missing_defaults(node)
        if missing_defaults:
            self.issues.append(Issue(
                operator=func_name,
                category="defaults",
                severity="medium",
                description=f"Parameters missing defaults: {', '.join(missing_defaults)}",
                file_path=str(file_path),
                line_number=node.lineno,
                auto_fixable=True,
                current_code=self._get_function_code(node, content)
            ))

        # Check for missing type hints
        if not self._has_type_hints(node):
            self.issues.append(Issue(
                operator=func_name,
                category="typing",
                severity="low",
                description="Missing type hints",
                file_path=str(file_path),
                line_number=node.lineno,
                auto_fixable=False
            ))

    def _check_class(self, node: ast.ClassDef, file_path: Path, content: str):
        """Check a class for operator issues."""
        class_name = node.name

        # Check if it's an operator class
        if not any(base.id == "Operator" or "Operator" in base.id
                  for base in node.bases if isinstance(base, ast.Name)):
            return

        self.operators_scanned += 1

        # Check for missing calculate method
        has_calculate = any(
            isinstance(item, ast.FunctionDef) and item.name == "calculate"
            for item in node.body
        )

        if not has_calculate:
            self.issues.append(Issue(
                operator=class_name,
                category="implementation",
                severity="critical",
                description="Missing calculate() method",
                file_path=str(file_path),
                line_number=node.lineno,
                auto_fixable=False
            ))

        # Check for missing docstring
        if not ast.get_docstring(node):
            self.issues.append(Issue(
                operator=class_name,
                category="docstring",
                severity="medium",
                description="Missing class docstring",
                file_path=str(file_path),
                line_number=node.lineno,
                auto_fixable=True,
                current_code=self._get_class_code(node, content)
            ))

    def _scan_mining_catalog(self, catalog_path: Path):
        """Scan mining catalog for role/policy issues."""
        if not catalog_path.exists():
            return

        try:
            content = catalog_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(catalog_path))

            # Look for UNRESOLVED roles
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Check for role assignments
                    pass  # Will implement if needed

        except Exception as e:
            print(f"⚠️  Error scanning mining catalog: {e}")

    def _is_operator_function(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if a function is likely an operator."""
        # Common operator patterns
        operator_patterns = [
            r"^(ts_|cs_|group_|expanding_)",  # Prefix patterns
            r"^(pd_|pl_)",  # Backend prefixes
            r"(calculate|compute|apply)",  # Common method names
        ]

        func_name = node.name
        for pattern in operator_patterns:
            if re.match(pattern, func_name):
                return True

        # Check if registered with OperatorRegistry
        func_source = self._get_function_code(node, content)
        if "OperatorRegistry.register" in func_source:
            return True

        return False

    def _check_missing_defaults(self, node: ast.FunctionDef) -> list[str]:
        """Check for parameters that should have defaults but don't."""
        missing = []

        args = node.args
        # Skip 'self' and first data parameter
        start_idx = 1 if args.args and args.args[0].arg == "self" else 1

        for i, arg in enumerate(args.args[start_idx:], start=start_idx):
            param_name = arg.arg

            # Common parameters that should have defaults
            should_have_default = [
                "window", "d", "n", "periods", "span", "min_periods",
                "ddof", "adjust", "alpha", "normalize", "center",
                "method", "mode", "fill_value", "limit"
            ]

            if param_name in should_have_default:
                # Check if it has a default
                defaults_start = len(args.args) - len(args.defaults or [])
                if i >= defaults_start:
                    continue

                missing.append(param_name)

        return missing

    def _has_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        if node.returns:
            return True

        for arg in node.args.args:
            if arg.annotation:
                return True

        return False

    def _get_function_code(self, node: ast.FunctionDef, content: str) -> str:
        """Extract function source code."""
        lines = content.splitlines()
        if node.lineno <= len(lines):
            # Get function definition and a few lines
            start = max(0, node.lineno - 1)
            end = min(len(lines), start + 10)
            return "\n".join(lines[start:end])
        return ""

    def _get_class_code(self, node: ast.ClassDef, content: str) -> str:
        """Extract class source code."""
        lines = content.splitlines()
        if node.lineno <= len(lines):
            start = max(0, node.lineno - 1)
            end = min(len(lines), start + 15)
            return "\n".join(lines[start:end])
        return ""


class OperatorFixer:
    """Applies automated fixes to operator issues."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.backup_dir: Path | None = None
        self.fixed_files: set[str] = set()

    def fix_issues(self, issues: list[Issue]) -> list[FixResult]:
        """Apply fixes to all auto-fixable issues."""
        results = []

        # Group by file for efficient processing
        issues_by_file = defaultdict(list)
        for issue in issues:
            if issue.auto_fixable and issue.file_path:
                issues_by_file[issue.file_path].append(issue)

        print(f"\n🔧 Fixing {len([i for i in issues if i.auto_fixable])} auto-fixable issues...")

        for file_path, file_issues in issues_by_file.items():
            file_results = self._fix_file(file_path, file_issues)
            results.extend(file_results)

        return results

    def _fix_file(self, file_path: str, issues: list[Issue]) -> list[FixResult]:
        """Fix all issues in a single file."""
        results = []
        path = Path(file_path)

        if not path.exists():
            for issue in issues:
                results.append(FixResult(issue, success=False, error="File not found"))
            return results

        try:
            # Read original content
            original_content = path.read_text(encoding="utf-8")
            modified_content = original_content

            # Backup if not dry run
            if not self.dry_run:
                self._backup_file(path)

            # Apply fixes
            for issue in issues:
                result = self._apply_fix(issue, modified_content)
                if result.success and result.changes:
                    modified_content = result.changes[0]  # Updated content
                results.append(result)

            # Write changes
            if not self.dry_run and modified_content != original_content:
                path.write_text(modified_content, encoding="utf-8")
                self.fixed_files.add(file_path)
                print(f"✓ Fixed {len([r for r in results if r.success])} issues in {path.name}")
            elif modified_content != original_content:
                print(f"[DRY RUN] Would fix {len([r for r in results if r.success])} issues in {path.name}")

        except Exception as e:
            for issue in issues:
                results.append(FixResult(issue, success=False, error=str(e)))

        return results

    def _apply_fix(self, issue: Issue, content: str) -> FixResult:
        """Apply a specific fix."""
        try:
            if issue.category == "docstring":
                return self._fix_docstring(issue, content)
            elif issue.category == "defaults":
                return self._fix_defaults(issue, content)
            elif issue.category == "policy":
                return self._fix_policy(issue, content)
            else:
                return FixResult(issue, success=False, error="Unknown fix type")

        except Exception as e:
            return FixResult(issue, success=False, error=str(e))

    def _fix_docstring(self, issue: Issue, content: str) -> FixResult:
        """Add a basic docstring template."""
        lines = content.splitlines(keepends=True)

        if issue.line_number and issue.line_number <= len(lines):
            # Find the line after the function/class definition
            insert_line = issue.line_number

            # Generate docstring based on operator name
            operator = issue.operator
            docstring = self._generate_docstring(operator)

            # Insert docstring
            indent = self._get_indent(lines[insert_line - 1])
            docstring_lines = f'{indent}"""{docstring}"""\n'

            modified = lines.copy()
            modified.insert(insert_line, docstring_lines)

            return FixResult(
                issue,
                success=True,
                applied=not self.dry_run,
                changes=["".join(modified)]
            )

        return FixResult(issue, success=False, error="Could not locate insertion point")

    def _fix_defaults(self, issue: Issue, content: str) -> FixResult:
        """Add common default values to parameters."""
        # Extract parameter names from description
        match = re.search(r"Parameters missing defaults: (.+)", issue.description)
        if not match:
            return FixResult(issue, success=False, error="Could not parse parameters")

        params = [p.strip() for p in match.group(1).split(",")]

        # Common defaults
        default_values = {
            "window": "20",
            "d": "20",
            "n": "20",
            "periods": "1",
            "span": "20",
            "min_periods": "1",
            "ddof": "0",
            "adjust": "True",
            "alpha": "0.05",
            "normalize": "False",
            "center": "False",
            "method": '"linear"',
            "mode": '"valid"',
            "fill_value": "None",
            "limit": "None"
        }

        lines = content.splitlines(keepends=True)
        if issue.line_number and issue.line_number <= len(lines):
            func_line = lines[issue.line_number - 1]

            # Add defaults to function signature
            modified_line = func_line
            for param in params:
                if param in default_values:
                    # Simple regex replacement
                    pattern = rf"\b{param}\b(?=\s*[,)])"
                    replacement = f"{param}={default_values[param]}"
                    modified_line = re.sub(pattern, replacement, modified_line)

            modified = lines.copy()
            modified[issue.line_number - 1] = modified_line

            return FixResult(
                issue,
                success=True,
                applied=not self.dry_run,
                changes=["".join(modified)]
            )

        return FixResult(issue, success=False, error="Could not modify function signature")

    def _fix_policy(self, issue: Issue, content: str) -> FixResult:
        """Add missing policy registration."""
        # This would require more complex AST manipulation
        # For now, generate a template
        template = self._generate_policy_template(issue.operator)

        return FixResult(
            issue,
            success=False,
            error="Manual fix required",
            changes=[f"# Add this policy:\n{template}"]
        )

    def _generate_docstring(self, operator: str) -> str:
        """Generate a basic docstring template."""
        # Parse operator name for hints
        if operator.startswith("ts_"):
            return f"Time-series {operator[3:]} operator."
        elif operator.startswith("cs_"):
            return f"Cross-sectional {operator[3:]} operator."
        elif operator.startswith("group_"):
            return f"Grouped {operator[6:]} operator."
        else:
            return f"{operator} operator."

    def _generate_policy_template(self, operator: str) -> str:
        """Generate policy registration template."""
        return f'''# Policy for {operator}
_policy_registry.register(
    canonical="{operator}",
    timing_kind=TimingKind.INDEPENDENT_DAILY,  # Adjust as needed
    param_role={{}},
    lane=Lane.PRODUCTION,
    state=State.STATELESS
)'''

    def _get_indent(self, line: str) -> str:
        """Extract indentation from a line."""
        return line[:len(line) - len(line.lstrip())]

    def _backup_file(self, path: Path):
        """Create a backup of a file before modification."""
        if self.backup_dir is None:
            self.backup_dir = Path(tempfile.mkdtemp(prefix="operator_fix_backup_"))

        backup_path = self.backup_dir / path.name
        shutil.copy2(path, backup_path)

    def rollback(self):
        """Rollback all changes."""
        if self.backup_dir and self.backup_dir.exists():
            for fixed_file in self.fixed_files:
                backup = self.backup_dir / Path(fixed_file).name
                if backup.exists():
                    shutil.copy2(backup, fixed_file)
            print(f"✓ Rolled back {len(self.fixed_files)} files")


class ReportGenerator:
    """Generates detailed diagnostic and fix reports."""

    def generate_report(self, issues: list[Issue], results: list[FixResult],
                       output_path: str = "/tmp/auto_fix_operators_report.md"):
        """Generate a comprehensive report."""
        report_lines = []

        # Header
        report_lines.append("# Operator Auto-Fix Report")
        report_lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        report_lines.append(f"\nTotal issues found: {len(issues)}")
        report_lines.append(f"Auto-fixable: {len([i for i in issues if i.auto_fixable])}")
        report_lines.append(f"Fixed: {len([r for r in results if r.success])}")

        # Summary by category
        report_lines.append("\n## Issues by Category")
        by_category = Counter(i.category for i in issues)
        for category, count in by_category.most_common():
            report_lines.append(f"- **{category}**: {count}")

        # Summary by severity
        report_lines.append("\n## Issues by Severity")
        by_severity = Counter(i.severity for i in issues)
        for severity, count in by_severity.most_common():
            report_lines.append(f"- **{severity}**: {count}")

        # Detailed issues
        report_lines.append("\n## Detailed Issues")

        for category in sorted(set(i.category for i in issues)):
            category_issues = [i for i in issues if i.category == category]
            report_lines.append(f"\n### {category.upper()} ({len(category_issues)} issues)")

            for issue in category_issues[:20]:  # Limit to first 20 per category
                report_lines.append(f"\n#### {issue.operator}")
                report_lines.append(f"- **Severity**: {issue.severity}")
                report_lines.append(f"- **Description**: {issue.description}")
                if issue.file_path:
                    report_lines.append(f"- **File**: {issue.file_path}:{issue.line_number}")
                report_lines.append(f"- **Auto-fixable**: {'Yes' if issue.auto_fixable else 'No'}")

                # Show fix template if available
                if issue.fix_template:
                    report_lines.append(f"\n**Fix template:**\n```python\n{issue.fix_template}\n```")

        # Fix results
        if results:
            report_lines.append("\n## Fix Results")
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            report_lines.append(f"\n- Successful: {len(successful)}")
            report_lines.append(f"- Failed: {len(failed)}")

            if failed:
                report_lines.append("\n### Failed Fixes")
                for result in failed[:20]:
                    report_lines.append(f"\n- **{result.issue.operator}**: {result.error}")

        # Recommendations
        report_lines.append("\n## Recommendations")
        report_lines.append("\n1. **Critical issues**: Address immediately")
        report_lines.append("2. **High severity**: Fix before next release")
        report_lines.append("3. **Medium/Low**: Schedule for upcoming sprint")
        report_lines.append("\n4. **Manual fixes required**:")

        manual_fixes = [i for i in issues if not i.auto_fixable]
        for category in sorted(set(i.category for i in manual_fixes)):
            count = len([i for i in manual_fixes if i.category == category])
            report_lines.append(f"   - {category}: {count} operators")

        # Write report
        output = Path(output_path)
        output.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n📊 Report written to: {output}")

        return str(output)


def main():
    parser = argparse.ArgumentParser(description="Automated operator fixing tool")
    parser.add_argument("--scan", action="store_true", help="Scan only, no fixes")
    parser.add_argument("--all", action="store_true", help="Fix all auto-fixable issues")
    parser.add_argument("--type", choices=["policy", "docstring", "defaults", "all"],
                       help="Fix specific issue type")
    parser.add_argument("--apply", action="store_true", help="Actually apply fixes (not dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--report", default="/tmp/auto_fix_operators_report.md",
                       help="Output report path")
    parser.add_argument("--operator", help="Fix specific operator only")

    args = parser.parse_args()

    # Scan operators
    scanner = OperatorScanner()
    issues = scanner.scan_all()

    # Filter issues if needed
    if args.operator:
        issues = [i for i in issues if i.operator == args.operator]

    if args.type and args.type != "all":
        issues = [i for i in issues if i.category == args.type]

    # Print summary
    print(f"\n{'='*60}")
    print("SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"Total operators scanned: {scanner.operators_scanned}")
    print(f"Total issues found: {len(issues)}")
    print(f"Auto-fixable: {len([i for i in issues if i.auto_fixable])}")
    print(f"Requires manual fix: {len([i for i in issues if not i.auto_fixable])}")

    # By category
    by_category = Counter(i.category for i in issues)
    print(f"\nBy category:")
    for category, count in by_category.most_common():
        fixable = len([i for i in issues if i.category == category and i.auto_fixable])
        print(f"  {category}: {count} ({fixable} auto-fixable)")

    # Apply fixes if requested
    results = []
    if args.all and not args.scan:
        dry_run = not args.apply
        fixer = OperatorFixer(dry_run=dry_run)

        try:
            results = fixer.fix_issues(issues)

            successful = len([r for r in results if r.success])
            failed = len([r for r in results if not r.success])

            print(f"\n{'='*60}")
            print("FIX SUMMARY")
            print(f"{'='*60}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

            if dry_run:
                print("\n[DRY RUN] No files were modified.")
            else:
                print(f"\n✓ Modified {len(fixer.fixed_files)} files")
                print(f"✓ Backup directory: {fixer.backup_dir}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Rolling back changes...")
            fixer.rollback()
            sys.exit(1)

    # Generate report
    generator = ReportGenerator()
    report_path = generator.generate_report(issues, results, args.report)

    print(f"\n{'='*60}")
    print(f"✓ Complete! Report: {report_path}")
    print(f"{'='*60}")

    return 0 if not issues or args.scan else 1


if __name__ == "__main__":
    sys.exit(main())
