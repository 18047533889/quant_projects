#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit script for placeholder/TODO operators.

Validates that operators marked with TODO/placeholder/approximation comments
are either:
1. Marked as research_only=True, OR
2. Have complete implementations with no placeholder patterns

Critical violations: production operators (no research_only flag) with TODOs/placeholders
"""
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Patterns to detect
TODO_PATTERN = re.compile(r'#\s*(TODO|FIXME|XXX|HACK)[:\s]*(.*)', re.IGNORECASE)
STUB_PATTERN = re.compile(r'raise NotImplementedError.*stub', re.IGNORECASE)
APPROX_PATTERN = re.compile(r'(approximation|placeholder|simplified)', re.IGNORECASE)
PLACEHOLDER_RETURN = re.compile(r'return\s+(None|0|np\.nan|pl\.col.*fill_nan)', re.MULTILINE)


class OperatorInfo:
    def __init__(self, name: str, class_name: str, line: int):
        self.name = name
        self.class_name = class_name
        self.line = line
        self.production_ready = None  # None means not specified (defaults to True)
        self.research_only = None
        self.todos = []
        self.stubs = []
        self.approx = []
        self.placeholder_returns = []

    @property
    def is_production(self) -> bool:
        """Check if operator is considered production-ready."""
        if self.research_only is True:
            return False
        if self.production_ready is False:
            return False
        # No flag means defaults to production
        return True

    @property
    def has_placeholders(self) -> bool:
        """Check if operator has any placeholder patterns."""
        return bool(self.todos or self.stubs or self.approx or self.placeholder_returns)

    @property
    def is_critical_violation(self) -> bool:
        """Production operator with placeholders is a critical violation."""
        return self.is_production and self.has_placeholders


def parse_operator_file(file_path: Path) -> List[OperatorInfo]:
    """Parse a Python file and extract operator information."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        print(f"WARNING: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    operators = []

    # Find all class definitions with @register_operator decorator
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check for @register_operator decorator
        has_register = False
        op_name = None
        prod_ready = None
        research_only = None

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if not (hasattr(dec.func, 'id') and dec.func.id == 'register_operator'):
                continue

            has_register = True

            # Extract decorator arguments
            for kw in dec.keywords:
                if kw.arg == 'name' and isinstance(kw.value, ast.Constant):
                    op_name = kw.value.value
                elif kw.arg == 'production_ready' and isinstance(kw.value, ast.Constant):
                    prod_ready = kw.value.value
                elif kw.arg == 'research_only' and isinstance(kw.value, ast.Constant):
                    research_only = kw.value.value

        if not has_register:
            continue

        # Create operator info
        op_info = OperatorInfo(
            name=op_name or node.name,
            class_name=node.name,
            line=node.lineno
        )
        op_info.production_ready = prod_ready
        op_info.research_only = research_only

        # Scan the class body for placeholders
        class_start = node.lineno - 1
        class_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

        for i in range(class_start, min(class_end, len(lines))):
            line = lines[i]
            line_num = i + 1

            if TODO_PATTERN.search(line):
                op_info.todos.append((line_num, line.strip()))
            if STUB_PATTERN.search(line):
                op_info.stubs.append((line_num, line.strip()))
            if APPROX_PATTERN.search(line):
                op_info.approx.append((line_num, line.strip()))
            if PLACEHOLDER_RETURN.search(line):
                op_info.placeholder_returns.append((line_num, line.strip()))

        operators.append(op_info)

    return operators


def audit_directory(root_dir: Path) -> Dict[str, List[OperatorInfo]]:
    """Audit all operator files in directory."""
    results = {}

    for pyfile in root_dir.rglob('*.py'):
        if 'test' in str(pyfile):
            continue

        rel_path = str(pyfile.relative_to(root_dir))
        operators = parse_operator_file(pyfile)

        if operators:
            results[rel_path] = operators

    return results


def print_report(results: Dict[str, List[OperatorInfo]]) -> Tuple[int, int]:
    """Print audit report and return (total_operators, critical_violations)."""
    total_ops = 0
    critical_count = 0

    # Summary by file
    print("=" * 80)
    print("PLACEHOLDER OPERATOR AUDIT REPORT")
    print("=" * 80)
    print()

    # Collect critical violations
    critical_files = []
    for fpath, ops in sorted(results.items()):
        critical_ops = [op for op in ops if op.is_critical_violation]
        if critical_ops:
            critical_files.append((fpath, critical_ops))
            critical_count += len(critical_ops)
        total_ops += len(ops)

    # Print critical violations first
    if critical_files:
        print("CRITICAL VIOLATIONS (production operators with placeholders):")
        print("-" * 80)
        for fpath, ops in critical_files:
            print(f"\n{fpath}:")
            for op in ops:
                print(f"  [{op.line:4d}] {op.name:40} (class: {op.class_name})")
                print(f"         production_ready={op.production_ready}, research_only={op.research_only}")
                if op.todos:
                    print(f"         TODOs: {len(op.todos)}")
                    for line_num, text in op.todos[:2]:
                        print(f"           L{line_num}: {text[:70]}")
                if op.stubs:
                    print(f"         Stubs: {len(op.stubs)}")
                if op.approx:
                    print(f"         Approximations: {len(op.approx)}")
                    for line_num, text in op.approx[:2]:
                        print(f"           L{line_num}: {text[:70]}")
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total operators audited: {total_ops}")
    print(f"  Critical violations: {critical_count}")

    # Count by type
    total_with_placeholders = sum(
        len([op for op in ops if op.has_placeholders])
        for ops in results.values()
    )
    research_with_placeholders = sum(
        len([op for op in ops if not op.is_production and op.has_placeholders])
        for ops in results.values()
    )

    print(f"  Operators with placeholders: {total_with_placeholders}")
    print(f"    - Research/non-production (OK): {research_with_placeholders}")
    print(f"    - Production (CRITICAL): {critical_count}")
    print("=" * 80)

    return total_ops, critical_count


def main():
    """Run audit and exit with error code if critical violations found."""
    # Support running from worktree or main repo
    script_parent = Path(__file__).parent.parent
    root = script_parent / 'cleaned_operators'

    # If not in worktree, try main repo
    if not root.exists():
        root = Path('/home/shw/quant_projects/factor_engine/cleaned_operators')

    if not root.exists():
        print(f"ERROR: cleaned_operators not found", file=sys.stderr)
        return 1

    print(f"Auditing operators in: {root}")
    print()

    results = audit_directory(root)
    total_ops, critical_count = print_report(results)

    if critical_count > 0:
        print()
        print(f"AUDIT FAILED: {critical_count} critical violations found")
        print("Fix required: Either implement operators fully OR mark research_only=True")
        return 1

    print()
    print("AUDIT PASSED: No critical violations")
    return 0


if __name__ == '__main__':
    sys.exit(main())
