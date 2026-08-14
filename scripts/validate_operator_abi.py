#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate operator ABI consistency: param_names/param_types must match function signatures.

Mission: Wave1-Agent3-OperatorABI
- Extract actual function signatures from operator implementations
- Compare with declared param_names/param_types in metadata
- Identify mismatches: missing params, phantom params, type mismatches
- Focus on production_ready=True operators

Run with memory limit: ulimit -v 15728640 && python3 scripts/validate_operator_abi.py
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class FunctionSignature:
    """Extracted function signature."""
    name: str
    params: list[str]
    param_types: dict[str, str]
    file_path: str
    lineno: int


@dataclass
class ABIViolation:
    """ABI consistency violation."""
    canonical: str
    file_path: str
    violation_type: str
    expected_signature: list[str]
    declared_params: list[str]
    details: str
    severity: str  # "critical", "high", "medium", "low"


class SignatureExtractor(ast.NodeVisitor):
    """Extract function signatures from AST, grouped by class."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions: dict[str, FunctionSignature] = {}
        self.class_methods: dict[str, dict[str, FunctionSignature]] = {}
        self.current_class = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track current class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.class_methods[node.name] = {}
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function signature."""
        params = []
        param_types = {}

        for arg in node.args.args:
            # Skip 'self' and 'cls'
            if arg.arg in ('self', 'cls'):
                continue
            params.append(arg.arg)

            # Extract type annotation if present
            if arg.annotation:
                try:
                    param_types[arg.arg] = ast.unparse(arg.annotation)
                except Exception:
                    param_types[arg.arg] = "Any"

        sig = FunctionSignature(
            name=node.name,
            params=params,
            param_types=param_types,
            file_path=self.file_path,
            lineno=node.lineno
        )

        if self.current_class:
            self.class_methods[self.current_class][node.name] = sig
        else:
            self.functions[node.name] = sig

        self.generic_visit(node)


def extract_signatures_from_file(file_path: Path) -> tuple[dict[str, FunctionSignature], dict[str, dict[str, FunctionSignature]]]:
    """Extract all function signatures from a Python file.

    Returns: (module_functions, class_methods)
        - module_functions: dict of module-level functions
        - class_methods: dict[class_name][method_name] -> signature
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        extractor = SignatureExtractor(str(file_path))
        extractor.visit(tree)
        return extractor.functions, extractor.class_methods
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
        return {}, {}


def extract_param_names_from_registration(file_path: Path) -> dict[str, list[str]]:
    """Extract _PARAMS dict mapping canonical -> param_names from registration section."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        # Handle both ast.Assign and ast.AnnAssign (type-annotated assignments)
        for node in ast.walk(tree):
            target_name = None
            value_node = None

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '_PARAMS':
                        target_name = '_PARAMS'
                        value_node = node.value
                        break
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == '_PARAMS':
                    target_name = '_PARAMS'
                    value_node = node.value

            if target_name == '_PARAMS' and value_node:
                # Found _PARAMS dict
                if isinstance(value_node, ast.Dict):
                    result = {}
                    for key, value in zip(value_node.keys, value_node.values):
                        if isinstance(key, ast.Constant) and isinstance(value, ast.List):
                            canonical = key.value
                            params = [elt.value for elt in value.elts if isinstance(elt, ast.Constant)]
                            result[canonical] = params
                    if result:
                        return result
        return {}
    except Exception as e:
        print(f"Warning: Failed to extract _PARAMS from {file_path}: {e}", file=sys.stderr)
        return {}


def extract_kernel_mapping(file_path: Path) -> dict[str, str]:
    """Extract _KERNELS dict mapping canonical -> function_name."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        # Handle both ast.Assign and ast.AnnAssign (type-annotated assignments)
        for node in ast.walk(tree):
            target_name = None
            value_node = None

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '_KERNELS':
                        target_name = '_KERNELS'
                        value_node = node.value
                        break
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == '_KERNELS':
                    target_name = '_KERNELS'
                    value_node = node.value

            if target_name == '_KERNELS' and value_node:
                if isinstance(value_node, ast.Dict):
                    result = {}
                    for key, value in zip(value_node.keys, value_node.values):
                        if isinstance(key, ast.Constant):
                            canonical = key.value
                            if isinstance(value, ast.Name):
                                result[canonical] = value.id
                    if result:
                        return result
        return {}
    except Exception as e:
        print(f"Warning: Failed to extract _KERNELS from {file_path}: {e}", file=sys.stderr)
        return {}


def extract_operator_class_metadata(file_path: Path) -> dict[str, tuple[list[str], str, str]]:
    """Extract param_names from OperatorMetadata and the corresponding method name.

    Returns dict: canonical -> (param_names, method_name, class_name)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        result = {}

        # Look for class definitions with metadata attributes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Find metadata = _metadata(...) or metadata = OperatorMetadata(...)
                metadata_node = None
                method_name = None

                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == 'metadata':
                                metadata_node = item.value
                                break
                    elif isinstance(item, ast.FunctionDef):
                        if item.name in ('_calculate_series', 'calculate', '_calculate'):
                            method_name = item.name

                if metadata_node and method_name:
                    # Extract param_names from metadata
                    param_names = extract_param_names_from_call(metadata_node)
                    canonical = extract_canonical_from_metadata(metadata_node)

                    if param_names and canonical:
                        result[canonical] = (param_names, method_name, node.name)

        return result
    except Exception as e:
        print(f"Warning: Failed to extract class metadata from {file_path}: {e}", file=sys.stderr)
        return {}


def extract_param_names_from_call(node: ast.AST) -> list[str] | None:
    """Extract param_names from a function call like OperatorMetadata(...) or _metadata(...)."""
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            if keyword.arg == 'param_names':
                if isinstance(keyword.value, ast.List):
                    return [elt.value for elt in keyword.value.elts if isinstance(elt, ast.Constant)]
        # Check positional args for _metadata(name, description, params, ...)
        if len(node.args) >= 3 and isinstance(node.args[2], ast.List):
            return [elt.value for elt in node.args[2].elts if isinstance(elt, ast.Constant)]
    return None


def extract_canonical_from_metadata(node: ast.AST) -> str | None:
    """Extract canonical name from metadata definition."""
    if isinstance(node, ast.Call):
        # Look for name= keyword
        for keyword in node.keywords:
            if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        # Check first positional arg for _metadata(name, ...)
        if node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
    return None


def validate_class_operator_abi(
    class_methods: dict[str, dict[str, FunctionSignature]],
    class_metadata: dict[str, tuple[list[str], str, str]],
    file_path: Path
) -> list[ABIViolation]:
    """Validate class-based operator declarations."""
    violations = []

    for canonical, (declared_params, method_name, class_name) in class_metadata.items():
        # Look up the method in the specific class
        if class_name not in class_methods:
            violations.append(ABIViolation(
                canonical=canonical,
                file_path=str(file_path),
                violation_type="missing_class",
                expected_signature=[],
                declared_params=declared_params,
                details=f"Class '{class_name}' not found in file",
                severity="critical"
            ))
            continue

        if method_name not in class_methods[class_name]:
            violations.append(ABIViolation(
                canonical=canonical,
                file_path=str(file_path),
                violation_type="missing_method",
                expected_signature=[],
                declared_params=declared_params,
                details=f"Method '{method_name}' not found in class '{class_name}'",
                severity="critical"
            ))
            continue

        sig = class_methods[class_name][method_name]
        actual_params = sig.params

        # Filter out kwargs catch-all params
        actual_params_filtered = [p for p in actual_params if not p.startswith('_')]

        # Check for mismatches
        if actual_params_filtered != declared_params:
            missing_in_decl = set(actual_params_filtered) - set(declared_params)
            phantom_in_decl = set(declared_params) - set(actual_params_filtered)

            details = []
            if missing_in_decl:
                details.append(f"Missing in declaration: {sorted(missing_in_decl)}")
            if phantom_in_decl:
                details.append(f"Phantom in declaration: {sorted(phantom_in_decl)}")

            if set(actual_params_filtered) == set(declared_params):
                details.append("Parameter order mismatch")
                severity = "medium"
            else:
                severity = "high"

            violations.append(ABIViolation(
                canonical=canonical,
                file_path=str(file_path),
                violation_type="param_mismatch",
                expected_signature=actual_params_filtered,
                declared_params=declared_params,
                details="; ".join(details),
                severity=severity
            ))

    return violations


def validate_abi_consistency(
    signatures: dict[str, FunctionSignature],
    param_names_map: dict[str, list[str]],
    kernel_map: dict[str, str],
    file_path: Path
) -> list[ABIViolation]:
    """Validate that declared param_names match actual function signatures."""
    violations = []

    for canonical, declared_params in param_names_map.items():
        kernel_name = kernel_map.get(canonical)
        if not kernel_name:
            continue

        if kernel_name not in signatures:
            violations.append(ABIViolation(
                canonical=canonical,
                file_path=str(file_path),
                violation_type="missing_function",
                expected_signature=[],
                declared_params=declared_params,
                details=f"Kernel function '{kernel_name}' not found in file",
                severity="critical"
            ))
            continue

        sig = signatures[kernel_name]
        actual_params = sig.params

        # Check for mismatches
        if actual_params != declared_params:
            # Determine specific issue
            missing_in_decl = set(actual_params) - set(declared_params)
            phantom_in_decl = set(declared_params) - set(actual_params)

            details = []
            if missing_in_decl:
                details.append(f"Missing in declaration: {sorted(missing_in_decl)}")
            if phantom_in_decl:
                details.append(f"Phantom in declaration: {sorted(phantom_in_decl)}")

            # Check if order is just wrong
            if set(actual_params) == set(declared_params):
                details.append("Parameter order mismatch")
                severity = "medium"
            else:
                severity = "high"

            violations.append(ABIViolation(
                canonical=canonical,
                file_path=str(file_path),
                violation_type="param_mismatch",
                expected_signature=actual_params,
                declared_params=declared_params,
                details="; ".join(details),
                severity=severity
            ))

    return violations


def scan_operator_files() -> list[ABIViolation]:
    """Scan all operator files for ABI violations."""
    # Handle both regular repo and factor_engine subdirectory
    if (REPO_ROOT / "cleaned_operators").exists():
        cleaned_ops_dir = REPO_ROOT / "cleaned_operators"
    elif (REPO_ROOT / "factor_engine" / "cleaned_operators").exists():
        cleaned_ops_dir = REPO_ROOT / "factor_engine" / "cleaned_operators"
    else:
        print(f"Error: cleaned_operators not found in {REPO_ROOT}")
        return []

    all_violations = []

    # Get all .py files
    py_files = sorted(cleaned_ops_dir.glob("*.py"))

    print(f"Scanning {len(py_files)} operator files...")

    files_with_kernels = 0
    files_with_classes = 0

    for py_file in py_files:
        # Skip special files
        if py_file.name.startswith('__') or py_file.name.startswith('.'):
            continue

        # Extract signatures
        module_functions, class_methods = extract_signatures_from_file(py_file)

        # Pattern 1: _PARAMS + _KERNELS registration
        param_names_map = extract_param_names_from_registration(py_file)
        kernel_map = extract_kernel_mapping(py_file)

        if param_names_map and kernel_map:
            files_with_kernels += 1
            violations = validate_abi_consistency(
                module_functions, param_names_map, kernel_map, py_file
            )
            all_violations.extend(violations)

        # Pattern 2: Class-based operators with metadata
        class_metadata = extract_operator_class_metadata(py_file)
        if class_metadata:
            files_with_classes += 1
            violations = validate_class_operator_abi(
                class_methods, class_metadata, py_file
            )
            all_violations.extend(violations)

    print(f"Found {files_with_kernels} files with _PARAMS/_KERNELS pattern")
    print(f"Found {files_with_classes} files with class-based operators")

    return all_violations


def print_violations_report(violations: list[ABIViolation]) -> None:
    """Print formatted violations report."""
    if not violations:
        print("\n✓ No ABI violations found!")
        return

    print(f"\n⚠ Found {len(violations)} ABI violations:\n")

    # Group by severity
    by_severity = defaultdict(list)
    for v in violations:
        by_severity[v.severity].append(v)

    for severity in ["critical", "high", "medium", "low"]:
        if severity not in by_severity:
            continue

        print(f"\n{'='*80}")
        print(f"{severity.upper()} SEVERITY ({len(by_severity[severity])} violations)")
        print(f"{'='*80}")

        for v in by_severity[severity]:
            print(f"\nCanonical: {v.canonical}")
            print(f"File: {Path(v.file_path).name}")
            print(f"Type: {v.violation_type}")
            print(f"Expected signature: {v.expected_signature}")
            print(f"Declared params:    {v.declared_params}")
            print(f"Details: {v.details}")


def main():
    """Main validation entry point."""
    print("=" * 80)
    print("Operator ABI Consistency Validation")
    print("=" * 80)

    violations = scan_operator_files()
    print_violations_report(violations)

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {len(violations)} total violations")

    by_severity = defaultdict(int)
    for v in violations:
        by_severity[v.severity] += 1

    for severity in ["critical", "high", "medium", "low"]:
        if by_severity[severity] > 0:
            print(f"  {severity.upper()}: {by_severity[severity]}")

    print("=" * 80)

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
