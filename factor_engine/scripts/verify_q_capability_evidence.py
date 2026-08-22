#!/usr/bin/env python3
"""Verification script for Q backend capability evidence.

Usage:
    python scripts/verify_q_capability_evidence.py
    python scripts/verify_q_capability_evidence.py --detailed
    python scripts/verify_q_capability_evidence.py --gates-only
"""

import argparse
import sys
from pathlib import Path

# Add factor_engine to path
FE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FE_ROOT.parent))
sys.path.insert(0, str(FE_ROOT))

MANDATORY_GATES = frozenset({
    "Q_NATIVE_WITHOUT_LOWERING",
    "Q_PRODUCTION_SAFE_SINGLE_DEFINITION",
    "Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY",
    "Q_CAPABILITY_SINGLE_AUTHORITY",
})


def _verification_passed(report, gates):
    """Return whether readiness and every mandatory gate pass."""
    return bool(report.get("production_ready")) and all(
        name in gates and gates[name][0] for name in MANDATORY_GATES
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify Q backend capability evidence"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed operator breakdown",
    )
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="Only run hard gates",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )
    args = parser.parse_args()

    from backend.q_backend.q_capability_evidence import (
        compute_q_capability_evidence,
        generate_capability_report,
        get_q_native_without_lowering,
        run_all_q_capability_gates,
    )

    # Run hard gates
    gates = run_all_q_capability_gates()

    if args.json:
        import json
        report = generate_capability_report()
        gate_results = {
            name: {"passed": passed, "message": message}
            for name, (passed, message) in gates.items()
        }
        output = {
            "report": report,
            "gates": gate_results,
        }
        print(json.dumps(output, indent=2))
        sys.exit(0 if _verification_passed(report, gates) else 1)

    if args.gates_only:
        report = generate_capability_report()
        print("=" * 80)
        print("Q BACKEND CAPABILITY HARD GATES")
        print("=" * 80)
        print()

        for gate_name, (passed, message) in gates.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            mandatory = " [MANDATORY]" if gate_name in MANDATORY_GATES else ""
            print(f"{status} {gate_name}{mandatory}")
            print(f"     {message}")
            print()

        if _verification_passed(report, gates):
            print("=" * 80)
            print("All mandatory gates PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("=" * 80)
            print("Some mandatory gates FAILED or Q backend is NOT production ready")
            print("=" * 80)
            sys.exit(1)

    # Full report
    report = generate_capability_report()

    print("=" * 80)
    print("Q BACKEND CAPABILITY EVIDENCE REPORT")
    print("=" * 80)
    print()
    print(f"Declared Native:           {report['declared_native_count']}")
    print(f"Lowering Exists:           {report['lowering_exists_count']}")
    print(f"Native WITHOUT Lowering:   {report['native_without_lowering_count']}")
    print(f"Production Safe:           {report['production_safe_count']}")
    print(f"Production Ready:          {'YES' if report['production_ready'] else 'NO'}")
    print()

    # Show gaps
    if report['native_without_lowering_count'] > 0:
        print("=" * 80)
        print(f"Q_NATIVE_WITHOUT_LOWERING ({report['native_without_lowering_count']} operators)")
        print("=" * 80)
        print()
        print("The following operators are declared native but lack lowering:")
        print()
        for i, op in enumerate(report['native_without_lowering'], 1):
            print(f"  {i:2d}. {op}")
        print()

    # Hard gates
    print("=" * 80)
    print("HARD GATE RESULTS")
    print("=" * 80)
    print()

    all_passed = True
    mandatory_gates = {
        "Q_NATIVE_WITHOUT_LOWERING",
        "Q_PRODUCTION_SAFE_SINGLE_DEFINITION",
        "Q_COMPILER_ADMISSION_USES_EVIDENCE_AUTHORITY_ONLY",
        "Q_CAPABILITY_SINGLE_AUTHORITY",
    }

    for gate_name, (passed, message) in gates.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        mandatory = " [MANDATORY]" if gate_name in MANDATORY_GATES else ""
        print(f"{status} {gate_name}{mandatory}")
        print(f"     {message}")
        print()
        if not passed and gate_name in mandatory_gates:
            all_passed = False

    # Detailed breakdown
    if args.detailed:
        print("=" * 80)
        print("DETAILED OPERATOR EVIDENCE")
        print("=" * 80)
        print()

        evidence = compute_q_capability_evidence()

        # Group by status
        has_both = []
        declared_only = []
        lowering_only = []

        for op, ev in sorted(evidence.items()):
            if ev.declared_native and ev.lowering_exists:
                has_both.append(op)
            elif ev.declared_native:
                declared_only.append(op)
            elif ev.lowering_exists:
                lowering_only.append(op)

        print(f"Operators with BOTH declaration and lowering ({len(has_both)}):")
        for op in has_both[:20]:
            print(f"  ✓ {op}")
        if len(has_both) > 20:
            print(f"  ... and {len(has_both) - 20} more")
        print()

        if declared_only:
            print(f"Operators with declaration but NO lowering ({len(declared_only)}):")
            for op in declared_only:
                print(f"  ✗ {op}")
            print()

        if lowering_only:
            print(f"Operators with lowering but NO declaration ({len(lowering_only)}):")
            for op in lowering_only:
                print(f"  ? {op}")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Check all mandatory gates
    all_mandatory_passed = all(
        name in gates and gates[name][0] for name in MANDATORY_GATES
    )

    if all_mandatory_passed and report['production_ready']:
        print("✓ Q backend is PRODUCTION READY")
        print("  All capability declarations have lowering implementations")
        print("  All lowerings pass all 5 evidence checks")
        print("  Single authority for production-safe definition")
        sys.exit(0)
    else:
        print("✗ Q backend is NOT production ready")
        if report['native_without_lowering_count'] > 0:
            print(f"  {report['native_without_lowering_count']} operators lack lowering implementations")
        if report['production_safe_count'] < report['lowering_exists_count']:
            incomplete = report['lowering_exists_count'] - report['production_safe_count']
            print(f"  {incomplete} lowerings lack full evidence passes (compile/runtime/parity)")
        if not all_mandatory_passed:
            print("  Some mandatory hard gates failed")
        print()
        print("Action required:")
        print("  1. Implement missing lowerings for declared operators, OR")
        print("  2. Remove operators from declared native set, AND")
        print("  3. Add compile/runtime/parity evidence for existing lowerings")
        print()
        print("See Q2_P0_001_005_REMEDIATION.md for detailed plan")
        sys.exit(1)


if __name__ == "__main__":
    main()
