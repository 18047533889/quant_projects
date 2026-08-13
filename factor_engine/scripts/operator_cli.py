#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for operator automation tools - user-friendly interface."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_section(text: str):
    """Print formatted section."""
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}\n")


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status."""
    print(f"🔧 {description}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0

        if success:
            print(f"✓ {description} - Success")
        else:
            print(f"✗ {description} - Failed (exit code {result.returncode})")

        return success, output

    except Exception as e:
        print(f"✗ {description} - Error: {e}")
        return False, str(e)


def cmd_quick_scan(args):
    """Quick scan for all issues."""
    print_header("Quick Operator Scan")

    cmd = ["python3", "scripts/auto_fix_operators.py", "--scan"]

    if args.report:
        cmd.extend(["--report", args.report])

    success, output = run_command(cmd, "Scanning operators")

    if success:
        print("\n📊 Scan complete!")
        report_path = args.report or "/tmp/auto_fix_operators_report.md"
        print(f"📄 Report: {report_path}")

        # Show summary
        for line in output.split("\n"):
            if "Total" in line or "Auto-fixable" in line or "category:" in line:
                print(f"  {line}")

    return 0 if success else 1


def cmd_fix_docstrings(args):
    """Fix missing docstrings."""
    print_header("Fix Missing Docstrings")

    if not args.apply:
        print("⚠️  Running in DRY-RUN mode (use --apply to make changes)")

    cmd = ["python3", "scripts/auto_fix_operators.py", "--type", "docstring"]

    if args.apply:
        cmd.append("--apply")
    else:
        cmd.append("--dry-run")

    success, output = run_command(cmd, "Fixing docstrings")

    if success:
        if args.apply:
            print("\n✓ Docstrings fixed!")
        else:
            print("\n✓ Preview complete (no changes made)")
            print("   Run with --apply to make actual changes")

    return 0 if success else 1


def cmd_fix_defaults(args):
    """Fix missing parameter defaults."""
    print_header("Fix Missing Parameter Defaults")

    if not args.apply:
        print("⚠️  Running in DRY-RUN mode (use --apply to make changes)")

    cmd = ["python3", "scripts/auto_fix_operators.py", "--type", "defaults"]

    if args.apply:
        cmd.append("--apply")
    else:
        cmd.append("--dry-run")

    success, output = run_command(cmd, "Fixing parameter defaults")

    if success:
        if args.apply:
            print("\n✓ Parameter defaults fixed!")
        else:
            print("\n✓ Preview complete (no changes made)")

    return 0 if success else 1


def cmd_policy_audit(args):
    """Audit and generate policy templates."""
    print_header("Policy Audit")

    cmd = ["python3", "scripts/auto_fix_operators_enhanced.py", "--scan"]

    policy_report = args.output or "/tmp/policy_gaps.json"
    cmd.extend(["--policy-report", policy_report])

    success, output = run_command(cmd, "Auditing policies")

    if success:
        print(f"\n✓ Policy audit complete!")
        print(f"📄 Policy gaps: {policy_report}")

        # Show statistics
        try:
            data = json.loads(Path(policy_report).read_text())
            print(f"\n📊 Found {data['total_gaps']} operators without policies")

            # Show sample
            if data['operators'][:5]:
                print("\nSample operators missing policies:")
                for op in data['operators'][:5]:
                    print(f"  • {op['name']} ({Path(op['file']).name})")

                if data['total_gaps'] > 5:
                    print(f"  ... and {data['total_gaps'] - 5} more")

        except Exception as e:
            print(f"⚠️  Could not parse policy report: {e}")

    return 0 if success else 1


def cmd_batch_fix(args):
    """Run batch fix workflow."""
    print_header("Batch Fix Workflow")

    if not args.apply:
        print("⚠️  Running in DRY-RUN mode")

    cmd = ["python3", "scripts/batch_operator_operations.py", "--workflow", "default"]

    if args.git:
        cmd.append("--git")
        print("✓ Git integration enabled")

    if not args.apply:
        cmd.append("--dry-run")

    if args.parallel:
        cmd.append("--parallel")
        print("✓ Parallel processing enabled")

    success, output = run_command(cmd, "Running batch workflow")

    if success:
        print("\n✓ Batch workflow complete!")
    else:
        print("\n✗ Batch workflow failed")
        print("   Check output above for details")

    return 0 if success else 1


def cmd_stats(args):
    """Show operator statistics."""
    print_header("Operator Statistics")

    # Run quick scan
    cmd = ["python3", "scripts/auto_fix_operators_enhanced.py", "--scan"]
    success, output = run_command(cmd, "Collecting statistics")

    if not success:
        return 1

    # Parse output for statistics
    lines = output.split("\n")

    stats = {}
    for line in lines:
        if "Total operators scanned:" in line:
            stats['total'] = line.split(":")[-1].strip()
        elif "Total issues found:" in line:
            stats['issues'] = line.split(":")[-1].strip()
        elif "Auto-fixable:" in line:
            stats['auto_fixable'] = line.split(":")[-1].strip()

    print("\n📊 Statistics:")
    print(f"  Total operators: {stats.get('total', 'N/A')}")
    print(f"  Issues found: {stats.get('issues', 'N/A')}")
    print(f"  Auto-fixable: {stats.get('auto_fixable', 'N/A')}")

    # Parse categories
    print("\n📁 By category:")
    in_category_section = False
    for line in lines:
        if "By category:" in line:
            in_category_section = True
        elif in_category_section and ":" in line and "(" in line:
            print(f"  {line.strip()}")

    return 0


def cmd_interactive(args):
    """Interactive mode."""
    print_header("Interactive Operator Fixer")

    while True:
        print("\nAvailable operations:")
        print("  1. Quick scan")
        print("  2. Fix docstrings")
        print("  3. Fix parameter defaults")
        print("  4. Policy audit")
        print("  5. Batch fix (safe)")
        print("  6. Show statistics")
        print("  7. Exit")

        try:
            choice = input("\nSelect operation (1-7): ").strip()

            if choice == "1":
                cmd_quick_scan(argparse.Namespace(report=None))
            elif choice == "2":
                apply = input("Apply fixes? (y/n): ").lower() == 'y'
                cmd_fix_docstrings(argparse.Namespace(apply=apply))
            elif choice == "3":
                apply = input("Apply fixes? (y/n): ").lower() == 'y'
                cmd_fix_defaults(argparse.Namespace(apply=apply))
            elif choice == "4":
                cmd_policy_audit(argparse.Namespace(output=None))
            elif choice == "5":
                apply = input("Apply fixes? (y/n): ").lower() == 'y'
                git = input("Use git integration? (y/n): ").lower() == 'y'
                cmd_batch_fix(argparse.Namespace(apply=apply, git=git, parallel=False))
            elif choice == "6":
                cmd_stats(argparse.Namespace())
            elif choice == "7":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Operator automation tools - user-friendly CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick scan
  ./scripts/operator_cli.py scan

  # Fix docstrings (preview)
  ./scripts/operator_cli.py fix-docstrings

  # Fix docstrings (apply)
  ./scripts/operator_cli.py fix-docstrings --apply

  # Policy audit
  ./scripts/operator_cli.py policy-audit

  # Batch fix with git
  ./scripts/operator_cli.py batch-fix --git --apply

  # Interactive mode
  ./scripts/operator_cli.py interactive
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Quick scan for all issues")
    scan_parser.add_argument("--report", help="Report output path")
    scan_parser.set_defaults(func=cmd_quick_scan)

    # Fix docstrings
    doc_parser = subparsers.add_parser("fix-docstrings", help="Fix missing docstrings")
    doc_parser.add_argument("--apply", action="store_true", help="Apply changes")
    doc_parser.set_defaults(func=cmd_fix_docstrings)

    # Fix defaults
    def_parser = subparsers.add_parser("fix-defaults", help="Fix parameter defaults")
    def_parser.add_argument("--apply", action="store_true", help="Apply changes")
    def_parser.set_defaults(func=cmd_fix_defaults)

    # Policy audit
    policy_parser = subparsers.add_parser("policy-audit", help="Audit operator policies")
    policy_parser.add_argument("--output", help="Output JSON path")
    policy_parser.set_defaults(func=cmd_policy_audit)

    # Batch fix
    batch_parser = subparsers.add_parser("batch-fix", help="Run batch fix workflow")
    batch_parser.add_argument("--apply", action="store_true", help="Apply changes")
    batch_parser.add_argument("--git", action="store_true", help="Enable git integration")
    batch_parser.add_argument("--parallel", action="store_true", help="Parallel processing")
    batch_parser.set_defaults(func=cmd_batch_fix)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
