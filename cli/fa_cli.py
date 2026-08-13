#!/usr/bin/env python3
"""
fa_cli: Command-line interface for Factor Analysis/Registry.

Query and inspect the factor operator registry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "factor_engine"))


@click.group()
@click.version_option(version="0.1.0")
def fa_cli():
    """Factor Analysis CLI - Query factor registry and operators."""
    pass


@fa_cli.command()
@click.option("--family", "-f", help="Filter by operator family")
@click.option("--timing", "-t", help="Filter by timing kind")
@click.option("--lane", "-l", help="Filter by lane")
@click.option("--count", "-c", is_flag=True, help="Show only count")
def list_operators(family: str | None, timing: str | None, lane: str | None, count: bool):
    """List available operators in the registry."""
    try:
        from cleaned_operators import load_all, OPERATOR_REGISTRY
        load_all()
    except ImportError as e:
        click.echo(f"Error: Cannot load factor engine - {e}", err=True)
        sys.exit(1)

    operators = OPERATOR_REGISTRY._operators

    # Apply filters
    filtered = {}
    for name, op_def in operators.items():
        if family and op_def.family != family:
            continue
        if timing and str(op_def.timing_kind) != timing:
            continue
        if lane and getattr(op_def, "lane", None) != lane:
            continue
        filtered[name] = op_def

    if count:
        click.echo(f"{len(filtered)}")
        return

    click.echo(f"Found {len(filtered)} operators")
    click.echo("=" * 80)

    for name, op_def in sorted(filtered.items()):
        family_str = getattr(op_def, "family", "unknown")
        timing_str = str(getattr(op_def, "timing_kind", "unknown"))
        lane_str = getattr(op_def, "lane", "unknown")
        click.echo(f"{name:40s} [{family_str:15s}] {timing_str:15s} {lane_str}")


@fa_cli.command()
@click.argument("operator_name")
def info(operator_name: str):
    """Show detailed information about an operator."""
    try:
        from cleaned_operators import load_all, OPERATOR_REGISTRY
        load_all()
    except ImportError as e:
        click.echo(f"Error: Cannot load factor engine - {e}", err=True)
        sys.exit(1)

    if operator_name not in OPERATOR_REGISTRY._operators:
        click.echo(f"Error: Operator '{operator_name}' not found", err=True)
        sys.exit(1)

    op_def = OPERATOR_REGISTRY._operators[operator_name]

    click.echo(f"Operator: {operator_name}")
    click.echo("=" * 80)
    click.echo(f"Family:       {getattr(op_def, 'family', 'unknown')}")
    click.echo(f"Timing:       {getattr(op_def, 'timing_kind', 'unknown')}")
    click.echo(f"Lane:         {getattr(op_def, 'lane', 'unknown')}")
    click.echo(f"Active When:  {getattr(op_def, 'active_when', 'always')}")

    if hasattr(op_def, 'params'):
        click.echo(f"\nParameters:")
        for param_name, param_spec in op_def.params.items():
            param_kind = getattr(param_spec, 'kind', 'unknown')
            param_role = getattr(param_spec, 'role', 'unknown')
            click.echo(f"  {param_name:20s} kind={param_kind:15s} role={param_role}")

    if hasattr(op_def, 'kernel'):
        click.echo(f"\nKernel: {op_def.kernel.__name__}")
        if op_def.kernel.__doc__:
            doc_lines = op_def.kernel.__doc__.strip().split("\n")
            click.echo(f"  {doc_lines[0]}")


@fa_cli.command()
def families():
    """List all operator families."""
    try:
        from cleaned_operators import load_all, OPERATOR_REGISTRY
        load_all()
    except ImportError as e:
        click.echo(f"Error: Cannot load factor engine - {e}", err=True)
        sys.exit(1)

    operators = OPERATOR_REGISTRY._operators
    family_counts = {}

    for op_def in operators.values():
        family = getattr(op_def, "family", "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1

    click.echo("Operator Families:")
    click.echo("=" * 60)
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        click.echo(f"{family:30s} {count:6d} operators")

    click.echo(f"\nTotal: {sum(family_counts.values())} operators across {len(family_counts)} families")


@fa_cli.command()
@click.argument("search_term")
def search(search_term: str):
    """Search for operators by name or description."""
    try:
        from cleaned_operators import load_all, OPERATOR_REGISTRY
        load_all()
    except ImportError as e:
        click.echo(f"Error: Cannot load factor engine - {e}", err=True)
        sys.exit(1)

    operators = OPERATOR_REGISTRY._operators
    search_lower = search_term.lower()

    matches = []
    for name, op_def in operators.items():
        if search_lower in name.lower():
            matches.append((name, op_def, "name"))
        elif hasattr(op_def, "kernel") and op_def.kernel.__doc__:
            if search_lower in op_def.kernel.__doc__.lower():
                matches.append((name, op_def, "doc"))

    if not matches:
        click.echo(f"No operators found matching '{search_term}'")
        return

    click.echo(f"Found {len(matches)} operators matching '{search_term}':")
    click.echo("=" * 80)

    for name, op_def, match_type in matches:
        family = getattr(op_def, "family", "unknown")
        timing = str(getattr(op_def, "timing_kind", "unknown"))
        click.echo(f"{name:40s} [{family:15s}] {timing:15s} ({match_type})")


@fa_cli.command()
def stats():
    """Show registry statistics."""
    try:
        from cleaned_operators import load_all, OPERATOR_REGISTRY
        load_all()
    except ImportError as e:
        click.echo(f"Error: Cannot load factor engine - {e}", err=True)
        sys.exit(1)

    operators = OPERATOR_REGISTRY._operators

    click.echo("Factor Registry Statistics:")
    click.echo("=" * 80)
    click.echo(f"Total operators: {len(operators)}")

    # Count by timing
    timing_counts = {}
    for op_def in operators.values():
        timing = str(getattr(op_def, "timing_kind", "unknown"))
        timing_counts[timing] = timing_counts.get(timing, 0) + 1

    click.echo(f"\nBy Timing Kind:")
    for timing, count in sorted(timing_counts.items(), key=lambda x: -x[1]):
        click.echo(f"  {timing:30s} {count:6d}")

    # Count by lane
    lane_counts = {}
    for op_def in operators.values():
        lane = getattr(op_def, "lane", "unknown")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

    click.echo(f"\nBy Lane:")
    for lane, count in sorted(lane_counts.items(), key=lambda x: -x[1]):
        click.echo(f"  {lane:30s} {count:6d}")

    # Count by family
    family_counts = {}
    for op_def in operators.values():
        family = getattr(op_def, "family", "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1

    click.echo(f"\nTop 10 Families:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1])[:10]:
        click.echo(f"  {family:30s} {count:6d}")


if __name__ == "__main__":
    fa_cli()
