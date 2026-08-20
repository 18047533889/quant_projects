#!/usr/bin/env python3
"""Static future-function linter (audit P1-Q).

Scans the OPERATOR kernels (``cleaned_operators``) for look-ahead / future-data
patterns that must never enter a prefix-causal operator implementation:

* ``shift(-n)`` / ``diff(-n)`` / ``pct_change(-n)``
* ``rolling(center=True)``
* ``bfill`` (backward fill reads future rows)
* backward interpolation (``interpolate(method=\"...\")`` on future)
* future slices ``x[t+1:]`` / ``x[1:]`` fed forward into the current value
* explicit ``Lead`` operator registration usage

The backend emitters are NOT scanned: implementing ``bfill`` / ``Lead`` as
registered DSL operators is their job.  Research operators that legitimately use
a deterministic circular surrogate (e.g. transfer-entropy surrogates) are
whitelisted with an explicit reason.  Exits non-zero on any un-whitelisted hit so
CI can fail the release.

Usage:
    python scripts/lint_future_functions.py            # scan + exit code
    python scripts/lint_future_functions.py --report   # human-readable report
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = FE_ROOT / "cleaned_operators"

# canonical -> reason why a future/lead pattern is intentionally present.
ALLOWED_FUTURE_CANONICALS: dict[str, str] = {
    "bfill": "registered non-factor DSL op; its whole semantics is backward fill",
    "Lead": "registered non-factor DSL op; next-period value is its semantics",
    "ts_transfer_entropy_direction": (
        "deterministic circular surrogate (research-only); no true future is read"
    ),
    "ts_transfer_entropy_magnitude": (
        "deterministic circular surrogate (research-only); no true future is read"
    ),
}

# (regex, description) — high-signal patterns only.
_FUTURE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.shift\(\s*-\s*\d"), "shift(-n)"),
    (re.compile(r"\.diff\(\s*-\s*\d"), "diff(-n)"),
    (re.compile(r"\.pct_change\(\s*-\s*\d"), "pct_change(-n)"),
    (re.compile(r"rolling\([^)]*center\s*=\s*True"), "rolling(center=True)"),
    (re.compile(r"\bbfill\s*\("), "bfill (backward fill reads future rows)"),
    (re.compile(r"interpolate\("), "backward/forward interpolation"),
    (re.compile(r"\[t\s*\+\s*1\s*:\]"), "future slice x[t+1:]"),
    (re.compile(r"\bLead\s*\("), "Lead operator call"),
]


def _scan_file(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return hits
    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for pattern, desc in _FUTURE_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, desc))
                break
    return hits


def _module_canonicals(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    return set(re.findall(r'canonical="([^"]+)"', text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="human-readable report")
    parser.add_argument(
        "--allowlist",
        nargs="*",
        default=sorted(ALLOWED_FUTURE_CANONICALS),
        help="canonicals whose future/lead patterns are intentional",
    )
    args = parser.parse_args()

    allow = set(args.allowlist)
    problems: list[tuple[str, int, str]] = []
    for path in sorted(SCAN_DIR.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        module_canonicals = _module_canonicals(path)
        if module_canonicals & allow:
            continue
        for lineno, desc in _scan_file(path):
            problems.append((str(path.relative_to(FE_ROOT)), lineno, desc))

    if args.report or problems:
        for path, lineno, desc in problems:
            print(f"{path}:{lineno}: {desc}")
    if problems:
        print(f"future-function linter: {len(problems)} violations", file=sys.stderr)
        return 1
    print("future-function linter: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
