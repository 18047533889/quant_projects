#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17-001 static audit: no production path may use bare legacy field resolution.

The legacy A-share ``resolve_field(name)`` / ``FIELD_REGISTRY.resolve*`` entry
points implicitly bind the A-share registry.  A US formula reaching one of them
would resolve against A-share aliases/units/price basis — a silent double-truth
field system.

This scanner flags, in *production* source (everything under
``factor_engine/`` except ``tests/``, ``tools/``, ``scripts/``, ``build/`` and
the two explicit legacy wrappers)::

    from fields import resolve_field      -> bare import
    resolve_field(<name>)                  -> bare call
    FIELD_REGISTRY.resolve_...             -> bare registry method

The two compatibility wrappers themselves (``fields/resolver.py`` and the
``resolve_field`` re-export in ``fields/__init__.py``) are whitelisted because
they ARE the legacy API.  Exit code 0 == clean (no production bare usage).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_IMPORT_RE = re.compile(r"^\s*from\s+fields\s+import\s+.*\bresolve_field\b", re.M)
# Module-level call: NOT preceded by ``.`` (a method call on a registry object is
# market-scoped and fine), and NOT a ``def`` definition.
_CALL_RE = re.compile(r"(?<!\.)(?<!def\s)\bresolve_field\s*\(")
_REQUIRE_RE = re.compile(r"(?<!\.)(?<!def\s)\brequire_field\s*\(")
_REGISTRY_RESOLVE_RE = re.compile(r"\bFIELD_REGISTRY\s*\.\s*resolve")
_REGISTRY_GET_RE = re.compile(r"\bFIELD_REGISTRY\s*\.\s*get\s*\(")

#: Files that legitimately implement or re-export the legacy API.
_LEGACY_ALLOWLIST = {
    "fields/resolver.py",
    "fields/__init__.py",
}

#: Documented ``market=None`` fallback pattern: a bare legacy call is allowed
#: ONLY when the SAME function/block routes through the market-aware resolver
#: first (``resolve_market_field`` / ``MULTI_MARKET_FIELD_REGISTRY`` appears
#: within the surrounding 40 lines) — i.e. the legacy call is the explicit
#: no-market fallback, not a primary path (R17-037).
#:
#: This is pattern-based (not line-based) so concurrent edits that shift lines do
#: not produce false positives.
_MARKET_AWARE_MARKER = re.compile(
    r"resolve_market_field|MULTI_MARKET_FIELD_REGISTRY"
)
_FALLBACK_WINDOW = 40


def _is_market_aware_fallback(rel: str, lines: list[str], lineno: int) -> bool:
    """True when a legacy call is an explicit market=None fallback.

    The surrounding window (``_FALLBACK_WINDOW`` lines before/after) must contain
    a market-aware resolver marker, proving the primary path routes per-market.
    """
    start = max(0, lineno - _FALLBACK_WINDOW)
    end = min(len(lines), lineno + _FALLBACK_WINDOW)
    for line in lines[start:end]:
        if _MARKET_AWARE_MARKER.search(line):
            return True
    return False


def _is_production_source(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    # REPO is the ``factor_engine/`` package itself; anything under tests/tools/
    # scripts/build/docs (or the legacy allowlist) is not a production caller.
    return not any(
        rel == prefix or rel.startswith(prefix + "/")
        for prefix in ("tests", "tools", "scripts", "build", "docs")
    )


def scan() -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for py in REPO.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if not _is_production_source(py):
            continue
        rel = py.relative_to(REPO).as_posix()
        if rel in _LEGACY_ALLOWLIST:
            continue
        text = py.read_text(encoding="utf-8")
        lines = text.splitlines()
        hits: list[str] = []
        for lineno, line in enumerate(lines, 1):
            # R17-001: a bare legacy call is allowed ONLY when the surrounding
            # block also has a market-aware primary path (explicit market=None
            # fallback), never as a primary resolver.
            def _is_fallback() -> bool:
                return _is_market_aware_fallback(rel, lines, lineno)

            if _IMPORT_RE.search(line) and not line.strip().startswith("#"):
                if not _is_fallback():
                    hits.append(f"{lineno}: bare `from fields import ... resolve_field`")
                continue
            if (
                (_CALL_RE.search(line) and "resolve_market_field" not in line)
                and not line.strip().startswith("#")
            ):
                if not _is_fallback():
                    hits.append(f"{lineno}: bare `resolve_field(...)` (legacy A-share bound)")
            if _REQUIRE_RE.search(line) and not line.strip().startswith("#"):
                if not _is_fallback():
                    hits.append(f"{lineno}: bare `require_field(...)` (legacy A-share bound)")
            if _REGISTRY_RESOLVE_RE.search(line):
                if not _is_fallback():
                    hits.append(f"{lineno}: bare `FIELD_REGISTRY.resolve*`")
            if _REGISTRY_GET_RE.search(line):
                if not _is_fallback():
                    hits.append(f"{lineno}: bare `FIELD_REGISTRY.get(...)`")
        if hits:
            findings[rel] = hits
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("R17-001: OK — no production bare legacy field resolution.")
        return 0
    print("R17-001: FAIL — production sources using legacy A-share field resolution:")
    for rel in sorted(findings):
        print(f"  {rel}")
        for hit in findings[rel]:
            print(f"    {hit}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
