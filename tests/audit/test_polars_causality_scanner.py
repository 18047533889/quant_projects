"""Bounded static audit for future-looking Polars production canonicals.

This is intentionally a detector, not a bulk rewrite or a semantic proof.  It
scans only the checked-in ``cleaned_operators/polars_native`` canonicals and
reports source locations requiring manual review.  Explicit research,
future-label, and target-generation modules are excluded because those are
allowed to use forward information by contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_ROOT = Path(__file__).resolve().parents[2]
_POLARS_NATIVE = _ROOT / "factor_engine" / "cleaned_operators" / "polars_native"

# Keep this bounded to expressions that can make a value at t depend on t+1,
# rather than trying to infer every possible Python/NumPy data dependency.
_PATTERNS = (
    ("negative_shift", re.compile(r"\.shift\s*\(\s*-\s*\d+")),
    ("centered_rolling", re.compile(r"(?:rolling_[a-z_]+|rolling)\s*\([^\n]*\bcenter\s*=\s*True")),
    ("forward_keyword", re.compile(r"\b(?:lead|forward|future)\s*\(")),
)
_EXCLUDED_PARTS = frozenset({"research", "future_label", "future_labels", "labels"})


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    text: str
    canonical: str | None


def _excluded(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    return bool(parts & _EXCLUDED_PARTS) or any(
        token in stem for token in ("research", "future", "label", "target")
    )


def _canonical_before(lines: list[str], line_index: int) -> str | None:
    """Return the nearest declared canonical/name for useful review output."""
    window = lines[max(0, line_index - 30) : line_index + 1]
    for source in reversed(window):
        match = re.search(r"\b(?:canonical|name)\s*=\s*[\"']([^\"']+)", source)
        if match:
            return match.group(1)
    return None


def scan_production_polars_canonicals() -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(_POLARS_NATIVE.glob("*.py")):
        if _excluded(path):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, source in enumerate(lines):
            # Comments/docstrings can describe an intentional forward label;
            # only flag executable-looking lines for this bounded scan.
            stripped = source.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for kind, pattern in _PATTERNS:
                if pattern.search(source):
                    findings.append(
                        Finding(
                            path=path,
                            line=index + 1,
                            kind=kind,
                            text=stripped,
                            canonical=_canonical_before(lines, index),
                        )
                    )
    return findings


def test_bounded_scanner_detects_forward_polars_code_without_importing_production() -> None:
    findings = scan_production_polars_canonicals()
    assert findings, "scanner unexpectedly found no future-looking Polars expressions"

    # The bounded scanner continues to surface the documented Ichimoku leading
    # spans for manual timing review; it must not silently become vacuous.
    assert any(
        finding.kind == "negative_shift"
        and finding.path.name == "panel_group_misc.py"
        and "senkou" in finding.text
        for finding in findings
    )


def test_scanner_excludes_explicit_research_or_label_modules() -> None:
    for path in _POLARS_NATIVE.glob("*.py"):
        if _excluded(path):
            assert not any(finding.path == path for finding in scan_production_polars_canonicals())
