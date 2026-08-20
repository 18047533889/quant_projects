"""Regression: no placeholder / TODO stub operators from target files are mining-selectable."""
from __future__ import annotations

import re
import pathlib
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from mining.operator_catalog import get_mining_operators
import cleaned_operators.operator_surface as S


_PLACEHOLDER_BODY_PATTERNS = [
    re.compile(r"result\[:\]\s*=\s*np\.nan"),
    re.compile(r"# Placeholder"),
    re.compile(r"pl\.Series\(\[None\]"),
    re.compile(r"pl\.lit\(None\)"),
    re.compile(r"fill_nan\(None\)"),
]

_TARGET_FILES = [
    pathlib.Path("cleaned_operators/polars_native/ts_advanced_batch1.py"),
    pathlib.Path("cleaned_operators/polars_native/ts_advanced_batch2.py"),
    pathlib.Path("cleaned_operators/polars_native/ts_advanced_batch3.py"),
    pathlib.Path("cleaned_operators/common/polars_ts_complex.py"),
]

_CLASS = re.compile(r"\nclass (\w+)\(.*?\):\n")


def _extract_stub_canonicals() -> list[str]:
    stubs: list[str] = []
    for path in _TARGET_FILES:
        src = path.read_text()
        parts = _CLASS.split(src)
        # parts = [preamble, name1, body1, name2, body2, ...]
        for i in range(1, len(parts), 2):
            cls_name = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            if "TODO" not in body:
                continue
            if any(p.search(body) for p in _PLACEHOLDER_BODY_PATTERNS):
                stubs.append(cls_name)
    return stubs


def test_todo_stub_quarantine_no_mining_selectable() -> None:
    load_all()
    stubs = _extract_stub_canonicals()
    assert len(stubs) > 100, "expected many stubs"
    catalog = OperatorRegistry._catalog
    mineable = {op.canonical for op in get_mining_operators(admission="all") if op.mining_eligible}

    violations = [n for n in stubs if n in mineable]
    assert not violations, f"mining-selectable placeholder stubs: {violations[:20]}"

    # none should have research_only status via batch5-style quarantine today
    research_only = [
        n
        for n in stubs
        if n in catalog
        and str(catalog[n].get("status", "")).lower() == "research_only"
    ]
    # This is a regression guard: today none carry research_only, so the list is empty.
    # Once the intended quarantine is added, flip this assertion to len(research_only) == len(stubs).
    assert research_only == [], f"unexpected research_only stubs today: {research_only}"

    # Final hard guard: no stub may be mining-selectable under any admission mode.
    for mode in ("eligible", "pending"):
        mineable_by_mode = {
            op.canonical
            for op in get_mining_operators(admission=mode)
            if op.mining_eligible
        }
        still_mineable = [n for n in stubs if n in mineable_by_mode]
        assert not still_mineable, f"mode={mode} mineable placeholder stubs: {still_mineable[:20]}"


def test_batch5_quarantine_pattern_exists() -> None:
    source = pathlib.Path("cleaned_operators/polars_native/ts_advanced_batch5.py").read_text()
    assert 'kwargs["status"] = "research_only"' in source
    assert "register_operator as _register_operator" in source


def test_quarantined_modules_do_not_register_new_canonicals_into_registry() -> None:
    """After a full load, importing the target modules must not add new canonicals."""
    load_all()
    before = set(OperatorRegistry._catalog)

    import importlib
    for mod in (
        "cleaned_operators.polars_native.ts_advanced_batch1",
        "cleaned_operators.polars_native.ts_advanced_batch3",
        "cleaned_operators.common.polars_ts_complex",
    ):
        try:
            importlib.import_module(mod)
        except (ValueError, RuntimeError):
            # governance/registry rejects post-load duplicate or frozen-state re-registration;
            # the goal is that NO NEW canonical was added, which we verify below.
            pass

    after = set(OperatorRegistry._catalog)
    added = sorted(after - before)
    assert not added, f"unexpected post-load registry additions: {added[:20]}"


def test_classified_daily_implied_by_surface_sets() -> None:
    """classify_canonical('daily') implies canonical is in DAILY_CANONICALS or DAILY_FACTOR_MIGRATED."""
    load_all()
    daily = S.DAILY_CANONICALS
    migrated = S.DAILY_FACTOR_MIGRATED
    for canonical in sorted(OperatorRegistry.list_canonical()):
        if S.classify_canonical(canonical) == "daily":
            assert canonical in daily or canonical in migrated, f"daylisted but not in surface sets: {canonical}"


def test_evidence_yaml_exists_and_matches_audit() -> None:
    evidence = pathlib.Path("evidence/r2/R21-TODO-STUB-AUDIT.yaml").read_text()
    assert "R21-TODO-STUB-AUDIT" in evidence
    assert "ts_advanced_batch1" in evidence
    assert "ts_advanced_batch2" in evidence
    assert "ts_advanced_batch3" in evidence
    assert "polars_ts_complex" in evidence
