#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"
CATALOG = FE / "cleaned_operators" / "docs" / "operators_catalog.json"


def _quoted_set(name: str, values: set[str]) -> str:
    body = "\n".join(f'        {value!r},' for value in sorted(values))
    return f"{name}: frozenset[str] = frozenset(\n    {{\n{body}\n    }}\n)"


def write_fail_closed_surface() -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_surface: dict[str, set[str]] = {}
    for row in payload["operators"]:
        by_surface.setdefault(str(row.get("surface", "unclassified")), set()).add(row["canonical"])

    daily = set(by_surface.get("daily", set()))
    research = set(by_surface.get("research", set()))
    unsafe = set(by_surface.get("unsafe", set()))
    legacy = set(by_surface.get("legacy", set()))

    research_add = {
        "ACF",
        "Mode",
        "autocorr",
        "pacf",
        "max_drawdown",
        "sharpe_ratio",
        "sem",
        "lasso",
        "ridge",
        "regress",
        "residual",
        "r_squared",
    }
    research_add |= {name for name in daily if name.startswith("row_")}
    unsafe_add = {"causal_bfill"}
    internal_add = {"constant"}

    research |= research_add & daily
    unsafe |= unsafe_add & daily
    internal = internal_add & daily
    daily -= research_add | unsafe_add | internal_add

    hidden_daily_names = {
        "inv",
        "reciprocal",
        "fmax",
        "fmin",
        "sqr",
        "cube",
        "cumulative_max",
        "cumulative_mean",
        "cumulative_min",
    }

    source = f'''# -*- coding: utf-8 -*-
"""Fail-closed operator surface policy for factor formula authoring.

Only canonicals explicitly listed in ``DAILY_CANONICALS`` may enter new daily
factor DSL formulas. Runtime registration is not sufficient: newly registered
operators remain ``unclassified`` until reviewed and assigned deliberately.
"""
from __future__ import annotations

from typing import Iterable, Literal

OperatorSurface = Literal[
    "daily", "research", "unsafe", "legacy", "internal", "unclassified", "all"
]

{_quoted_set("DAILY_CANONICALS", daily)}

{_quoted_set("RESEARCH_ONLY_CANONICALS", research)}

{_quoted_set("UNSAFE_CANONICALS", unsafe)}

{_quoted_set("LEGACY_ONLY_CANONICALS", legacy)}

{_quoted_set("INTERNAL_ONLY_CANONICALS", internal)}

{_quoted_set("HIDDEN_DAILY_NAMES", hidden_daily_names)}


def classify_canonical(canonical: str) -> str:
    """Return the reviewed surface; unknown registrations fail closed."""
    if canonical in DAILY_CANONICALS:
        return "daily"
    if canonical in RESEARCH_ONLY_CANONICALS:
        return "research"
    if canonical in UNSAFE_CANONICALS:
        return "unsafe"
    if canonical in LEGACY_ONLY_CANONICALS:
        return "legacy"
    if canonical in INTERNAL_ONLY_CANONICALS:
        return "internal"
    return "unclassified"


def is_dsl_name_allowed(name: str, canonical: str, *, surface: OperatorSurface = "daily") -> bool:
    """Return whether a registry name is visible on the requested surface."""
    if surface == "all":
        return True
    category = classify_canonical(canonical)
    if surface == "daily":
        return category == "daily" and name not in HIDDEN_DAILY_NAMES
    if surface == "research":
        return category == "research"
    if surface == "unsafe":
        return category == "unsafe"
    if surface == "legacy":
        return category == "legacy" or name in HIDDEN_DAILY_NAMES
    if surface == "internal":
        return category == "internal"
    if surface == "unclassified":
        return category == "unclassified"
    raise ValueError(f"unknown operator surface: {{surface!r}}")


def unclassified_canonicals(canonicals: Iterable[str]) -> tuple[str, ...]:
    """Return runtime canonicals that have not passed surface review."""
    return tuple(sorted(c for c in canonicals if classify_canonical(c) == "unclassified"))


def surface_summary(canonicals: Iterable[str]) -> dict[str, int]:
    """Count canonical operators by reviewed surface."""
    out = {{
        "daily": 0,
        "research": 0,
        "unsafe": 0,
        "legacy": 0,
        "internal": 0,
        "unclassified": 0,
    }}
    for canonical in canonicals:
        out[classify_canonical(canonical)] += 1
    return out
'''
    (FE / "cleaned_operators" / "operator_surface.py").write_text(source, encoding="utf-8")


def patch_polars_stateful_policy() -> None:
    path = FE / "backend" / "polars_long_policy.py"
    text = path.read_text(encoding="utf-8")

    def read_set(var: str) -> set[str]:
        match = re.search(
            rf"{re.escape(var)}: frozenset\[str\] = frozenset\(\n    \{{\n(.*?)\n    \}}\n\)",
            text,
            flags=re.S,
        )
        if not match:
            raise RuntimeError(f"cannot locate {var}")
        return set(re.findall(r'[\"\']([^\"\']+)[\"\']', match.group(1)))

    def replace_set(src: str, var: str, values: set[str]) -> str:
        pattern = rf"{re.escape(var)}: frozenset\[str\] = frozenset\(\n    \{{\n.*?\n    \}}\n\)"
        replacement = _quoted_set(var, values)
        updated, count = re.subn(pattern, replacement, src, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"cannot replace {var}")
        return updated

    native = read_set("POLARS_LONG_NATIVE")
    stateful = read_set("POLARS_LONG_STATEFUL")
    recursive = {
        "ema",
        "RSI_WILDER",
        "ATR_WILDER",
        "MACD",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "KAMA",
        "TRIX",
        "ADX",
        "ADXR",
        "vpmacd",
        "vpmacd_signal",
    }
    native -= recursive
    stateful |= recursive
    text = replace_set(text, "POLARS_LONG_NATIVE", native)
    text = replace_set(text, "POLARS_LONG_STATEFUL", stateful)
    path.write_text(text, encoding="utf-8")


def canonicalize_evidence() -> None:
    sys.path.insert(0, str(FE))
    from cleaned_operators import load_all  # type: ignore
    from cleaned_operators.registry import OperatorRegistry  # type: ignore

    load_all()
    explicit = {
        "clip": "cap",
        "ts_delay": "delay",
        "safe_div_null": "safe_div",
        "ts_ema": "ema",
        "ts_regression": "ts_regression_slope",
        "ts_decay_linear": "decay_linear",
    }

    def resolve(name: str) -> str:
        current = explicit.get(name, name)
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            nxt = OperatorRegistry._aliases.get(current, current)
            if nxt == current:
                break
            current = nxt
        return current

    paths = list(FE.rglob("primitive_verified.json"))
    if not paths:
        raise RuntimeError("primitive_verified.json not found")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in list(payload.items()):
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                payload[key] = sorted({resolve(x) for x in value})
        operators = payload.get("operators")
        if isinstance(operators, dict):
            merged: dict[str, dict] = {}
            for old_name, metadata in operators.items():
                canonical = resolve(old_name)
                row = merged.setdefault(canonical, {})
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        row.setdefault(key, value)
                    aliases = set(row.get("certified_aliases", []))
                    if old_name != canonical:
                        aliases.add(old_name)
                    if aliases:
                        row["certified_aliases"] = sorted(aliases)
            payload["operators"] = merged
        payload["canonicalization"] = {
            "applied_after_registry_deduplication": True,
            "legacy_names_are_metadata_only": True,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_edge_policy() -> None:
    path = FE / "cleaned_operators" / "edge_requirements.py"
    path.write_text('''# -*- coding: utf-8 -*-
"""Operator-specific IEEE edge evidence requirements.

This module does not manufacture certification. It reports which production
operators still lack required NaN/Inf evidence so routing can remain fail-closed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

NAN_REQUIRED = frozenset({
    "c_mean", "c_std", "c_sum", "normalize", "zscore", "scale", "winsorize",
    "group_mean", "group_std", "group_zscore", "group_normalize", "group_rank",
    "ts_mean", "ts_std", "ts_var", "ts_corr", "ts_cov", "ts_beta", "ts_zscore",
    "ts_sharpe", "volatility", "maximum", "minimum", "where", "coalesce",
})
INF_REQUIRED = NAN_REQUIRED


def _factor_engine_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_primitive_evidence() -> dict:
    paths = sorted(_factor_engine_root().rglob("primitive_verified.json"))
    if not paths:
        return {}
    return json.loads(paths[0].read_text(encoding="utf-8"))


def required_edge_dimensions(canonical: str) -> frozenset[str]:
    required: set[str] = set()
    if canonical in NAN_REQUIRED:
        required.add("nan")
    if canonical in INF_REQUIRED:
        required.add("inf")
    return frozenset(required)


def missing_edge_dimensions(canonical: str, evidence: dict | None = None) -> frozenset[str]:
    payload = evidence if evidence is not None else load_primitive_evidence()
    missing: set[str] = set()
    required = required_edge_dimensions(canonical)
    if "nan" in required and canonical not in set(payload.get("duckdb_nan_edge_verified", [])):
        missing.add("nan")
    if "inf" in required and canonical not in set(payload.get("duckdb_inf_edge_verified", [])):
        missing.add("inf")
    return frozenset(missing)


def production_edge_evidence_complete(canonical: str, evidence: dict | None = None) -> bool:
    return not missing_edge_dimensions(canonical, evidence)
''', encoding="utf-8")


def patch_backend_capability_gate() -> None:
    candidates: list[Path] = []
    for path in (FE / "backend").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "def _polars_status(canon" in text and "POLARS_PRODUCTION_SAFE" in text:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one backend capability registry, found {candidates}")
    path = candidates[0]
    text = path.read_text(encoding="utf-8")
    old = '''    if canon in POLARS_PRODUCTION_SAFE:\n        return "production_safe"\n'''
    new = '''    if canon in POLARS_PRODUCTION_SAFE:\n        from cleaned_operators.edge_requirements import production_edge_evidence_complete\n\n        return "production_safe" if production_edge_evidence_complete(canon) else "parity_verified"\n'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "production_edge_evidence_complete(canon)" not in text:
        raise RuntimeError("Polars production-safe capability anchor not found")
    path.write_text(text, encoding="utf-8")


def write_tests() -> None:
    path = FE / "tests" / "test_operator_production_hardening.py"
    path.write_text('''from __future__ import annotations

import json
from pathlib import Path

from api.operator_registry import build_dsl_allowlist
from backend.polars_long_policy import classify_plan_op
from cleaned_operators import load_all
from cleaned_operators.edge_requirements import missing_edge_dimensions
from cleaned_operators.operator_surface import (
    classify_canonical,
    unclassified_canonicals,
)
from cleaned_operators.registry import OperatorRegistry


def test_surface_is_fail_closed() -> None:
    assert classify_canonical("brand_new_unreviewed_operator") == "unclassified"
    assert "brand_new_unreviewed_operator" not in build_dsl_allowlist()


def test_runtime_registry_has_no_unreviewed_canonicals() -> None:
    load_all()
    assert unclassified_canonicals(OperatorRegistry.list_canonical()) == ()


def test_problematic_names_are_not_daily_dsl() -> None:
    allowed = build_dsl_allowlist()
    for name in (
        "causal_bfill", "ACF", "Mode", "autocorr", "pacf", "max_drawdown",
        "sharpe_ratio", "sem", "lasso", "ridge", "regress", "residual",
        "r_squared", "constant",
    ):
        assert name not in allowed
    assert classify_canonical("causal_bfill") == "unsafe"


def test_recursive_operators_are_not_polars_native() -> None:
    for name in (
        "ema", "RSI_WILDER", "ATR_WILDER", "MACD", "KAMA", "TRIX", "ADX", "ADXR"
    ):
        assert classify_plan_op(name) == "stateful"


def test_primitive_evidence_uses_final_canonical_names() -> None:
    old = {"clip", "ts_delay", "safe_div_null", "ts_ema", "ts_regression", "ts_decay_linear"}
    paths = sorted((Path(__file__).resolve().parents[1]).rglob("primitive_verified.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                assert old.isdisjoint(value)
        assert old.isdisjoint((payload.get("operators") or {}).keys())


def test_ieee_edge_policy_is_fail_closed_for_statistics() -> None:
    evidence = {"duckdb_nan_edge_verified": [], "duckdb_inf_edge_verified": []}
    assert missing_edge_dimensions("ts_std", evidence) == {"nan", "inf"}
    assert missing_edge_dimensions("abs", evidence) == set()
''', encoding="utf-8")


def write_docs() -> None:
    path = FE / "docs" / "operator_production_hardening.md"
    path.write_text('''# Operator production hardening

The public daily DSL is fail-closed. Registering a runtime does not make it
available to factor authors. Every canonical must be assigned explicitly to one
of: daily, research, unsafe, legacy, or internal. Unknown registrations are
`unclassified` and fail CI.

Key production rules:

- `causal_bfill` is unsafe and unavailable to daily formulas.
- Global diagnostics and model-fitting utilities are research-only.
- `constant` is an internal IR/runtime helper, not a public operator.
- Recursive indicators are stateful, not native Polars expressions.
- Primitive evidence is stored under the final post-dedup canonical name.
- High-risk statistical operators cannot be marked Polars production-safe until
  their required DuckDB NaN and Inf edge evidence exists.
''', encoding="utf-8")


def main() -> int:
    write_fail_closed_surface()
    patch_polars_stateful_policy()
    canonicalize_evidence()
    write_edge_policy()
    patch_backend_capability_gate()
    write_tests()
    write_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
