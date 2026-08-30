#!/usr/bin/env python3
"""Generate post-deduplication operator runtime and surface catalogs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = FE_ROOT / "cleaned_operators" / "docs" / "operators_catalog.md"
OUT_JSON = FE_ROOT / "cleaned_operators" / "docs" / "operators_catalog.json"


def _load():
    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    if str(FE_ROOT) not in sys.path:
        sys.path.insert(0, str(FE_ROOT))
    # Import order matters for deterministic policy inference: importing
    # ``operator_surface`` / ``registry`` BEFORE ``load_all()`` triggers operator
    # module registration side-effects that run against a partially-loaded
    # registry, sealing some promoted operators' explicit policies to False
    # (observed for ts_quantile_beta_spread / ts_variance_ratio_slope).  The
    # manifest generator imports load_all first, so both manifests must observe
    # the identical post-load ``infer_operator_policy`` state.
    from factor_engine.cleaned_operators import load_all

    load_all()  # includes apply_operator_deduplication()

    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
    from factor_engine.cleaned_operators.operator_surface import classify_canonical, surface_summary
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
    return OperatorRegistry, infer_operator_policy, classify_canonical, surface_summary


def _payload(registry, infer_policy, classify_canonical, surface_summary) -> dict:
    rows: list[dict] = []
    for canonical in registry.list_canonical():
        operator = registry.get(canonical, mode="any")
        meta = dict(registry.catalog().get(canonical) or {})
        from factor_engine.cleaned_operators.operator_spec import build_operator_spec
        spec = build_operator_spec(canonical)
        if operator is not None:
            policy = infer_policy(operator, canonical=canonical)
        elif spec is not None:
            policy = None
        else:
            policy = None
        rows.append(
            {
                "canonical": canonical,
                "surface": classify_canonical(canonical),
                "aliases": sorted(meta.get("aliases") or []),
                "backends": sorted(meta.get("backends") or []),
                "pit_safe": bool(getattr(spec, "pit_safe", None)) if operator is None and spec is not None else (bool(policy.pit_safe) if policy else None),
                "scope": str(policy.scope) if policy else (str(getattr(spec, "scope", "")) if spec is not None else None),
                "lookback": policy.lookback_window if policy else None,
                "min_periods": policy.min_periods if policy else None,
                "lag": policy.lag if policy else None,
            }
        )
        continue
        meta = dict(registry.catalog().get(canonical) or {})
        policy = infer_policy(operator, canonical=canonical)
        rows.append(
            {
                "canonical": canonical,
                "surface": classify_canonical(canonical),
                "aliases": sorted(meta.get("aliases") or []),
                "backends": sorted(meta.get("backends") or []),
                "pit_safe": bool(policy.pit_safe) if policy else None,
                "scope": str(policy.scope) if policy else None,
                "lookback": policy.lookback_window if policy else None,
                "min_periods": policy.min_periods if policy else None,
                "lag": policy.lag if policy else None,
            }
        )
    return {
        "schema_version": 2,
        "canonical_count": len(rows),
        "surface_counts": surface_summary([row["canonical"] for row in rows]),
        "operators": rows,
    }


def _render(payload: dict) -> str:
    counts = payload["surface_counts"]
    lines = [
        "# Operators Catalog（自动生成）",
        "",
        "> 从 `load_all()` 去重后的最终 runtime registry 生成。",
        "> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。",
        "",
        "## 摘要",
        "",
        f"- canonical 总数：{payload['canonical_count']}",
        f"- daily：{counts['daily']}",
        f"- research：{counts['research']}",
        f"- unsafe：{counts['unsafe']}",
        f"- legacy：{counts['legacy']}",
        "",
        "| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["operators"]:
        lines.append(
            "| {canonical} | {surface} | {backends} | {aliases} | {pit_safe} | {scope} | {lookback} | {min_periods} | {lag} |".format(
                canonical=row["canonical"],
                surface=row["surface"],
                backends=", ".join(row["backends"]),
                aliases=", ".join(row["aliases"]),
                pit_safe=row["pit_safe"],
                scope=row["scope"],
                lookback=row["lookback"],
                min_periods=row["min_periods"],
                lag=row["lag"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    registry, infer_policy, classify_canonical, surface_summary = _load()
    payload = _payload(registry, infer_policy, classify_canonical, surface_summary)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
