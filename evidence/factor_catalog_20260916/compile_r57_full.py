# -*- coding: utf-8 -*-
"""R57: streaming re-compile of the whole factor catalog with the CURRENT tree.

Compile only. No data reads, no execution, no publication.

Differences from the R20 compile preflight (evidence/factor_catalog_20260916/
compile_r20_full.py):

* the source is the R20 revision CSV, so both the R20 formula and the formula
  produced by the *current* migration chain are recorded side by side;
* every row also records the operator surface it needs: the operator names
  actually called, their canonical, whether they resolve at all, whether they
  resolve in production mode, and their status. That is what stage A of the
  handover asks for and what the R20 manifest never recorded;
* the whole engine and source tree is hashed before and after, so the manifest
  is pinned to a code identity.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

R20_SHA = "80044269dc29b209dfae5136488bde50adcfc283dd8ee2d90147746ce89463d5"

_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# DSL surface syntax that looks like a call but is not a registered operator.
_DSL_KEYWORDS = frozenset({"field", "where", "if_else", "iif", "and_", "or_", "not_"})

_PARSER_ALLOWED: frozenset[str] | None = None


def parser_allowed_names() -> frozenset[str]:
    """Names the DSL parser resolves itself on the compile surface.

    The catalog is compiled with surface="compat_research", where dsl_parser
    installs its own native macros (holder_top10_* and source_col) on top of the
    registry allowlist. Those names are reachable with no registry entry at all,
    so registry absence alone must not be reported as a missing operator.
    """
    global _PARSER_ALLOWED
    if _PARSER_ALLOWED is None:
        try:
            from factor_engine.api.dsl_parser import _ExprBuilder

            built = _ExprBuilder(surface="compat_research", dialect="native")._allowed
            _PARSER_ALLOWED = frozenset(str(name) for name in built)
        except Exception:  # noqa: BLE001
            _PARSER_ALLOWED = frozenset()
    return _PARSER_ALLOWED


def digest(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def collect_call_names(formula: str) -> list[str]:
    """Operator names called in a DSL formula (language keywords filtered later)."""
    return sorted(set(_CALL_RE.findall(formula or "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-sha", default=R20_SHA)
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--rebuild-every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="debug only: stop after N rows")
    args = ap.parse_args()

    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    if digest(args.source) != args.source_sha:
        raise ValueError("source hash mismatch")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "evidence" / "factor_catalog_20260915"))

    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields

    from factor_engine.api.factor import Factor
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
    from factor_engine.tools.catalog_recipe_migration import (
        migrate_catalog_recipe_formula,
        redesign_catalog_intraday_limit_formula,
    )
    from factor_engine.tools.catalog_sketch_migration import migrate_catalog_sketch_formula

    code_paths = sorted(
        set(
            p
            for p in itertools.chain(
                (root / "factor_engine").rglob("*.py"),
                (root / "data_access").rglob("*.py"),
            )
        )
    ) + [Path(__file__)]
    hashes = lambda: {str(p.relative_to(root)): digest(p) for p in code_paths}  # noqa: E731
    before = hashes()

    parser, engine = build_runtime()
    surface_manifest = {
        "catalog_size": len(OperatorRegistry._catalog),
        "alias_count": len(OperatorRegistry._aliases),
        "names": sorted(OperatorRegistry._catalog),
    }

    output = Path(str(args.prefix) + ".jsonl.gz")
    summary_path = Path(str(args.prefix) + ".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(args.prefix)

    counts: collections.Counter[str] = collections.Counter()
    missing_ops: collections.Counter[str] = collections.Counter()
    nonprod_ops: collections.Counter[str] = collections.Counter()
    parser_ops: collections.Counter[str] = collections.Counter()
    parser_allowed = parser_allowed_names()
    op_status: dict[str, str] = {}
    error_groups: collections.Counter[str] = collections.Counter()
    total = 0
    start = time.monotonic()

    with gzip.open(args.source, "rt", encoding="utf-8-sig", newline="") as src, gzip.open(
        output, "xt", encoding="utf-8"
    ) as dst:
        for row in csv.DictReader(src):
            if not row.get("id"):
                continue
            index = total
            total += 1
            if index % args.shards != args.shard:
                continue

            rec = {
                "source_row": row.get("source_row"),
                "id": row["id"],
                "name": row.get("name"),
                "original_formula": row.get("original_formula"),
                "r20_formula": row.get("current_formula"),
                "r20_compile_status": row.get("compile_status"),
                "r57_formula": None,
                "migration_changes": [],
                "bindings": [],
                "binding_failures": [],
                "compile_status": None,
                "compile_error_type": "",
                "compile_error": "",
                "ops_used": [],
            }
            try:
                formula = rec["r20_formula"]
                if not isinstance(formula, str) or not formula.strip():
                    raise ValueError("MISSING_FORMULA")

                # Identical order and flags to compile_catalog.compile_record.
                sketch = migrate_catalog_sketch_formula(formula, enabled=True, parser=parser)
                redesign = redesign_catalog_intraday_limit_formula(
                    sketch.formula, enabled=False
                )
                price = migrate_adjusted_price_fields(redesign.formula, market="ashare")
                recipe = migrate_catalog_recipe_formula(price.formula)
                changes = (("parameter sketch -> keyword call",) if sketch.converted else ())
                changes += tuple(redesign.changes) + tuple(price.changes) + tuple(recipe.changes)
                rec["r57_formula"] = recipe.formula
                rec["migration_changes"] = list(changes)

                expr = parser.parse(recipe.formula)
                rec["bindings"], rec["binding_failures"] = bind_fields(expr)
                if rec["binding_failures"]:
                    raise ValueError(
                        "FIELD_BINDING_FAILED: " + json.dumps(rec["binding_failures"], ensure_ascii=False)
                    )

                ops = []
                for call in collect_call_names(recipe.formula):
                    try:
                        canonical = OperatorRegistry.resolve_canonical_optional(call)
                    except Exception:  # noqa: BLE001
                        canonical = None
                    if canonical is None and call in _DSL_KEYWORDS:
                        continue
                    # NOTE: OperatorRegistry.get(name, backend="any") always returns
                    # None -- "any" is not a real backend. Availability must be read
                    # from the canonical's registered backend set instead.
                    backends: list[str] = []
                    if canonical:
                        try:
                            backends = sorted(map(str, OperatorRegistry.backends_for(canonical) or []))
                        except Exception:  # noqa: BLE001
                            backends = []
                    entry = OperatorRegistry._catalog.get(canonical or call) or {}
                    status = entry.get("status")
                    prod_backends = []
                    for backend in backends:
                        try:
                            if OperatorRegistry.get(canonical, backend, mode="production") is not None:
                                prod_backends.append(backend)
                        except Exception:  # noqa: BLE001
                            continue
                    ops.append(
                        {
                            "call": call,
                            "canonical": canonical,
                            "resolved": bool(backends),
                            "parser_resolved": call in parser_allowed,
                            "backends": backends,
                            "production_admitted": bool(prod_backends),
                            "status": status,
                        }
                    )
                    if not backends and call in parser_allowed:
                        parser_ops[call] += 1
                    elif not backends:
                        missing_ops[call] += 1
                    elif not prod_backends:
                        nonprod_ops[call] += 1
                    if canonical:
                        op_status[canonical] = status
                rec["ops_used"] = ops

                engine.compile(
                    Factor(
                        name=rec["id"],
                        expr=expr,
                        source_expr=recipe.formula,
                        surface="compat_research",
                    )
                )
                rec["compile_status"] = "COMPILED"
            except Exception as exc:  # noqa: BLE001
                rec["compile_status"] = "COMPILE_FAILED"
                rec["compile_error_type"] = type(exc).__name__
                rec["compile_error"] = str(exc)[:1500]
                error_groups[f"{type(exc).__name__}: {str(exc)[:200]}"] += 1

            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[rec["compile_status"]] += 1

            processed = sum(counts.values())
            if args.limit and processed >= args.limit:
                break
            if processed % args.rebuild_every == 0:
                dst.flush()
                print(
                    json.dumps(
                        {"shard": args.shard, "counts": counts, "seconds": round(time.monotonic() - start, 1)}
                    ),
                    flush=True,
                )
                parser, engine = build_runtime()

    result = {
        "scope": "real_engine_strict_field_bind_compile_only_no_read_no_execution",
        "source_sha256": args.source_sha,
        "output_sha256": digest(output),
        "shard": args.shard,
        "shards": args.shards,
        "source_factor_count": total,
        "counts": dict(counts),
        "seconds": round(time.monotonic() - start, 1),
        "registry_surface": {
            "catalog_size": surface_manifest["catalog_size"],
            "alias_count": surface_manifest["alias_count"],
        },
        "operators_missing_from_surface": dict(missing_ops),
        "operators_resolved_by_parser": dict(parser_ops),
        "operators_not_production_admitted": dict(nonprod_ops),
        "canonical_status_seen": op_status,
        "error_groups": dict(error_groups.most_common(200)),
        "code_hashes_before": before,
        "code_hashes_after": hashes(),
        "git_head": None,
    }
    with summary_path.open("x") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "code_hashes_before"}, ensure_ascii=False)[:4000], flush=True)


if __name__ == "__main__":
    main()
