#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R28 Phase 1: fresh canonical inventory bound to the current git HEAD.

Fresh Python process -> ``load_all()`` -> enumerate OperatorRegistry.  Every
canonical is recorded with surface classification, production certification,
metadata, source module/class, parameter contract and backends.  Output is
written to ``docs/evidence/r28/R28_OPERATOR_INVENTORY.{csv,json}`` and the
canonical-set digest is printed so every other R28 artifact binds the same set.

Run:  python3 scripts/audit_r28_inventory.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "evidence" / "r28"


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def canonical_set_digest(canonicals: list[str]) -> str:
    blob = "\n".join(canonicals).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _safe(v) -> str:
    if v is None:
        return ""
    return str(v)


def main() -> None:
    _load()
    OUT.mkdir(parents=True, exist_ok=True)

    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.operator_surface import (
        classify_canonical,
        production_certification,
    )

    load_all()

    canonicals = sorted(OperatorRegistry.list_canonical())
    aliases = dict(getattr(OperatorRegistry, "_aliases", {}) or {})
    sha = git_sha()
    digest = canonical_set_digest(canonicals)

    rows = []
    surface_counts: dict[str, int] = {}
    cert_counts: dict[str, int] = {}
    for name in canonicals:
        surface = ""
        try:
            surface = classify_canonical(name)
        except Exception:
            surface = "error"
        surface_counts[surface] = surface_counts.get(surface, 0) + 1

        cert = ""
        try:
            cert = production_certification(name).name
        except Exception:
            cert = "ERROR"
        cert_counts[cert] = cert_counts.get(cert, 0) + 1

        meta = None
        backends_map = OperatorRegistry._operators.get(name, {}) or {}
        backends = sorted(backends_map.keys())
        src_module, src_class = "", ""
        op = None
        if backends:
            chosen = "pandas_numpy" if "pandas_numpy" in backends_map else backends[0]
            op = backends_map.get(chosen)
            if op is not None:
                src_module = type(op).__module__
                src_class = type(op).__name__
        try:
            meta = getattr(op, "metadata", None) or (OperatorRegistry.get(name, "pandas_numpy").metadata if backends else None)
        except Exception:
            meta = None

        category = _safe(getattr(meta, "category", ""))
        business_category = _safe(getattr(meta, "business_category", ""))
        param_names = list(getattr(meta, "param_names", None) or ())
        param_specs = getattr(meta, "param_specs", None) or {}
        default_params = {}
        for p in param_names:
            spec = param_specs.get(p)
            if spec is not None:
                dv = getattr(spec, "default", None)
                if dv is not None:
                    default_params[p] = _safe(dv)
        role = _safe(getattr(meta, "role", ""))
        input_grain = _safe(getattr(meta, "input_grain", ""))
        output_grain = _safe(getattr(meta, "output_grain", ""))
        input_units = list(getattr(meta, "input_units", None) or ())
        output_unit = _safe(getattr(meta, "output_unit", ""))
        input_arity = _safe(getattr(meta, "input_arity", ""))
        window_semantics = _safe(getattr(meta, "window_semantics", ""))
        description = _safe(getattr(meta, "description", ""))
        tags = list(getattr(meta, "tags", None) or ())

        rows.append(
            {
                "canonical": name,
                "aliases": "|".join(sorted(k for k, v in aliases.items() if v == name)),
                "source_file": src_module,
                "source_symbol": src_class,
                "category": category,
                "business_category": business_category,
                "surface": surface,
                "production_certification": cert,
                "role": role,
                "terminal_allowed": _safe(getattr(meta, "terminal_allowed", "")),
                "deterministic": _safe(getattr(meta, "deterministic", "")),
                "pit_safe_claim": _safe(getattr(meta, "pit_safe", "")),
                "stateful": _safe(getattr(meta, "stateful", "")),
                "input_arity": input_arity,
                "input_grain": input_grain,
                "output_grain": output_grain,
                "input_units": "|".join(_safe(u) for u in input_units),
                "output_unit": output_unit,
                "params": "|".join(param_names),
                "default_params": json.dumps(default_params, ensure_ascii=False, sort_keys=True),
                "backends": "|".join(backends),
                "window_semantics": window_semantics,
                "tags": "|".join(_safe(t) for t in tags),
                "description": description[:200],
            }
        )

    inv = {
        "schema_version": "factor_engine.r28.inventory.v1",
        "git_sha": sha,
        "canonical_set_digest": digest,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version.split()[0],
        "total_canonicals": len(rows),
        "surface_counts": dict(sorted(surface_counts.items())),
        "certification_counts": dict(sorted(cert_counts.items())),
        "operators": rows,
    }

    json_path = OUT / "R28_OPERATOR_INVENTORY.json"
    json_path.write_text(json.dumps(inv, indent=1, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else ["canonical"]
    csv_path = OUT / "R28_OPERATOR_INVENTORY.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"R28 inventory: {json_path}")
    print(f"  git_sha={sha}")
    print(f"  canonical_set_digest={digest}")
    print(f"  total={len(rows)}")
    print(f"  surfaces={dict(sorted(surface_counts.items()))}")
    print(f"  certifications={dict(sorted(cert_counts.items()))}")
    with (OUT / "R28_CANONICAL_SET_DIGEST.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"git_sha={sha}\ncanonical_set_digest={digest}\ntotal={len(rows)}\n")


if __name__ == "__main__":
    main()
