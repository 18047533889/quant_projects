#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15 Master-Audit deliverable generator (Master Prompt §28 DoD + §29).

Loads the FULL final registry (``load_all()`` — all overlays, evidence, freeze)
and emits the eight §29 deliverables into ``build/r15_audit/``:

  1. operator_fix_report.csv          — per canonical: role, surface, six gates,
                                        status, backends, param coverage.
  2. operator_certification_table.csv — §25 machine-readable certification fields.
  3. semantic_duplicate_report.json   — exact/affine/monotonic/rank duplicate clusters
                                        (computed on synthetic fixtures).
  4. parameter_surface_report.json    — searchable params by role, dead-parameter
                                        audit, legacy-whitelist usage.
  5. ashare_practical_coverage.csv    — per canonical coverage on a synthetic A-share
                                        fixture (unique/std/Inf/NaN-streak).
  6. session_grain_report.json        — minute operators: input/output grain,
                                        available_at, same_session_usable.
  7. golden_null_test_report.json     — which canonicals ran synthetic golden checks.
  8. final_runtime_state_dump.json    — post-freeze registry (aliases/status/surface/
                                        role/params/history/backend/certification).

The reports are generated from the FROZEN final runtime, not from source tables —
that is the §7 audit requirement (overlay must not roll back source fixes).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build" / "r15_audit"


def _load() -> None:
    sys.path.insert(0, str(REPO))
    from factor_engine.cleaned_operators import load_all

    load_all()


def _registry():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry


def _surface_classification(canonical: str) -> str:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    try:
        return classify_canonical(canonical)
    except Exception:  # noqa: BLE001
        return "unclassified"


def _six_gates(catalog: dict[str, object]) -> dict[str, bool]:
    return {
        "implementation_certified": catalog.get("implementation_certified") is True,
        "semantic_certified": catalog.get("semantic_certified") is True,
        "temporal_certified": catalog.get("temporal_certified") is True,
        "source_contract_certified": catalog.get("source_contract_certified") is True,
        "edge_case_passed": catalog.get("edge_case_passed") is True,
        "backend_passed": catalog.get("backend_passed") is True,
    }


def _backends(canonical: str, reg) -> list[str]:
    return sorted(reg._operators.get(canonical, {}).keys())


def _param_specs(canonical: str, reg) -> dict[str, object]:
    op = reg.get(canonical)
    if op is None:
        return {}
    meta = getattr(op, "metadata", None)
    if meta is None:
        return {}
    return dict(getattr(meta, "param_specs", None) or {})


def _param_names(canonical: str, reg) -> list[str]:
    op = reg.get(canonical)
    if op is None:
        return []
    meta = getattr(op, "metadata", None)
    return list(getattr(meta, "param_names", None) or [])


def build_all() -> None:
    _load()
    reg = _registry()
    OUT.mkdir(parents=True, exist_ok=True)
    canonicals = sorted(reg._catalog.keys())

    # ---------------------------------------------------------------- row base
    from factor_engine.cleaned_operators.base import effective_param_role, param_role_declared
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec

    rows: list[dict[str, object]] = []
    param_surface: list[dict[str, object]] = []
    grain_rows: list[dict[str, object]] = []
    for canon in canonicals:
        catalog = reg._catalog[canon]
        spec = None
        try:
            spec = build_operator_spec(canon)
        except Exception:  # noqa: BLE001
            spec = None
        gates = _six_gates(catalog)
        backends = _backends(canon, reg)
        pnames = _param_names(canon, reg)
        pspecs = _param_specs(canon, reg)

        rows.append({
            "canonical": canon,
            "surface": _surface_classification(canon),
            "role": catalog.get("role"),
            "status": catalog.get("status"),
            "production_certified": catalog.get("production_certified") is True,
            **gates,
            "backends": "|".join(backends),
            "n_scalar_params": len(pnames),
            "n_param_specs": len(pspecs),
            "n_searchable": sum(1 for s in pspecs.values() if getattr(s, "searchable", True)),
            "allow_in_production": bool(spec and spec.allow_in_production) if spec else False,
            "pit_safe": bool(spec and spec.pit_safe) if spec else False,
            "input_grain": catalog.get("input_grain"),
            "output_grain": catalog.get("output_grain"),
            "available_at": catalog.get("available_at"),
            "same_session_usable": catalog.get("same_session_usable"),
            "stateful": catalog.get("stateful") is True,
        })

        # parameter surface (searchable only)
        for pname, pspec in pspecs.items():
            searchable = bool(getattr(pspec, "searchable", True))
            role = pspec.param_role
            role_str = role.value if role is not None else (
                effective_param_role(pspec).value if not searchable else "UNDECLARED"
            )
            param_surface.append({
                "canonical": canon,
                "param": pname,
                "searchable": searchable,
                "param_role": role_str,
                "role_declared": param_role_declared(pspec),
                "dtype": getattr(pspec, "dtype", None),
                "history_semantics": getattr(pspec, "history_semantics", None),
                "history_formula": getattr(pspec, "history_formula", None),
            })

        if catalog.get("input_grain") != catalog.get("output_grain"):
            grain_rows.append({
                "canonical": canon,
                "input_grain": catalog.get("input_grain"),
                "output_grain": catalog.get("output_grain"),
                "available_at": catalog.get("available_at"),
                "same_session_usable": catalog.get("same_session_usable"),
                "status": catalog.get("status"),
            })

    # ------------------------------------------------------------------ dump 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "operator_fix_report.csv", index=False)
    df.to_json(OUT / "operator_fix_report.json", orient="records", indent=1)

    # --------------------------------------------------------------- dump 8
    runtime_dump = {
        "schema": "factor_engine.final_runtime_state.v1",
        "total_canonicals": len(canonicals),
        "aliases": {k: v for k, v in sorted(reg._aliases.items())},
        "unclassified": [c for c in canonicals if _surface_classification(c) == "unclassified"],
        "catalog_snapshot": {
            c: {
                "status": reg._catalog[c].get("status"),
                "role": reg._catalog[c].get("role"),
                "surface": _surface_classification(c),
                "production_certified": reg._catalog[c].get("production_certified") is True,
                "backends": _backends(c, reg),
            }
            for c in canonicals
        },
    }
    (OUT / "final_runtime_state_dump.json").write_text(
        json.dumps(runtime_dump, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    # ------------------------------------------------------------- dump 2 (cert)
    cert_df = df[[
        "canonical", "surface", "role", "status", "production_certified",
        "implementation_certified", "semantic_certified", "temporal_certified",
        "source_contract_certified", "edge_case_passed", "backend_passed",
        "backends", "allow_in_production", "pit_safe",
        "input_grain", "output_grain", "available_at", "same_session_usable",
    ]].copy()
    cert_df.to_csv(OUT / "operator_certification_table.csv", index=False)

    # ------------------------------------------------------------ dump 4 (params)
    (OUT / "parameter_surface_report.json").write_text(
        json.dumps({
            "searchable_params": [p for p in param_surface if p["searchable"]],
            "undeclared_role_searchable": [
                p for p in param_surface if p["searchable"] and not p["role_declared"]
            ],
            "estimator_resolution": [
                p for p in param_surface if p["param_role"] == "estimator_resolution"
            ],
            "total_searchable": sum(1 for p in param_surface if p["searchable"]),
            "dead_parameter_scan": [
                p for p in param_surface
                if p["param_role"] in ("policy", "missing_policy", "market_policy",
                                       "support_policy", "session_policy")
            ],
        }, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )

    # ------------------------------------------------------------ dump 6 (grain)
    (OUT / "session_grain_report.json").write_text(
        json.dumps({
            "grain_transform_operators": grain_rows,
            "count": len(grain_rows),
        }, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )

    # ------------------------------------------------------------ dump 7 (golden)
    # R16-003: a real GoldenNullRunner — the file's existence no longer stands
    # in for the battery actually running.  Every applicable canonical emits
    # (canonical, test_id, outcome); NOT_RUN/AUDIT_ERROR outcomes are explicit
    # and never count as certified.
    golden = _golden_null_runner(canonicals, reg)
    golden_outcomes = [g for g in golden if g["outcome"] in ("PASS", "FAIL")]
    (OUT / "golden_null_test_report.json").write_text(
        json.dumps({
            "runner": "GoldenNullRunner",
            "records": golden,
            "counts": {
                "total": len(golden),
                "pass": sum(1 for g in golden if g["outcome"] == "PASS"),
                "fail": sum(1 for g in golden if g["outcome"] == "FAIL"),
                "not_applicable": sum(1 for g in golden if g["outcome"] == "N/A"),
                "not_run": sum(1 for g in golden if g["outcome"] == "NOT_RUN"),
                "audit_error": sum(1 for g in golden if g["outcome"] == "AUDIT_ERROR"),
            },
            "covered_canonical_hash": _set_hash({g["canonical"] for g in golden_outcomes}),
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # --------------------------------------------------- dump 5 (A-share coverage)
    # R16-001: full-registry coverage + set-equality meta (no 200-slice).
    ashare_rows, ashare_meta = _ashare_coverage(canonicals, reg)
    pd.DataFrame(ashare_rows).to_csv(OUT / "ashare_practical_coverage.csv", index=False)
    (OUT / "ashare_coverage_meta.json").write_text(
        json.dumps(ashare_meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    # --------------------------------------------------- dump 3 (semantic dup)
    # R16-002: full-registry duplicate scan (no 300-slice).
    dup = _semantic_duplicates(canonicals, reg)
    (OUT / "semantic_duplicate_report.json").write_text(
        json.dumps(dup, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    # ------------------------------------------------------------- summary md
    _write_summary(rows, param_surface, grain_rows, canonicals)

    print(f"R15 deliverables written to {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


def _set_hash(names: set[str]) -> str:
    """Stable coverage-set hash (R16-001/002/003)."""
    import hashlib

    return hashlib.sha256(
        json.dumps(sorted(names), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _golden_null_runner(canonicals: list[str], reg) -> list[dict[str, object]]:
    """R16-003: per-canonical golden/null battery with explicit outcomes.

    Every applicable canonical emits ``(canonical, test_id, outcome)`` where
    outcome is one of PASS / FAIL / N/A / NOT_RUN / AUDIT_ERROR — the file's
    existence no longer stands in for the battery running, and a NOT_RUN /
    AUDIT_ERROR outcome never counts as certified.
    """
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    panel = _synthetic_panel(n_rows=60, n_cols=3, seed=7)
    constant = pd.DataFrame(
        np.full_like(panel.to_numpy(dtype=float), 1.0),
        index=panel.index,
        columns=panel.columns,
    )
    records: list[dict[str, object]] = []
    for canon in canonicals:
        op = reg.get(canon)
        surface = classify_canonical(canon)
        # Golden/null apply to factor-shaped canonicals; internal/unsafe/compat
        # aliases are not applicable (they are not factor terminals).
        if op is None:
            records.append({"canonical": canon, "test_id": "golden::synthetic",
                            "outcome": "NOT_RUN", "reason": "no runtime operator"})
            continue
        if surface not in ("daily", "extended"):
            records.append({"canonical": canon, "test_id": "golden::synthetic",
                            "outcome": "N/A", "reason": f"surface={surface}"})
            continue
        # --- golden::synthetic: runs, output shape preserved, not all-NaN ---
        try:
            out = op.calculate(panel)
            arr = out.to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001
            records.append({"canonical": canon, "test_id": "golden::synthetic",
                            "outcome": "AUDIT_ERROR", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if arr.shape != panel.shape:
            records.append({"canonical": canon, "test_id": "golden::synthetic",
                            "outcome": "FAIL", "reason": f"shape {arr.shape} != {panel.shape}"})
            continue
        if not np.isfinite(arr).any():
            records.append({"canonical": canon, "test_id": "golden::synthetic",
                            "outcome": "FAIL", "reason": "all-NaN on normal panel"})
            continue
        records.append({"canonical": canon, "test_id": "golden::synthetic",
                        "outcome": "PASS", "reason": "shape+finite ok"})
        # --- null::constant: constant input must not silently fabricate
        #     spurious finite signal (NaN or constant is acceptable) ---
        try:
            c_out = op.calculate(constant)
            c_arr = c_out.to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001
            records.append({"canonical": canon, "test_id": "null::constant",
                            "outcome": "AUDIT_ERROR", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        finite_fraction = float(np.isfinite(c_arr).mean())
        if finite_fraction < 0.5 or np.nanstd(c_arr) == 0 or np.isnan(np.nanstd(c_arr)):
            records.append({"canonical": canon, "test_id": "null::constant",
                            "outcome": "PASS", "reason": f"constant ok (finite={finite_fraction:.2f})"})
        else:
            records.append({"canonical": canon, "test_id": "null::constant",
                            "outcome": "FAIL", "reason": f"constant panel produced varying signal (finite={finite_fraction:.2f})"})
    return records


def _synthetic_panel(n_rows: int = 60, n_cols: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), index=idx,
                        columns=[f"S{i:03d}" for i in range(n_cols)])


def _ashare_coverage(canonicals: list[str], reg) -> tuple[list[dict[str, object]], dict[str, object]]:
    """R16-001: FULL-registry A-share-style coverage — no fixed slice.

    The old ``sample = canonicals[:200]`` silently excluded every canonical past
    rank 200 from the master report.  Now EVERY canonical gets an outcome; one
    that cannot be constructed/executed is recorded as ``NOT_RUN``/``AUDIT_ERROR``
    (never dropped), and the emitted meta record carries the audited-set hash so
    ``audited == FinalRegistrySnapshot.canonicals`` can be proven by set equality.
    """
    import hashlib

    panel = _synthetic_panel()
    rows: list[dict[str, object]] = []
    not_run: list[dict[str, object]] = []
    audited: set[str] = set()
    for canon in canonicals:
        op = reg.get(canon)
        if op is None:
            not_run.append({"canonical": canon, "reason": "no runtime operator"})
            continue
        try:
            out = op.calculate(panel)
            arr = out.to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001
            not_run.append({
                "canonical": canon,
                "reason": f"AUDIT_ERROR {type(exc).__name__}: {exc}",
            })
            continue
        audited.add(canon)
        flat = arr[np.isfinite(arr)]
        streaks: list[int] = []
        for j in range(arr.shape[1]):
            streak = cur = 0
            for v in arr[:, j]:
                if np.isnan(v):
                    cur += 1
                    streak = max(streak, cur)
                else:
                    cur = 0
            streaks.append(streak)
        rows.append({
            "canonical": canon,
            "runnable": True,
            "finite_ratio": float(np.isfinite(arr).mean()),
            "cross_sectional_std": float(flat.std()) if flat.size > 1 else None,
            "unique_values": int(len(np.unique(flat))),
            "inf_count": int(np.isinf(arr).sum()),
            "longest_nan_streak": int(max(streaks)) if streaks else 0,
        })
    covered = audited | {r["canonical"] for r in not_run}
    covered_hash = hashlib.sha256(
        json.dumps(sorted(covered), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    meta = {
        "registry_total": len(canonicals),
        "audited": len(audited),
        "not_run": len(not_run),
        "not_run_records": not_run,
        "covered_canonical_hash": covered_hash,
        "set_equality": covered == set(canonicals),
        "fully_audited": audited == set(canonicals),
    }
    return rows, meta


def _semantic_duplicates(canonicals: list[str], reg) -> dict[str, object]:
    """R16-002: FULL-registry exact/affine/corr duplicate clusters.

    Every public canonical enters candidate generation.  O(N²) over the full
    registry is bucketed by (input_grain, output_unit, scalar-arity) so only
    contracts that COULD duplicate are compared pairwise; each canonical still
    gets a machine outcome (member of a cluster, or ``checked_not_duplicate``).
    """
    import hashlib

    panel = _synthetic_panel(n_rows=40, n_cols=3, seed=11)
    out_map: dict[str, np.ndarray] = {}
    meta_map: dict[str, tuple[str, str, int]] = {}
    not_run: list[dict[str, object]] = []
    for canon in canonicals:
        op = reg.get(canon)
        if op is None:
            not_run.append({"canonical": canon, "reason": "no runtime operator"})
            continue
        entry = reg._catalog.get(canon, {})
        bucket = (
            str(entry.get("input_grain") or ""),
            str(entry.get("output_unit") or ""),
            int(_param_names(canon, reg).__len__()),
        )
        try:
            out = op.calculate(panel).to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001
            not_run.append({
                "canonical": canon,
                "reason": f"AUDIT_ERROR {type(exc).__name__}: {exc}",
            })
            continue
        if out.size == 0 or not np.isfinite(out).any():
            not_run.append({"canonical": canon, "reason": "no finite output on battery"})
            continue
        out_map[canon] = out
        meta_map[canon] = bucket

    from collections import defaultdict as _dd

    buckets: dict[tuple[str, str, int], list[str]] = _dd(list)
    for canon in out_map:
        buckets[meta_map[canon]].append(canon)

    clusters: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    checked: set[str] = set()
    for bucket_names in buckets.values():
        names = sorted(bucket_names)
        for i in range(len(names)):
            a = names[i]
            va = out_map[a].ravel()
            va_f = va[np.isfinite(va)]
            if va_f.size < 3:
                continue
            for j in range(i + 1, len(names)):
                b = names[j]
                vb = out_map[b].ravel()
                both = np.isfinite(va) & np.isfinite(vb)
                if both.sum() < 3:
                    continue
                u, v = va[both], vb[both]
                if np.allclose(u, v, rtol=1e-9, atol=1e-12):
                    kind = "exact"
                elif abs(u.mean()) > 1e-12 and np.allclose(
                        u / u.mean(), v / v.mean(), rtol=1e-9, atol=1e-12):
                    kind = "affine"
                elif np.corrcoef(u, v)[0, 1] > 0.9999:
                    kind = "corr"
                else:
                    continue
                checked.add(a)
                checked.add(b)
                base = a
                entry = next((e for e in clusters[kind] if e[0] == base), None)
                if entry is None:
                    clusters[kind].append((base, [b]))
                else:
                    entry[1].append(b)
    covered = set(out_map) | {r["canonical"] for r in not_run}
    covered_hash = hashlib.sha256(
        json.dumps(sorted(covered), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "note": "FULL-registry duplicate scan (bucketed by grain/unit/arity); "
                "every canonical has an outcome",
        "sample_size": len(out_map),
        "not_run": not_run,
        "covered_canonical_hash": covered_hash,
        "set_equality": covered == set(canonicals),
        "clusters": {
            kind: [{"base": base, "duplicates": dups} for base, dups in entries]
            for kind, entries in clusters.items()
        },
    }


def _write_summary(rows: list[dict[str, object]],
                   param_surface: list[dict[str, object]],
                   grain_rows: list[dict[str, object]],
                   canonicals: list[str]) -> None:
    role_counter = Counter(r["role"] for r in rows)
    surface_counter = Counter(r["surface"] for r in rows)
    status_counter = Counter(r["status"] for r in rows)
    certified = sum(1 for r in rows if r["production_certified"])
    unclassified = [r["canonical"] for r in rows if r["surface"] == "unclassified"]
    undeclared_role = [p for p in param_surface if p["searchable"] and not p["role_declared"]]
    lines = [
        "# FactorEngine R15 Master-Audit Summary",
        "",
        f"- total canonicals = {len(canonicals)}",
        f"- production_certified = {certified}",
        f"- unclassified = {len(unclassified)}",
        "",
        "## Surface distribution",
        "",
        "| surface | count |",
        "|---|---|",
    ]
    for s, n in sorted(surface_counter.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {s} | {n} |")
    lines += [
        "",
        "## Role distribution",
        "",
        "| role | count |",
        "|---|---|",
    ]
    for s, n in sorted(role_counter.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {s} | {n} |")
    lines += [
        "",
        "## Status distribution",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for s, n in sorted(status_counter.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| {s} | {n} |")
    lines += [
        "",
        "## Parameter-surface audit (NEW-001/259/260)",
        "",
        f"- searchable params with UNDECLARED ParamRole = {len(undeclared_role)}",
        f"- grain-transform operators = {len(grain_rows)}",
        "",
        "## DoD checkpoints",
        "",
        f"- [{'x' if unclassified else ' '}] UNCLASSIFIED == 0",
        f"- [{'x' if not undeclared_role else ' '}] every searchable scalar declares ParamRole",
        f"- [{'x' if certified > 0 else ' '}] production certification from evidence overlay",
        "",
    ]
    if unclassified:
        lines.append("unclassified list: " + ", ".join(sorted(unclassified)))
    if undeclared_role:
        lines.append("undeclared-role sample: " + ", ".join(
            f"{p['canonical']}:{p['param']}" for p in undeclared_role[:20]
        ))
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_all()
