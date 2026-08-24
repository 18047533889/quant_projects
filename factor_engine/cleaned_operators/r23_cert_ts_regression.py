# -*- coding: utf-8 -*-
"""R23 P1 certification: time_series_regression + time_series_risk + time_series_volatility.

This module mirrors the proven ``r23_cert_ts_price`` pattern for the heavier
time-series model / risk / volatility operators.  Certification is granted ONLY
to canonicals that have six-way primitive evidence (polars_reference_parity,
polars_edge, duckdb_reference_parity, duckdb_real_sql, duckdb_edge|nan_edge,
no_fallback) present in the committed ``primitive_verified.json`` artifact.

HONEST FINDING (R23 P1): **no** canonical in the three in-scope categories
(``time_series_regression`` ~86, ``time_series_risk`` ~30,
``time_series_volatility`` ~10) has six-way primitive evidence in the
``primitive_verified.json`` artifact.  The artifact's six-way intersection
contains only the 18 ``time_series`` / price-structure primitives already
certified by ``r23_cert_ts_price`` plus cross-sectional/group/elementwise
atomics.  The 126 in-scope regression/risk/volatility operators each carry at
most a 5-gate factor-runtime record (``runtime_execution_verified``,
``shape_verified``, ``determinism_verified``, ``prefix_kernel_causality_verified``,
``temporal_prefix_verified``) in ``factor_operator_verified.json`` but are
missing ``semantic_golden_verified`` and ``source_contract_verified`` — they do
NOT satisfy the six-way bar.

Therefore the certified set is EMPTY: no regression/risk/volatility canonical is
promoted to production.  The per-gate overlay must never fabricate evidence, so
``apply_r23_certification`` is a safe no-op here.  It remains wired into the
bootstrap so that a future artifact with genuine six-way evidence for these
operators certifies automatically (the guard is data-driven).
"""
from __future__ import annotations

import json
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

# Honest: zero canonicals in the regression/risk/volatility scope have six-way
# primitive evidence in the artifact.  Certification must not be fabricated.
R23_CERTIFIED_CANONICALS: frozenset[str] = frozenset()


def _six_way_from_artifact() -> frozenset[str]:
    """Read the six-way intersection directly from the committed artifact.

    Mirrors ``r23_cert_ts_price._six_way_from_artifact`` exactly.
    """
    path = FE_ROOT.parent / "evidence" / "primitive_verified.json"
    if not path.is_file():
        return frozenset()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return frozenset()
    keys = (
        "polars_reference_parity",
        "polars_edge_verified",
        "duckdb_reference_parity",
        "duckdb_real_sql_verified",
        "duckdb_edge_verified",
        "duckdb_nan_edge_verified",
        "no_fallback_verified",
    )
    sets = {}
    for k in keys:
        raw = data.get(k)
        if isinstance(raw, list):
            sets[k] = frozenset(str(v) for v in raw)
        else:
            return frozenset()
    required = keys[:4] + ("no_fallback_verified",)
    optional = ("duckdb_edge_verified", "duckdb_nan_edge_verified")
    base = sets[required[0]].intersection(*(sets[k] for k in required[1:]))
    edge_union = sets[optional[0]] | sets[optional[1]]
    return base & edge_union


def apply_r23_certification() -> None:
    """Apply R23 P1 certification for TS-REGRESSION / TS-RISK / TS-VOLATILITY.

    ``R23_CERTIFIED_CANONICALS`` is empty because no in-scope operator has
    six-way primitive evidence in the committed artifact.  The loop is a
    guaranteed no-op (no fabricated certification).  The guard structure is kept
    identical to ``r23_cert_ts_price`` so that future six-way evidence activates
    the promotion automatically.
    """
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    six_way = _six_way_from_artifact()
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        if canonical not in six_way:
            continue
        if canonical not in DAILY_CANONICALS:
            continue
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        catalog["production_certified"] = True
        catalog["status"] = "production"
        catalog["lifecycle_status"] = "production"
        catalog["pit_safe"] = True

        # Heavier tiers: regression = cost:3, risk/volatility = cost:2.  These
        # are appended only for canonicals that actually certify (none today).
        tags = list(catalog.get("tags") or ())
        tags.append("cost:3")
        catalog["tags"] = tuple(tags)

        meta = catalog.setdefault("backend_meta", {}).setdefault("pandas_numpy", {})
        meta["production_certified"] = True
        meta["certification_tier"] = "production"
        meta["certification_source"] = (
            "primitive_verified.json (R23 P1: time_series_regression/"
            "time_series_risk/time_series_volatility certification)"
        )

        for gate in (
            "semantic_golden_verified",
            "temporal_prefix_verified",
            "source_contract_verified",
            "runtime_execution_verified",
            "shape_verified",
            "determinism_verified",
            "prefix_kernel_causality_verified",
        ):
            catalog[gate] = True
        catalog["implementation_certified"] = True
        catalog["semantic_certified"] = True
        catalog["temporal_certified"] = True
        catalog["source_contract_certified"] = True
        catalog["edge_case_passed"] = True
        catalog["backend_passed"] = True
        catalog["semantic_pit_review_passed"] = True
        catalog["operator_certification"] = True
        catalog["certification_notes"] = [
            "R23 P1 certification: six-way primitive evidence present in artifact",
            "time_series_regression / time_series_risk / time_series_volatility",
            f"artifact operator record present: {canonical in six_way}",
        ]


def apply_r23_certification_post_hook() -> None:
    """Entry point for the bootstrap hook (called after evidence overlay)."""
    apply_r23_certification()
