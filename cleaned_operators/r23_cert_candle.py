# -*- coding: utf-8 -*-
"""R23 P1 certification: candle_pattern + candle_geometry + chart_pattern + candle_state_space.

This module mirrors the proven ``r23_cert_ts_price`` pattern for the candle /
chart (K-line) pattern-recognition family.  Certification is granted ONLY to
canonicals that have six-way primitive evidence (polars_reference_parity,
polars_edge_verified, duckdb_reference_parity, duckdb_real_sql_verified,
duckdb_edge_verified|duckdb_nan_edge_verified, no_fallback_verified) present in
the committed ``evidence/primitive_verified.json`` artifact.

HONEST FINDING (R23 P1): **no** canonical in the four in-scope categories
(``candle_pattern`` ~36, ``candle_geometry`` ~13, ``chart_pattern`` ~27,
``candle_state_space`` ~6 = 82 canonicals total) has six-way primitive evidence
in the ``primitive_verified.json`` artifact.  The artifact's six-way intersection
(79 operators) covers only the time_series / price_structure /
price_volume_extension / cross_sectional / group / elementwise / math
primitives already certified by the sibling R23 P1 passes.

Each of the 82 in-scope candle/chart operators carries at most a 5-gate
factor-runtime record (``runtime_execution_verified``, ``shape_verified``,
``determinism_verified``, ``prefix_kernel_causality_verified``,
``temporal_prefix_verified``) in ``factor_operator_verified.json`` but is
MISSING ``semantic_golden_verified`` and ``source_contract_verified`` — they do
NOT satisfy the six-way bar.  No P0 severity and none are in the PIT11/PIT18
denied set, but the certification gate (six-way primitive evidence) is not met.

Therefore the certified set is EMPTY: no candle/chart canonical is promoted to
production.  The per-gate overlay must never fabricate evidence, so
``apply_r23_certification`` is a safe no-op here.  It remains wired into the
bootstrap so that a future artifact with genuine six-way evidence for these
pattern operators certifies automatically (the guard is data-driven).

Cost tiers for the family (used only if a canonical ever actually certifies):
candle_pattern / candle_geometry / chart_pattern = tier 2 (pattern recognition)
=> cost:2; candle_state_space = tier 2 (state-space pattern) => cost:2.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

# Honest: zero canonicals in the candle/chart scope have six-way primitive
# evidence in the artifact.  Certification must not be fabricated.
R23_CERTIFIED_CANONICALS: frozenset[str] = frozenset()

# Declared cost tier for completeness (tier 2 pattern-recognition family); the
# map is populated only for canonicals that actually certify (none today).
_COST_TIER_BY_CANONICAL: dict[str, int] = {}


def _six_way_from_artifact() -> frozenset[str]:
    """Read the six-way intersection directly from the committed artifact.

    Mirrors ``r23_cert_ts_price._six_way_from_artifact`` exactly.
    """
    path = FE_ROOT / "evidence" / "primitive_verified.json"
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


def _is_pit_denied(canonical: str) -> bool:
    """Is the canonical in the R23-P0-PIT11/PIT18 denied set?"""
    try:
        from cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS

        return canonical in PRODUCTION_DENIED_CANONICALS
    except Exception:
        return False


def apply_r23_certification() -> None:
    """Apply R23 P1 certification for the candle/chart family.

    ``R23_CERTIFIED_CANONICALS`` is empty because no in-scope operator has
    six-way primitive evidence in the committed artifact.  The loop is a
    guaranteed no-op (no fabricated certification).  The guard structure is kept
    identical to ``r23_cert_ts_price`` (including the six-way check, the PIT
    denial check, and the cost-contract declaration) so that genuine future
    six-way evidence activates the promotion automatically.
    """
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.registry import OperatorRegistry

    six_way = _six_way_from_artifact()
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        # Honest filter: only operators with genuine six-way evidence get claimed.
        if canonical not in six_way:
            continue
        if _is_pit_denied(canonical):
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

        # R15-INC-011: production_admitted requires an explicit cost contract.
        # Pattern-recognition family is tier 2 => cost:2 (only applied to
        # canonicals that actually certify; none today).
        tier = _COST_TIER_BY_CANONICAL.get(canonical, 2)
        tags = list(catalog.get("tags") or ())
        tags = [str(t) for t in tags]
        if not any(re.fullmatch(r"cost:\d+", t.strip()) for t in tags):
            tags.append(f"cost:{tier}")
        catalog["tags"] = tuple(tags)

        meta = catalog.setdefault("backend_meta", {}).setdefault("pandas_numpy", {})
        meta["production_certified"] = True
        meta["certification_tier"] = "production"
        meta["certification_source"] = (
            "primitive_verified.json (R23 P1: candle/chart certification)"
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
            "candle_pattern / candle_geometry / chart_pattern / candle_state_space",
            f"artifact operator record present: {canonical in six_way}",
        ]


def apply_r23_certification_post_hook() -> None:
    """Entry point for the bootstrap hook (called after evidence overlay)."""
    apply_r23_certification()
