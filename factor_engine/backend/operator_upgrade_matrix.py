# -*- coding: utf-8
"""算子 production 升级矩阵：按阻断原因分层（Batch A–E）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BatchId = Literal["A", "B", "C", "D", "E", "F"]
ProductionStatus = Literal[
    "certified_dual",
    "backend_execution_certified",
    "operational_production_certified",
    "structural_candidate",
    "semantic_pending",
    "python_rolling",
    "missing_primitive",
    "stateful",
    "pit_session",
    "external_kernel",
    "forbidden",
    "deferred",
]

# Batch A：双端已实现，主要缺 evidence / edge / no-fallback
BATCH_A_CERTIFICATION_ONLY: frozenset[str] = frozenset(
    """
    add subtract multiply neg abs sign exp floor ceil maximum minimum
    gt lt eq ge le ne and_ or_ not_ where coalesce fillna_const
    delay ts_delta ts_pct ts_sum ts_min ts_max ts_var ts_median ts_zscore
    rank_pct cs_pct_rank cs_demean c_mean c_std c_sum c_count
    log_returns volatility vwap group_rank group_std group_neutralize
    """.split()
)

# Batch B：语义/契约须先对齐再认证
BATCH_B_SEMANTIC_FIX: frozenset[str] = frozenset(
    """
    gt lt eq ge le ne and_ or_ not_ where if_else
    maximum minimum scale protected_div protected_log protected_sqrt
    is_null is_nan is_finite is_infinite is_not_null
    ts_sharpe safe_div divide log exp
    """.split()
)

# Batch C：伪 native / Python rolling callback
BATCH_C_PYTHON_ROLLING: frozenset[str] = frozenset(
    """
    ts_argmax ts_argmin ts_quantile ts_skew decay_linear WMA Slope
    ts_product ts_median_abs_deviation ts_mean_abs_deviation
    """.split()
)

# Batch D：缺底层 primitive 或 stable kernel
BATCH_D_MISSING_PRIMITIVE: frozenset[str] = frozenset(
    """
    ts_regression_slope Slope rolling_beta cs_regression cs_resid
    expanding_std ts_product AROON CCI Donchian
    """.split()
)

# Batch E：Stateful / EWM / Wilder
BATCH_E_STATEFUL: frozenset[str] = frozenset(
    """
    ema ema ewm_std ewm_var ewm_corr ewm_cov
    RSI_WILDER ATR_WILDER MACD TRIX ADX KAMA
    """.split()
)

# Batch F：PIT / session / 微观结构
BATCH_F_PIT_SESSION: frozenset[str] = frozenset(
    """
    current_ratio quick_ratio operating_margin debt_to_equity micro_spread
    intraday_vwap_deviation quarter_from_cumulative ttm_from_quarterly
    """.split()
)

EVIDENCE_KEYS = (
    "polars_reference_parity",
    "polars_edge_verified",
    "duckdb_reference_parity",
    "duckdb_real_sql_verified",
    "duckdb_edge_verified",
    "no_fallback_verified",
)


@dataclass(frozen=True)
class OperatorUpgradeRow:
    canonical: str
    execution_kind: str
    polars_implemented: bool
    polars_pure_native: bool
    duckdb_implemented: bool
    duckdb_real_sql: bool
    semantic_contract_complete: bool
    parameter_domain_complete: bool
    type_signature_complete: bool
    normal_parity: bool
    edge_parity: bool
    alignment_verified: bool
    chunk_invariance: bool
    performance_verified: bool
    semantic_certified: bool
    backend_execution_certified: bool
    operational_production_certified: bool
    production_status: ProductionStatus
    upgrade_batch: BatchId | None
    block_reason: str | None
    certification_gaps: tuple[str, ...] = field(default_factory=tuple)
    execution_variants: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["certification_gaps"] = list(self.certification_gaps)
        d["execution_variants"] = list(self.execution_variants)
        d["certified_dual"] = self.backend_execution_certified
        d["semantic_certified"] = self.semantic_certified
        d["backend_execution_certified"] = self.backend_execution_certified
        d["operational_production_certified"] = self.operational_production_certified
        return d


def _evidence_sets() -> dict[str, frozenset[str]]:
    from backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_REAL_SQL_VERIFIED,
        DUCKDB_REFERENCE_PARITY_VERIFIED,
        NO_FALLBACK_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )

    return {
        "polars_reference_parity": POLARS_REFERENCE_PARITY_VERIFIED,
        "polars_edge_verified": POLARS_EDGE_VERIFIED,
        "duckdb_reference_parity": DUCKDB_REFERENCE_PARITY_VERIFIED,
        "duckdb_real_sql_verified": DUCKDB_REAL_SQL_VERIFIED,
        "duckdb_edge_verified": DUCKDB_EDGE_VERIFIED,
        "no_fallback_verified": NO_FALLBACK_VERIFIED,
        "certified_dual": PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    }


def certification_gaps(canon: str) -> tuple[str, ...]:
    """返回尚未满足的 evidence 维度。"""
    sets = _evidence_sets()
    missing: list[str] = []
    for key in EVIDENCE_KEYS:
        if canon not in sets[key]:
            missing.append(key)
    return tuple(missing)


def infer_upgrade_batch(canon: str) -> BatchId | None:
    if canon in BATCH_F_PIT_SESSION:
        return "F"
    if canon in BATCH_E_STATEFUL:
        return "E"
    if canon in BATCH_D_MISSING_PRIMITIVE:
        return "D"
    if canon in BATCH_C_PYTHON_ROLLING:
        return "C"
    if canon in BATCH_B_SEMANTIC_FIX and canon not in BATCH_A_CERTIFICATION_ONLY:
        return "B"
    if canon in BATCH_A_CERTIFICATION_ONLY:
        return "A"
    return None


def infer_block_reason(canon: str, *, batch: BatchId | None, gaps: tuple[str, ...]) -> str | None:
    from backend.primitive_evidence import primitive_dual_backend_production_safe

    if primitive_dual_backend_production_safe(canon):
        return None
    if batch == "F":
        return "pit_or_session_contract_missing"
    if batch == "E":
        return "stateful_scan_required"
    if batch == "D":
        return "missing_stable_primitive"
    if batch == "C":
        return "python_rolling_callback"
    if batch == "B":
        return "semantic_contract_incomplete"
    if batch == "A":
        if gaps:
            return "evidence_incomplete:" + ",".join(gaps)
        return "evidence_incomplete"
    from backend.production_fastpath_tiers import FORBIDDEN_PRODUCTION_FASTPATH

    if canon in FORBIDDEN_PRODUCTION_FASTPATH:
        return "forbidden_fastpath"
    return "not_in_phase1_scope"


def _parameter_domain_complete(canon: str) -> bool:
    from backend.operator_evidence_schema import operator_evidence_record
    from backend.production_signature import operational_production_allowed, signature_for

    if not operational_production_allowed(canon):
        return False
    sig = signature_for(canon)
    if sig is None:
        return True
    rec = operator_evidence_record(canon)
    if rec and rec.get("supported_calls"):
        return True
    return sig.default_status == "production"


def _type_signature_complete(canon: str) -> bool:
    from backend.operator_types import OPERATOR_SIGNATURES

    return canon in OPERATOR_SIGNATURES


def _alignment_verified() -> bool:
    from backend.ordering_spec import duplicate_ts_inst_keys_forbidden

    return duplicate_ts_inst_keys_forbidden()


def build_operator_upgrade_row(canon: str) -> OperatorUpgradeRow:
    from backend.operator_capability import polars_long_tier
    from backend.polars_long_policy import (
        POLARS_LONG_NATIVE,
        POLARS_LONG_PYTHON_ROLLING,
        POLARS_LONG_STATEFUL,
        classify_plan_op,
    )
    from backend.production_fastpath_tiers import (
        FASTPATH_DEFERRED_CANONICALS,
        FORBIDDEN_PRODUCTION_FASTPATH,
        dual_backend_structural_candidates,
    )
    from backend.primitive_evidence import (
        primitive_dual_backend_production_safe,
        primitive_operational_production_certified,
    )
    from backend.sql_tiers import is_sql_implemented
    from cleaned_operators.operator_spec import infer_production_policy

    gaps = certification_gaps(canon)
    batch = infer_upgrade_batch(canon)
    tier = classify_plan_op(canon)
    from cleaned_operators.registry import OperatorRegistry

    polars_meta = dict(
        ((OperatorRegistry.catalog().get(canon, {}).get("backend_meta") or {}).get("polars") or {})
    )
    registered_polars_native = (
        "polars" in OperatorRegistry.backends_for(canon)
        and polars_meta.get("execution_kind") == "expression_native"
        and not bool(polars_meta.get("materializes_full_panel"))
    )
    polars_impl = registered_polars_native or tier in {
        "native",
        "python_rolling",
        "map_groups",
        "stateful",
        "nonstandard_alg",
        "registry",
    } or canon in POLARS_LONG_NATIVE | POLARS_LONG_PYTHON_ROLLING
    polars_pure = registered_polars_native or (tier == "native" and canon in POLARS_LONG_NATIVE)
    duck_impl = is_sql_implemented(canon)
    certified = primitive_dual_backend_production_safe(canon)
    operational = primitive_operational_production_certified(canon)
    param_complete = _parameter_domain_complete(canon)
    type_complete = _type_signature_complete(canon)
    align_ok = _alignment_verified()
    chunk_ok = False
    perf_ok = False
    semantic_ok = batch not in {"B", None} or certified

    if operational and param_complete and type_complete and align_ok and chunk_ok and perf_ok:
        status: ProductionStatus = "operational_production_certified"
    elif certified:
        status = "backend_execution_certified"
    elif batch == "C" or canon in POLARS_LONG_PYTHON_ROLLING:
        status = "python_rolling"
    elif canon in FORBIDDEN_PRODUCTION_FASTPATH:
        status = "forbidden"
    elif canon in FASTPATH_DEFERRED_CANONICALS:
        status = "deferred"
    elif batch == "E" or canon in POLARS_LONG_STATEFUL:
        status = "stateful"
    elif batch == "F":
        status = "pit_session"
    elif batch == "D":
        status = "missing_primitive"
    elif batch == "B":
        status = "semantic_pending"
    elif canon in dual_backend_structural_candidates():
        status = "structural_candidate"
    else:
        status = "structural_candidate"

    if infer_production_policy(canon) == "denied" and status not in {
        "certified_dual",
        "python_rolling",
        "stateful",
        "missing_primitive",
        "pit_session",
        "deferred",
    }:
        status = "forbidden"

    variants: tuple[str, ...] = ()
    try:
        from tests.backend_parity.evidence_case_registry import EXECUTION_VARIANTS

        variants = tuple(EXECUTION_VARIANTS.get(canon, ()))
    except Exception:
        pass

    return OperatorUpgradeRow(
        canonical=canon,
        execution_kind=tier,
        polars_implemented=polars_impl,
        polars_pure_native=polars_pure,
        duckdb_implemented=duck_impl,
        duckdb_real_sql=canon in _evidence_sets()["duckdb_real_sql_verified"],
        semantic_contract_complete=semantic_ok,
        parameter_domain_complete=param_complete,
        type_signature_complete=type_complete,
        normal_parity=canon in _evidence_sets()["polars_reference_parity"],
        edge_parity=canon in _evidence_sets()["polars_edge_verified"],
        alignment_verified=align_ok,
        chunk_invariance=chunk_ok,
        performance_verified=perf_ok,
        semantic_certified=semantic_ok,
        backend_execution_certified=certified,
        operational_production_certified=operational and param_complete and type_complete and align_ok,
        production_status=status,
        upgrade_batch=batch,
        block_reason=infer_block_reason(canon, batch=batch, gaps=gaps),
        certification_gaps=gaps,
        execution_variants=variants,
    )


def build_upgrade_matrix(*, primitives: list[str] | None = None) -> dict[str, Any]:
    from cleaned_operators.operator_spec import (
        PRODUCTION_ALLOWED_DEFERRED_CANONICALS,
        PRODUCTION_DUAL_BACKEND_CORE_CANONICALS,
    )

    if primitives is None:
        from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

        primitives = sorted(
            PRODUCTION_DUAL_BACKEND_CORE_CANONICALS
            | PRODUCTION_ALLOWED_DEFERRED_CANONICALS
            | {"ts_argmax", "ts_argmin"}
            | PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
        )
    rows = {c: build_operator_upgrade_row(c) for c in primitives}
    by_batch: dict[str, list[str]] = {b: [] for b in "ABCDEF"}
    for canon, row in rows.items():
        if row.upgrade_batch:
            by_batch[row.upgrade_batch].append(canon)
    certified = [c for c, r in rows.items() if r.backend_execution_certified]
    operational = [c for c, r in rows.items() if r.operational_production_certified]
    batch_a = [c for c in BATCH_A_CERTIFICATION_ONLY if c in certified]
    return {
        "schema_version": 2,
        "description": "算子 production 升级矩阵：semantic / backend_execution / operational 三层认证",
        "counts": {
            "primitives": len(rows),
            "certified_dual": len(certified),
            "backend_execution_certified": len(certified),
            "operational_production_certified": len(operational),
            "batch_a_total": len(BATCH_A_CERTIFICATION_ONLY),
            "batch_a_certified": len(batch_a),
            "batch_a_pending": len(BATCH_A_CERTIFICATION_ONLY) - len(batch_a),
        },
        "batches": {k: sorted(v) for k, v in by_batch.items()},
        "operators": {c: rows[c].to_dict() for c in sorted(rows)},
    }
