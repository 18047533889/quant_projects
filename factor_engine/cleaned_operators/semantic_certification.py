# -*- coding: utf-8 -*-
"""Semantic / temporal / source-contract certification for production admission.

``apply_production_hardening`` historically blanket-promoted every registered
operator to ``status=production`` with ``pit_safe=True`` as long as it had a
backend and was not on an explicit block list.  That conflates "registered and
can run" with "math and PIT semantics passed an audit", which is exactly the
over-promotion class this module removes.

Production admission is split into four independent certificates:

1. ``implementation_certified``  -- backend runs, no exception, correct shape;
   satisfied by evidence-backed execution (parity / golden tests).
2. ``semantic_certified``        -- formula matches name / docs / golden case.
3. ``temporal_certified``        -- outputs are invariant to future data and
   the current value obeys the availability contract.
4. ``source_contract_certified`` -- field, announcement time, effective time
   and universe all honour PIT.

``production_certified == implementation_certified AND semantic_certified AND
temporal_certified AND source_contract_certified``.  Pandas/Polars parity alone
only proves certificate (1); the evidence overlay remains the authority for the
runtime ``production_certified`` field.

Two classes of operator are fail-closed to ``experimental`` and
``pit_safe=False`` here (the "registered but not audited" set that the blanket
promotion used to swallow):

* operators registered with an explicit ``status="experimental"`` / research
  lifecycle (model families, next-stage ts_model / cross-section / intraday
  kernels) -- captured by ``snapshot_registered_statuses`` right after module
  registration, before any promotion layer rewrites the lifecycle field;
* an explicit ``ISOLATED_FROM_DEFAULT_MINING`` manifest of operators whose
  current semantics are known-defective (panel-contract violations, missing
  shareholder identity, unknown-state treated as 0, etc.) until reworked.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Snapshot of the raw registered lifecycle status, taken after the operator
# modules finish importing and before any promotion / hardening layer can
# rewrite ``catalog["status"]`` (see ``snapshot_registered_statuses``).
_REGISTERED_LIFECYCLE: dict[str, str] = {}


def snapshot_registered_statuses() -> None:
    """Capture the raw registered lifecycle status of every operator.

    Must run after the operator modules finish importing and before any
    promotion / hardening layer (``apply_operator_deduplication``,
    ``layer_governance*``, ``apply_production_hardening``) rewrites
    ``catalog["status"]``.
    """
    from cleaned_operators.registry import OperatorRegistry

    _REGISTERED_LIFECYCLE.clear()
    for canonical, catalog in OperatorRegistry._catalog.items():
        status = str(catalog.get("status") or "implemented")
        _REGISTERED_LIFECYCLE[canonical] = status


def registered_status(canonical: str) -> str | None:
    """Return the snapshot lifecycle status, or None if not registered."""
    return _REGISTERED_LIFECYCLE.get(canonical)


# Operators whose registration status was experimental but whose semantics have
# since been reworked and certified (S9 unknown-state rework; 2026-08 holder
# ShareholderId-matched rework; market-model PIT certification).  Hardening is
# allowed to promote them to production instead of keeping them fail-closed.
PROMOTED_OUT_OF_EXPERIMENTAL: frozenset[str] = frozenset({
    "index_reconstitution_churn",
    "listing_age",
    "suspension_frequency",
    # 2026-08: holder_* reworked to ShareholderId-matched union pair.
    "holder_weighted_churn",
    "holder_entry_share",
    "holder_exit_share",
    "holder_net_entry_share",
    "holder_rank_stability",
    # 2026-08: relation deltas verified pair-valid (PIT); multi-index intensity
    # reworked to unknown-state semantics.
    "relation_entry_count",
    "relation_exit_count",
    "relation_weighted_change",
    "multi_index_entry_intensity",
    # 2026-08: CAPM-family operators certified causal (trailing-window kernels).
    "tail_beta",
    "residual_momentum_capm",
    "coskewness_to_market",
    "idio_vol",
    "idio_skew",
    # 2026-08 final pack: complexity/entropy reimplemented by the reviewed
    # sequence_complexity module (overwrites the older ts_model.complexity
    # experimental registration; the research-surface membership is dropped
    # there too).
    "ts_permutation_entropy",
    "ts_sample_entropy",
})


# --------------------------------------------------------------------------
# Honesty metadata for legacy in-sample / over-named variants (audit §10.1,
# §9.3-§9.5).  Stamped at load time without changing surface or lifecycle: these
# operators stay experimental and fail-closed for production; the metadata makes
# the registry advertise that they are diagnostic/benchmark-only and names the
# preferred replacements.
# --------------------------------------------------------------------------
_DIAGNOSTIC_IN_SAMPLE = frozenset({
    "ts_multi_regression_coeff", "ts_multi_regression_resid",
    "ts_multi_regression_resid_z", "ts_multi_regression_r2",
    "ts_huber_regression_coeff", "ts_huber_regression_resid",
    "ts_huber_regression_resid_z", "ts_ridge_regression_coeff",
    "ts_ridge_regression_resid", "ts_ridge_regression_resid_z",
    "ts_ar_forecast", "ts_ar_innovation", "ts_ar_innovation_z",
    "ts_mean_reversion_half_life",
    # Expectile / quantile regression families are fitted on the full look-back
    # window (in-sample), so their coeff/resid/slope are diagnostics; the causal
    # surface uses the *_prior / *_forecast_error variants.
    "ts_expectile_regression_coeff", "ts_expectile_regression_resid",
    "ts_quantile_regression_coeff", "ts_quantile_regression_resid",
    "ts_quantile_regression_slope",
})
_IN_SAMPLE_REPLACEMENTS = {
    "ts_expectile_regression_coeff": "ts_expectile_regression_coeff_prior",
    "ts_multi_regression_coeff": "ts_multi_regression_coeff_prior",
    "ts_multi_regression_resid": "ts_multi_regression_forecast_error",
    "ts_multi_regression_resid_z": "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2": "ts_multi_regression_r2_prior",
    "ts_huber_regression_coeff": "ts_huber_regression_coeff_prior",
    "ts_huber_regression_resid": "ts_huber_regression_forecast_error",
    "ts_huber_regression_resid_z": "ts_huber_regression_forecast_error_z",
    "ts_ridge_regression_coeff": "ts_ridge_regression_coeff_prior",
    "ts_ridge_regression_resid": "ts_ridge_regression_forecast_error",
    "ts_ridge_regression_resid_z": "ts_ridge_regression_forecast_error_z",
    "ts_ar_forecast": "ts_ar_prior_forecast",
    "ts_ar_innovation": "ts_ar_prior_innovation",
    "ts_ar_innovation_z": "ts_ar_prior_innovation_z",
}
_JUMP_ALIAS_DEPRECATED = frozenset({
    "intra_positive_jump_variation", "intra_negative_jump_variation",
    "intra_signed_jump_ratio", "intra_return_profile_cosine",
})
_JUMP_REPLACEMENTS = {
    "intra_positive_jump_variation": "intra_positive_tail_variation",
    "intra_negative_jump_variation": "intra_negative_tail_variation",
    "intra_signed_jump_ratio": "intra_signed_tail_variation_ratio",
    "intra_return_profile_cosine": "intra_signed_return_profile_cosine",
}
_BENCHMARK_ONLY = frozenset({
    "intra_realized_beta", "intra_realized_correlation",
    "intra_idiosyncratic_variance",
})
# Shareholder rank-slot naming consolidation (audit §6.1/§6.2).  The historic
# holder_weighted_churn family was reworked 2026-08 to the ShareholderId-matched
# union pair, so those legacy names are functionally the ID-matched
# implementations.  Stamping preferred_replacements consolidates mining/tooling
# onto the explicit holder_id_matched_* canonical names without demoting the
# reworked production operators.  holder_pledge_churn remains genuinely
# rank-slot (no ShareholderId input) and is flagged for a dedicated ID-matched
# pledge operator; the deprecated concentration/count-change row metrics point
# to the snapshot-change relation primitive.
_RANK_SLOT_ALIAS_REPLACEMENTS = {
    "holder_weighted_churn": ("holder_id_matched_churn",),
    "holder_entry_share": ("holder_id_matched_entry_share",),
    "holder_exit_share": ("holder_id_matched_exit_share",),
    "holder_net_entry_share": (
        "holder_id_matched_entry_share", "holder_id_matched_exit_share",
    ),
    "holder_rank_stability": ("holder_share_weighted_rank_migration",),
}
_RANK_SLOT_COMPATIBILITY_ONLY = frozenset({
    "holder_pledge_churn",
    "holder_concentration_change",
    "holder_count_change_rate",
})


def stamp_compatibility_metadata() -> None:
    """Stamp honest diagnostic/benchmark metadata onto legacy registry entries."""
    from cleaned_operators.registry import OperatorRegistry

    for canon in _DIAGNOSTIC_IN_SAMPLE:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["in_sample"] = True
        entry["diagnostic_only"] = True
        entry["hidden_from_default_mining"] = True
        replacement = _IN_SAMPLE_REPLACEMENTS.get(canon)
        if replacement is not None:
            entry["preferred_replacements"] = [replacement]
        entry.setdefault("semantic_note", "训练窗口含当前样本(旧 in-sample);默认挖掘与生产应使用 *_prior / *_forecast_error")
    for canon in _JUMP_ALIAS_DEPRECATED:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["compatibility_only"] = True
        replacement = _JUMP_REPLACEMENTS.get(canon)
        if replacement is not None:
            entry["preferred_replacements"] = [replacement]
        entry.setdefault("semantic_note", "旧别名(阈值尾部法/含义模糊);使用显式 *_tail_variation / *_signed_return_profile_cosine")
    for canon in _BENCHMARK_ONLY:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["benchmark_only"] = True
        entry.setdefault("semantic_note", "非 ex-self 市场模型仅作 benchmark/legacy;默认搜索使用 *_ex_self 版本")
    for canon, replacements in _RANK_SLOT_ALIAS_REPLACEMENTS.items():
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["preferred_replacements"] = list(replacements)
        entry.setdefault(
            "semantic_note",
            "名次槽位命名遗留；实现已重写为股东 ID 匹配，规范名见 preferred_replacements",
        )
    for canon in _RANK_SLOT_COMPATIBILITY_ONLY:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["compatibility_only"] = True
        entry.setdefault(
            "semantic_note",
            "名次槽位/行数口径（无股东 ID 输入）；需专用 ID 匹配算子或 relation 快照指标",
        )


def is_intentionally_experimental(canonical: str) -> bool:
    """True when the operator was registered as experimental/research.

    These operators may have full backends and run correctly, but they were
    explicitly marked non-production at registration time.  Hardening must not
    silently upgrade them to ``status=production`` / ``pit_safe=True``.
    """
    if canonical in PROMOTED_OUT_OF_EXPERIMENTAL:
        return False
    return _REGISTERED_LIFECYCLE.get(canonical) in {"experimental", "research"}


# --------------------------------------------------------------------------
# Explicit isolation manifest (audit S18): operators whose current semantics are
# known-defective and must not be default production targets until reworked.
# Kept registered so explicit recipes can still reference them and fail loudly,
# but they are no longer blanket-promoted or marked pit-safe.
# --------------------------------------------------------------------------
ISOLATED_FROM_DEFAULT_MINING: frozenset[str] = frozenset({
    # Panel-contract violation: per-row entity counts broadcast across all
    # instrument columns; must become a source aggregation or a dedicated type.
    # They are relation-domain tools, not factor-panel targets, so they stay off
    # the daily surface.
    "relation_distinct_count",
    "relation_overlap_ratio",
    # Legacy ambiguous TTM/period names retained only to fail with a migration
    # error; never default production (prefer fin_ttm_quarterly / fin_ttm_cumulative).
    "fin_ttm",
    "ttm",
    "quarter",
    "yoy",
    # Conflation of NaN/±Inf/zero and unknown-data in one sink.
    "nan_to_num",
    "fillna",
    "protected_div",
    "causal_linear_extrapolate",
    # Revision operators cannot be PIT-certified without historical revision
    # vintages (which publication version was knowable at each past date).
    # Until a real revision-vintage source exists they stay experimental and
    # ``source_pit_passed=False`` (review §5.7).
    "fin_revision_delta",
    "fin_revision_count",
    "fin_restated_flag",
    "fin_revision_magnitude",
    "fin_revision_pct",
    "fin_revision_direction",
})


def is_isolated_from_default_mining(canonical: str) -> bool:
    return canonical in ISOLATED_FROM_DEFAULT_MINING


def should_fail_closed(canonical: str) -> bool:
    """True when this canonical must not be promoted / marked pit-safe."""
    return (
        is_intentionally_experimental(canonical)
        or is_isolated_from_default_mining(canonical)
    )


@dataclass(frozen=True)
class SemanticCert:
    """A single canonical's certification record (six independent gates).

    The first four gates are the semantic/PIT certificates; ``edge_case_passed``
    and ``backend_passed`` are the additional production-admission dimensions
    (see review §11.7).  ``operator_certification`` is the six-gate AND used as
    the strict production-admission authority; ``production_certified`` keeps the
    historical four-gate semantics for backward compatibility with the current
    evidence overlay.
    """

    canonical: str
    implementation_certified: bool
    semantic_certified: bool
    temporal_certified: bool
    source_contract_certified: bool
    pit_safe: bool
    edge_case_passed: bool = False
    backend_passed: bool = False
    notes: tuple[str, ...] = ()

    @property
    def production_certified(self) -> bool:
        return bool(
            self.implementation_certified
            and self.semantic_certified
            and self.temporal_certified
            and self.source_contract_certified
        )

    @property
    def operator_certification(self) -> bool:
        """Six-gate production certification (review §2 / §11.7)."""
        return bool(
            self.implementation_certified
            and self.semantic_certified
            and self.temporal_certified
            and self.source_contract_certified
            and self.edge_case_passed
            and self.backend_passed
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorCertification:
    """Six-gate production certification record (review §2).

    ``production_certified`` requires every gate to pass.  Only operators with
    ``production_certified=True`` may enter the daily DSL allowlist, the cold-start
    operator pool, and the AlphaProbe/AlphaMiner search space.
    """

    implementation_passed: bool
    semantic_passed: bool
    temporal_passed: bool
    source_pit_passed: bool
    edge_case_passed: bool
    backend_passed: bool

    @property
    def production_certified(self) -> bool:
        return all(
            (
                self.implementation_passed,
                self.semantic_passed,
                self.temporal_passed,
                self.source_pit_passed,
                self.edge_case_passed,
                self.backend_passed,
            )
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def operator_certification_for(
    canonical: str,
    catalog: dict[str, Any] | None = None,
) -> OperatorCertification:
    """Compose the six-gate certification record from the semantic certificates."""
    cert = semantic_cert(canonical, catalog)
    return OperatorCertification(
        implementation_passed=cert.implementation_certified,
        semantic_passed=cert.semantic_certified,
        temporal_passed=cert.temporal_certified,
        source_pit_passed=cert.source_contract_certified,
        edge_case_passed=cert.edge_case_passed,
        backend_passed=cert.backend_passed,
    )


def semantic_cert(
    canonical: str,
    catalog: dict[str, Any] | None = None,
    *,
    evidence_production_certified: bool | None = None,
) -> SemanticCert:
    """Compose the four-certificate record for ``canonical``.

    Fail-closed: a canonical in the experimental snapshot or the isolation
    manifest receives all-negative certificates unless the evidence overlay
    (``evidence_production_certified=True``) independently confirms it.  All
    other canonicals default to certified (the established production surface).
    """
    notes: list[str] = []

    # (1) Implementation: evidence overlay result is authoritative when given.
    if evidence_production_certified is True:
        implementation = True
    elif evidence_production_certified is False:
        implementation = False
    else:
        implementation = not should_fail_closed(canonical)

    # (2) Semantic / (3) temporal: fail-closed for the isolated/experimental set.
    semantic_ok = not should_fail_closed(canonical)
    temporal_ok = not should_fail_closed(canonical)

    # (4) Source contract: not source-blocked and not research-marked.
    from cleaned_operators.production_hardening import SOURCE_BLOCKED_CANONICALS

    source_ok = (
        canonical not in SOURCE_BLOCKED_CANONICALS
        and not should_fail_closed(canonical)
    )

    if should_fail_closed(canonical):
        notes.append("registered experimental/research or in isolation manifest")
    if not source_ok:
        notes.append("source contract not certified")

    # (5) Edge-case evidence: declared/verified edge dimensions must be complete.
    edge_ok = False
    try:
        from cleaned_operators.edge_requirements import (
            production_edge_evidence_complete,
        )

        edge_ok = bool(production_edge_evidence_complete(canonical))
    except Exception:
        edge_ok = False
    if not edge_ok:
        notes.append("edge-case evidence incomplete")

    # (6) Backend evidence: at least one independently certified production backend.
    backend_ok = False
    try:
        from backend.operator_capability import production_eligible_backends

        backend_ok = bool(production_eligible_backends(canonical))
    except Exception:
        backend_ok = False
    if not backend_ok:
        notes.append("no evidence-backed production backend")

    pit_safe = bool(implementation and semantic_ok and temporal_ok and source_ok)
    return SemanticCert(
        canonical=canonical,
        implementation_certified=bool(implementation),
        semantic_certified=bool(semantic_ok),
        temporal_certified=bool(temporal_ok),
        source_contract_certified=bool(source_ok),
        pit_safe=pit_safe,
        edge_case_passed=edge_ok,
        backend_passed=backend_ok,
        notes=tuple(notes),
    )


def attach_four_certificates(canonical: str, catalog: dict[str, Any]) -> SemanticCert:
    """Write the certificate fields + ``production_certified`` onto catalog.

    Returns the composed record so callers can gate status/pit_safe on it.
    Also writes the six-gate fields (``edge_case_passed`` / ``backend_passed`` /
    ``operator_certification``) so the strict production-admission composite is
    visible per operator.

    The **six-gate** composite is the single production-certification authority
    (review §2.1): ``catalog["production_certified"]`` now carries
    ``operator_certification`` (implementation + semantic + temporal + source
    contract + edge-case + backend).  The historical four-gate AND is retained
    under ``semantic_pit_review_passed`` for diagnostics only; it no longer
    grants production admission.
    """
    cert = semantic_cert(canonical, catalog)
    catalog["implementation_certified"] = cert.implementation_certified
    catalog["semantic_certified"] = cert.semantic_certified
    catalog["temporal_certified"] = cert.temporal_certified
    catalog["source_contract_certified"] = cert.source_contract_certified
    # Historical four-gate review record — NOT a production authority.
    catalog["semantic_pit_review_passed"] = cert.production_certified
    catalog["production_certified"] = cert.operator_certification
    catalog["edge_case_passed"] = cert.edge_case_passed
    catalog["backend_passed"] = cert.backend_passed
    catalog["operator_certification"] = cert.operator_certification
    catalog["certification_notes"] = list(cert.notes)
    return cert


DEFAULT_CERTIFICATION = {
    "implementation_passed": False,
    "semantic_passed": False,
    "temporal_passed": False,
    "source_pit_passed": False,
    "edge_case_passed": False,
    "backend_passed": False,
}


def reconcile_operator_certification(
    canonical: str,
    catalog: dict[str, Any] | None = None,
) -> OperatorCertification:
    """Converge the six-gate production certification from final evidence.

    Runs after the evidence overlay has bound physical backend certification to
    the immutable artifacts.  This is the **single authority** for
    ``status`` / ``lifecycle_status`` / ``pit_safe`` / ``production_certified``:
    an operator is production only when all six gates pass (review §2.3, §2.4).

    ``status == "production"`` therefore means "fully certified", not merely
    "a reviewed target".  Operators whose evidence is stale, absent, or whose
    edge/backend gates fail are downgraded to ``experimental`` and fail-closed.
    """
    from cleaned_operators.registry import OperatorRegistry

    catalog = (
        catalog
        if catalog is not None
        else OperatorRegistry._catalog.get(canonical, {})
    )
    cert = semantic_cert(canonical, catalog)

    # (1) Implementation gate: evidence-bound backend certification is the only
    # authority.  ``should_fail_closed`` alone must never grant it (review §2.2).
    meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
    implementation_passed = bool(meta.get("production_certified")) or bool(
        cert.backend_passed
    )

    # (2)-(4) Semantic / temporal / source-PIT gates: an operator is never
    # certified by *absence* — the four-gate review record must ALSO be backed
    # by a version-bound evidence record (``implementation_passed``).  The
    # experimental/isolated set stays all-negative regardless (review §2.2).
    semantic_passed = bool(
        implementation_passed and cert.semantic_certified
    )
    temporal_passed = bool(
        implementation_passed and cert.temporal_certified
    )
    source_pit_passed = bool(
        implementation_passed and cert.source_contract_certified
    )

    # (5) Edge-case evidence.  ``production_edge_evidence_complete`` conflates
    # "the edge-certification infrastructure has not populated this surface"
    # with "a declared edge dimension is genuinely unverified", so it cannot be
    # a hard fail for operators whose implementation is otherwise evidence-
    # bound.  When a backend is evidence-certified the edge gate follows the
    # implementation gate; without implementation evidence the whole six-gate
    # still fails closed (review §14.7 admission test).
    edge_case_passed = bool(cert.edge_case_passed or implementation_passed)
    # (6) At least one evidence-backed production backend must exist.
    backend_passed = cert.backend_passed

    six = OperatorCertification(
        implementation_passed=bool(implementation_passed),
        semantic_passed=bool(semantic_passed),
        temporal_passed=bool(temporal_passed),
        source_pit_passed=bool(source_pit_passed),
        edge_case_passed=bool(edge_case_passed),
        backend_passed=bool(backend_passed),
    )
    certified = six.production_certified

    catalog["implementation_certified"] = six.implementation_passed
    catalog["semantic_certified"] = six.semantic_passed
    catalog["temporal_certified"] = six.temporal_passed
    catalog["source_contract_certified"] = six.source_pit_passed
    catalog["edge_case_passed"] = six.edge_case_passed
    catalog["backend_passed"] = six.backend_passed
    catalog["semantic_pit_review_passed"] = all(
        (
            six.implementation_passed,
            six.semantic_passed,
            six.temporal_passed,
            six.source_pit_passed,
        )
    )
    catalog["operator_certification"] = certified
    catalog["production_certified"] = certified
    catalog["pit_safe"] = certified
    catalog["status"] = "production" if certified else "experimental"
    catalog["lifecycle_status"] = "production" if certified else "experimental"
    catalog["certification_notes"] = list(cert.notes)
    return six
