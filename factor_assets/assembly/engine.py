"""Deterministic assembly of factor sets from registered assets."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping, Optional, Union

from factor_assets.contracts.admission import FactorAdmissionArtifact
from factor_assets.contracts.asset import FactorAsset
from factor_assets.contracts.factor_set import FactorMembership, FactorSetArtifact, FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.similarity import SimilarityArtifact
from factor_assets.optimizer.pareto import ParetoPoint
from factor_assets.selection.policy import SelectionDecision


class FactorSetAssembler:
    """Build immutable :class:`FactorSetArtifact` values from factor asset metadata.

    Assembly is deliberately metadata-only.  Candidates are filtered using the
    fields represented by ``FactorSetSpec`` and ordered by an evidence-based
    ranking (selection-decision metadata first, ``factor_id`` as deterministic
    tiebreak) before applying ``max_factors``.  No factor values or metric
    computations are performed here.

    The canonical output is :class:`FactorSetArtifact`; the legacy
    :class:`FactorSet` is available via ``FactorSetArtifact.to_legacy_view()``.
    """

    #: Selection policies with differentiated ranking semantics.  Anything
    #: outside this set (including "manual") falls back to deterministic
    #: factor_id ordering, preserving the legacy behaviour.
    _KNOWN_POLICIES = frozenset({"manual", "pareto_front", "family_robust", "diverse"})

    def assemble(
        self,
        spec: FactorSetSpec,
        candidates: Iterable[FactorAsset],
        *,
        selection_decisions: Optional[Iterable[SelectionDecision]] = None,
        selection_run_id: Optional[str] = None,
        created_at: Optional[str] = None,
        similarity_provider: Optional[
            Union[
                Callable[[str, str], Optional[float]],
                Callable[[str, str], Optional[SimilarityArtifact]],
            ]
        ] = None,
        admission_artifacts: Optional[
            Mapping[str, FactorAdmissionArtifact]
        ] = None,
        production: bool = False,
    ) -> FactorSetArtifact:
        """Assemble a factor set from candidate assets.

        Args:
            spec: Selection criteria and output identity.
            candidates: Registered factor assets to consider.
            selection_decisions: Admission decisions for non-manual policies.
            selection_run_id: Optional provenance identifier for this run.
            created_at: Optional explicit creation timestamp, useful for replay.
            similarity_provider: Optional ``(factor_a, factor_b)`` callable used
                by the ``diverse`` (MMR) policy.  It may return either a bare
                similarity float in ``[0, 1]`` (backward compatible) or a
                :class:`SimilarityArtifact` whose ``primary_view`` is consumed.
                Required for ``diverse``; the policy fails closed without it.
            admission_artifacts: Optional ``{factor_id: FactorAdmissionArtifact}``
                map supplying the production-mandatory membership provenance
                (``health_state_ref``, ``cluster_id``, ``orientation``,
                ``factor_version``).  Membership fields are populated ONLY from
                these artifacts — never synthesized.
            production: When True, membership provenance must be fully resolved
                from admission artifacts; any unresolvable mandatory field fails
                closed (raises ValueError) rather than fabricating a value.

        Raises:
            TypeError: If ``spec``, a candidate, or an admission artifact value
                is not the expected contract.
            ValueError: If no candidates match, ``max_factors`` is invalid, a
                production-mandatory membership field is unresolvable, or an
                admission artifact key does not match its ``factor_id``.
        """
        if not isinstance(spec, FactorSetSpec):
            raise TypeError("spec must be a FactorSetSpec")
        if spec.max_factors is not None and spec.max_factors < 1:
            raise ValueError("max_factors must be >= 1")
        if spec.selection_policy not in self._KNOWN_POLICIES:
            raise ValueError(
                f"unknown selection_policy: {spec.selection_policy!r}; "
                f"expected one of {sorted(self._KNOWN_POLICIES)}"
            )
        if spec.selection_policy == "diverse" and similarity_provider is None:
            raise ValueError(
                "diverse (MMR) selection requires a similarity_provider; "
                "failing closed rather than silently degrading to non-MMR"
            )

        admission_artifacts = self._validate_admission_artifacts(admission_artifacts)

        admitted_factor_ids: Optional[set[str]] = None
        latest_decisions: dict[str, SelectionDecision] = {}
        if spec.selection_policy != "manual":
            if selection_decisions is None:
                raise ValueError(
                    "selection_decisions are required for non-manual selection policies"
                )
            decisions_by_factor: dict[str, dict[str, SelectionDecision]] = {}
            for decision in selection_decisions:
                if not isinstance(decision, SelectionDecision):
                    raise TypeError(
                        "selection_decisions must contain SelectionDecision values"
                    )
                if decision.is_approved and (
                    not decision.has_evidence
                    or not decision.has_gate_results
                    or decision.reason.value not in ("APPROVED", "SHADOWED_COEXIST")
                ):
                    raise ValueError(
                        "approved selection decisions require evidence references, "
                        "gate results, and an APPROVED/SHADOWED_COEXIST reason"
                    )
                by_timestamp = decisions_by_factor.setdefault(decision.factor_id, {})
                existing = by_timestamp.get(decision.timestamp)
                if existing is not None and existing != decision:
                    raise ValueError(
                        "conflicting selection decisions for factor_id "
                        f"{decision.factor_id} at timestamp {decision.timestamp}"
                    )
                by_timestamp[decision.timestamp] = decision

            latest_decisions = {
                factor_id: by_timestamp[max(by_timestamp)]
                for factor_id, by_timestamp in decisions_by_factor.items()
            }
            admitted_factor_ids = {
                factor_id
                for factor_id, decision in latest_decisions.items()
                if decision.is_approved
            }

        matched: list[FactorAsset] = []
        assets_by_id: dict[str, FactorAsset] = {}
        for asset in candidates:
            if not isinstance(asset, FactorAsset):
                raise TypeError("candidates must contain FactorAsset values")
            existing = assets_by_id.get(asset.factor_id)
            if existing is not None:
                if asset != existing:
                    raise ValueError(
                        f"conflicting candidate assets for factor_id: {asset.factor_id}"
                    )
                continue
            assets_by_id[asset.factor_id] = asset
            if (
                admitted_factor_ids is None or asset.factor_id in admitted_factor_ids
            ):
                if admitted_factor_ids is not None and not asset.has_evidence:
                    raise ValueError(
                        f"approved factor asset lacks evidence bundle: {asset.factor_id}"
                    )
                if self._matches(asset, spec):
                    matched.append(asset)

        matched = self._rank(matched, spec, latest_decisions, similarity_provider)
        matched = self._apply_family_constraints(matched, spec.family_constraints)
        if spec.max_factors is not None:
            matched = matched[: spec.max_factors]
        if not matched:
            raise ValueError("no factor assets match the FactorSetSpec")

        factor_ids = tuple(asset.factor_id for asset in matched)
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        memberships = self._build_memberships(
            matched, latest_decisions, spec, admission_artifacts, production
        )
        assembly_hash = self._assembly_hash(factor_ids, spec, memberships)
        policy_hash = self._policy_hash(spec)
        versions = tuple(
            member.factor_version
            if member.factor_version is not None
            else asset.metadata.canonical_hash
            for member, asset in zip(memberships, matched)
        )
        evidence_refs = tuple(
            asset.latest_evidence_ref.bundle_id
            for asset in matched
            if asset.latest_evidence_ref is not None
        )
        return FactorSetArtifact(
            set_id=spec.set_id,
            name=spec.name,
            members=memberships,
            created_at=timestamp,
            policy_hash=policy_hash,
            assembly_hash=assembly_hash,
            snapshot_ref=spec.data_snapshot_ref,
            universe_ref=spec.universe_ref,
            split_ref=spec.split_ref,
            versions=versions,
            evidence_refs=evidence_refs,
            spec=spec,
        )

    @staticmethod
    def _validate_admission_artifacts(
        admission_artifacts: Optional[Mapping[str, FactorAdmissionArtifact]],
    ) -> dict[str, FactorAdmissionArtifact]:
        """Validate the admission-artifact map (typed values, matching keys).

        Every value must be a :class:`FactorAdmissionArtifact` and every key
        must match the artifact's own ``factor_id``.  Returns a plain dict so
        downstream membership construction never touches a caller-owned
        mapping after validation.
        """
        if admission_artifacts is None:
            return {}
        if not isinstance(admission_artifacts, Mapping):
            raise TypeError(
                "admission_artifacts must be a mapping of factor_id to "
                "FactorAdmissionArtifact"
            )
        validated: dict[str, FactorAdmissionArtifact] = {}
        for factor_id, artifact in admission_artifacts.items():
            if not isinstance(artifact, FactorAdmissionArtifact):
                raise TypeError(
                    f"admission_artifacts[{factor_id!r}] must be a "
                    "FactorAdmissionArtifact"
                )
            if factor_id != artifact.factor_id:
                raise ValueError(
                    f"admission_artifacts key {factor_id!r} must match "
                    f"artifact.factor_id {artifact.factor_id!r}"
                )
            validated[factor_id] = artifact
        return validated

    @staticmethod
    def _admission_for(
        factor_id: str,
        admission_artifacts: Mapping[str, FactorAdmissionArtifact],
        production: bool,
    ) -> Optional[FactorAdmissionArtifact]:
        """Return the admission artifact for a factor.

        In production mode a missing artifact is a hard failure — the
        membership's mandatory provenance cannot be resolved, and the
        assembler never fabricates it.
        """
        artifact = admission_artifacts.get(factor_id)
        if artifact is None and production:
            raise ValueError(
                "production assembly requires an admission artifact for every "
                f"member; missing artifact for factor_id {factor_id!r}"
            )
        return artifact

    @staticmethod
    def _rank(
        assets: list[FactorAsset],
        spec: FactorSetSpec,
        decisions: dict[str, SelectionDecision],
        similarity_provider: Optional[
            Union[
                Callable[[str, str], Optional[float]],
                Callable[[str, str], Optional[SimilarityArtifact]],
            ]
        ] = None,
    ) -> list[FactorAsset]:
        """Order candidates by policy-appropriate, evidence-based ranking.

        ``manual`` keeps the legacy deterministic ``factor_id`` ordering.
        ``pareto_front`` ranks by true Pareto dominance over the objective
        metrics carried on the admission decisions' metadata
        (``objectives`` mapping, higher-is-better; wired through
        :class:`factor_assets.optimizer.pareto.ParetoPoint` dominance) with
        decision recency as tiebreak, then ``factor_id``.  When no decision
        carries objectives (metadata-only assembly), all points are
        mutually non-dominated and the ordering falls back to
        recency-then-id, preserving the previous behaviour.

        ``family_robust`` ranks by a composite robustness score read from the
        admission decision's ``metadata`` (``quality``, ``stability``,
        ``worst_slice``, ``recent_decay``, ``tradability``,
        ``residual_novelty``, ``family_budget``), then recency, then
        ``factor_id``.  When the composite scores are absent the ranking fails
        safe to recency ordering — it never invents a score.

        ``diverse`` ranks by greedy Maximal Marginal Relevance (MMR):
        ``lambda*quality - (1-lambda)*max_similarity_to_selected``, where
        quality comes from the decision metadata (fallback to recency rank)
        and similarity comes from the caller-supplied ``similarity_provider``.
        The policy fails closed (raises) when ``diverse`` is selected without
        a similarity provider.

        Ranking never *admits* anyone — it only orders the set already
        admitted by the selection decisions.
        """
        if spec.selection_policy == "manual":
            return sorted(assets, key=lambda asset: asset.factor_id)

        def decision_key(asset: FactorAsset) -> tuple:
            decision = decisions.get(asset.factor_id)
            return (decision.timestamp if decision is not None else "", asset.factor_id)

        # Most recent admission evidence first; factor_id as deterministic
        # tiebreak.  Two stable passes implement (timestamp desc, id asc).
        ranked = sorted(assets, key=lambda asset: asset.factor_id)
        ranked.sort(key=lambda asset: decision_key(asset)[0], reverse=True)

        if spec.selection_policy == "pareto_front":
            objectives = _pareto_objectives(ranked, decisions)
            if objectives:
                ranked = _pareto_rank(ranked, decisions, objectives)
        elif spec.selection_policy == "family_robust":
            ranked = _family_robust_rank(ranked, decisions)
        elif spec.selection_policy == "diverse":
            ranked = _diverse_mmr_rank(ranked, decisions, similarity_provider)
        return ranked

    @staticmethod
    def _build_memberships(
        assets: list[FactorAsset],
        decisions: dict[str, SelectionDecision],
        spec: FactorSetSpec,
        admission_artifacts: Optional[Mapping[str, FactorAdmissionArtifact]] = None,
        production: bool = False,
    ) -> tuple[FactorMembership, ...]:
        """Build per-member provenance from the admission decisions.

        In production mode the mandatory membership fields (``health_state_ref``,
        ``cluster_id``, ``orientation``, ``factor_version``) are populated ONLY
        from the corresponding admission artifact — never synthesized — and a
        missing mandatory field fails closed with ``ValueError``.  In
        non-production mode the fields stay None unless an artifact supplies
        them, preserving the legacy behaviour.
        """
        artifacts = {} if admission_artifacts is None else admission_artifacts
        members: list[FactorMembership] = []
        for rank, asset in enumerate(assets):
            decision = decisions.get(asset.factor_id)
            artifact = FactorSetAssembler._admission_for(
                asset.factor_id, artifacts, production
            )
            health_state_ref = (
                artifact.health_state_ref
                if artifact is not None
                else None
            )
            cluster_id = artifact.cluster_id if artifact is not None else None
            orientation = artifact.orientation if artifact is not None else None
            # factor_version must NEVER fall back to the FE compiler generation
            # — the compiler is a separate identity axis (FactorCompilerIdentity)
            # from the factor definition version (FactorDefinitionIdentity).
            factor_version = (
                artifact.factor_version
                if artifact is not None and artifact.factor_version is not None
                else None
            )
            if production:
                if factor_version is None:
                    raise ValueError(
                        "production assembly requires factor_version from the "
                        f"admission artifact; missing for factor_id {asset.factor_id!r}"
                    )
                if health_state_ref is None:
                    raise ValueError(
                        "production assembly requires health_state_ref from the "
                        f"admission artifact; missing for factor_id {asset.factor_id!r}"
                    )
                if cluster_id is None:
                    raise ValueError(
                        "production assembly requires cluster_id from the "
                        f"admission artifact; missing for factor_id {asset.factor_id!r}"
                    )
                if orientation is None:
                    raise ValueError(
                        "production assembly requires orientation from the "
                        f"admission artifact; missing for factor_id {asset.factor_id!r}"
                    )
            evidence_ref = (
                asset.latest_evidence_ref.bundle_id
                if asset.latest_evidence_ref is not None
                else None
            )
            members.append(
                FactorMembership(
                    factor_id=asset.factor_id,
                    role="member",
                    family_id=asset.family,
                    factor_version=factor_version,
                    health_state_ref=health_state_ref,
                    cluster_id=cluster_id,
                    orientation=orientation,
                    representative_of=(
                        f"cluster:{cluster_id}"
                        if cluster_id is not None
                        else None
                    ),
                    selection_decision_ref=(
                        decision.decision_id if decision is not None else None
                    ),
                    evidence_ref=evidence_ref,
                    novelty_ref=(
                        decision.novelty_refs[0]
                        if decision is not None and decision.novelty_refs
                        else None
                    ),
                    similarity_ref=(
                        decision.similarity_refs[0]
                        if decision is not None and decision.similarity_refs
                        else None
                    ),
                    assembly_score=_decision_score(decision),
                    selection_rank=rank,
                    reason=decision.reason.value if decision is not None else None,
                )
            )
        return tuple(members)

    @staticmethod
    def _assembly_hash(
        factor_ids: tuple[str, ...],
        spec: FactorSetSpec,
        memberships: tuple[FactorMembership, ...],
    ) -> str:
        """Canonical content hash over the assembly identity.

        Covers every semantic field of every member (identity + role +
        orientation + family/cluster + representative + every provenance ref +
        score + rank) plus the snapshot / universe / split / policy identity.
        The member fields are length-prefixed in a fixed canonical order, so a
        new field added to :class:`FactorMembership` must be added here too
        rather than hand-picking a subset that can silently drift.
        """
        digest = hashlib.sha256()

        def _prefixed(field_value: object) -> None:
            encoded = str(field_value).encode("utf-8")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)

        for member in memberships:
            # Canonical, exhaustive semantic field order (matches FactorMembership).
            for field_value in (
                member.factor_id,
                member.role,
                member.orientation,
                member.family_id,
                member.cluster_id,
                member.factor_version,
                member.representative_of,
                member.selection_decision_ref,
                member.evidence_ref,
                member.novelty_ref,
                member.similarity_ref,
                member.health_state_ref,
                member.assembly_score,
                member.selection_rank,
                member.reason,
            ):
                _prefixed(field_value)

        # Assembly identity: snapshot / universe / split / policy identity.
        _prefixed(spec.set_id)
        _prefixed(spec.selection_policy)
        _prefixed(spec.data_snapshot_ref)
        _prefixed(spec.universe_ref)
        _prefixed(spec.split_ref)
        _prefixed(spec.frequency)
        _prefixed(spec.max_factors)
        _prefixed(spec.min_evidence_date)
        _prefixed(spec.required_domains)
        _prefixed(spec.excluded_domains)
        _prefixed(spec.min_lifecycle_state)
        _prefixed(spec.family_constraints)
        return digest.hexdigest()

    @staticmethod
    def _policy_hash(spec: FactorSetSpec) -> str:
        """Hash of the fields that define selection semantics."""
        digest = hashlib.sha256()
        semantic_fields = (
            spec.selection_policy,
            spec.universe_ref or "",
            spec.frequency or "",
            str(spec.max_factors),
            str(spec.min_evidence_date),
            *spec.required_domains,
            "#",
            *spec.excluded_domains,
            str(spec.min_lifecycle_state),
            spec.family_constraints or "",
            spec.split_ref or "",
        )
        for field in semantic_fields:
            encoded = field.encode("utf-8")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _apply_family_constraints(
        assets: list[FactorAsset], constraints: Optional[str]
    ) -> list[FactorAsset]:
        """Apply the small, explicit family-constraint grammar.

        Supported syntax is ``max_per_family=N`` (positive integer).  A missing
        family is treated as its own bucket.  Other strings fail closed rather
        than silently changing selection semantics.
        """
        if constraints is None or not constraints.strip():
            return assets
        key, separator, value = constraints.partition("=")
        if not separator or key.strip() != "max_per_family":
            raise ValueError(
                "family_constraints must use 'max_per_family=N' syntax"
            )
        try:
            limit = int(value.strip())
        except ValueError as exc:
            raise ValueError("max_per_family must be a positive integer") from exc
        if limit < 1:
            raise ValueError("max_per_family must be a positive integer")
        counts: dict[Optional[str], int] = {}
        selected: list[FactorAsset] = []
        for asset in assets:
            family = asset.family
            count = counts.get(family, 0)
            if count < limit:
                selected.append(asset)
                counts[family] = count + 1
        return selected

    @staticmethod
    def _matches(asset: FactorAsset, spec: FactorSetSpec) -> bool:
        metadata = asset.metadata
        if spec.frequency is not None and metadata.frequency != spec.frequency:
            return False
        domains = set(metadata.domains)
        if not set(spec.required_domains).issubset(domains):
            return False
        if domains.intersection(spec.excluded_domains):
            return False
        if spec.min_lifecycle_state is not None:
            try:
                minimum = LifecycleState(spec.min_lifecycle_state)
            except ValueError as exc:
                raise ValueError(
                    f"unknown min_lifecycle_state: {spec.min_lifecycle_state}"
                ) from exc
            if _LIFECYCLE_RANK.get(asset.lifecycle_state, -1) < _LIFECYCLE_RANK[minimum]:
                return False
        if spec.min_evidence_date is not None:
            evidence_timestamp = (
                asset.latest_evidence_ref.timestamp
                if asset.latest_evidence_ref is not None
                else None
            )
            if evidence_timestamp is None or evidence_timestamp < spec.min_evidence_date:
                return False
        return True


_LIFECYCLE_RANK = {
    LifecycleState.REGISTERED: 0,
    LifecycleState.EVALUATED: 1,
    LifecycleState.APPROVED: 2,
    LifecycleState.PRODUCTION_READY: 3,
    LifecycleState.DEPRECATED: -1,
    LifecycleState.RETIRED: -1,
}


def _decision_objectives(
    decision: Optional[SelectionDecision],
) -> Optional[dict[str, float]]:
    """Extract higher-is-better objectives from a decision's metadata.

    ``pareto_front`` selection carries the per-factor objective metrics on
    the admission decision's ``metadata`` under the ``objectives`` key
    (a mapping of objective name to finite float).  Returns None when the
    decision or the objectives are absent or malformed — a factor without
    measurable objectives is never silently scored.
    """
    if decision is None:
        return None
    raw = decision.metadata.get("objectives") if decision.metadata else None
    if not isinstance(raw, Mapping):
        return None
    objectives: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        objectives[str(name)] = value
    return objectives or None


def _pareto_objectives(
    assets: list[FactorAsset],
    decisions: dict[str, SelectionDecision],
) -> list[str]:
    """Objective names shared by every ranked asset's decision.

    Fail-closed on the union: any objective missing from one decision would
    make that factor's dominance comparisons undefined (missing objectives
    default to -inf in ParetoPoint, silently demoting it), so only the
    intersection — and only when every decision carries objectives — is a
    valid objective space.
    """
    common: Optional[set[str]] = None
    for asset in assets:
        objectives = _decision_objectives(decisions.get(asset.factor_id))
        if objectives is None:
            return []
        keys = set(objectives)
        common = keys if common is None else (common & keys)
    if not common:
        return []
    return sorted(common)


def _pareto_rank(
    assets: list[FactorAsset],
    decisions: dict[str, SelectionDecision],
    objectives: list[str],
) -> list[FactorAsset]:
    """Rank assets by Pareto dominance (non-dominated first).

    Uses :class:`factor_assets.optimizer.pareto.ParetoPoint` dominance —
    a point is dominated when another is at least as good on every
    objective and strictly better on at least one.  Non-dominated assets
    rank ahead of dominated ones; within each group the input order
    (recency desc, factor_id asc) is preserved, keeping the ordering
    deterministic and the previous behaviour intact when nothing is
    dominated.
    """
    points = {
        asset.factor_id: ParetoPoint(
            point_id=asset.factor_id,
            objectives=_decision_objectives(decisions.get(asset.factor_id)) or {},
        )
        for asset in assets
    }
    non_dominated: list[FactorAsset] = []
    dominated: list[FactorAsset] = []
    for asset in assets:
        point = points[asset.factor_id]
        is_dominated = any(
            points[other.factor_id].dominates(point, objectives)
            for other in assets
            if other.factor_id != asset.factor_id
        )
        (dominated if is_dominated else non_dominated).append(asset)
    return non_dominated + dominated

def _decision_score(decision: Optional[SelectionDecision]) -> Optional[float]:
    """Extract a finite composite score from a decision's metadata.

    Reads the ``score`` key (a finite float) from the decision metadata.  Used
    to populate ``FactorMembership.assembly_score``.  Returns None when absent
    or malformed — a member is never scored on a fabricated value.
    """
    if decision is None or not decision.metadata:
        return None
    raw = decision.metadata.get("score")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _family_robust_rank(
    assets: list[FactorAsset],
    decisions: dict[str, SelectionDecision],
) -> list[FactorAsset]:
    """Rank by a composite family-robustness score, then recency, then id.

    The composite score is read from the admission decision's ``metadata``
    under the ``family_robust`` key (a finite float combining quality,
    stability, worst_slice, recent_decay, tradability, residual_novelty and
    family_budget — computed upstream by the policy).  When the composite is
    absent the ranking fails safe to the recency ordering already established
    by the caller; it never invents a score.
    """
    def score(asset: FactorAsset) -> Optional[float]:
        decision = decisions.get(asset.factor_id)
        if decision is None or not decision.metadata:
            return None
        raw = decision.metadata.get("family_robust")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value

    scored = [(asset, score(asset)) for asset in assets]
    # Stable sort: composite desc first, then preserve the recency ordering
    # (which the caller already established) for assets without a score.
    scored.sort(key=lambda pair: pair[0].factor_id)
    scored.sort(key=lambda pair: pair[1] if pair[1] is not None else float("-inf"), reverse=True)
    return [asset for asset, _ in scored]


def _diverse_mmr_rank(
    assets: list[FactorAsset],
    decisions: dict[str, SelectionDecision],
    similarity_provider: Optional[
        Union[
            Callable[[str, str], Optional[float]],
            Callable[[str, str], Optional[SimilarityArtifact]],
        ]
    ],
) -> list[FactorAsset]:
    """Rank by greedy Maximal Marginal Relevance (MMR).

    MMR score for candidate ``c`` given the already-selected set ``S`` is::

        lambda * quality(c) - (1 - lambda) * max_{s in S} similarity(c, s)

    Quality comes from the decision metadata ``quality`` key (finite float);
    when absent it falls back to the recency rank (higher recency = higher
    quality).  Similarity comes from the caller-supplied ``similarity_provider``,
    which may return either a bare float (backward compatible) or a
    :class:`SimilarityArtifact` whose ``primary_view`` value is consumed
    (``None`` similarity — from either form — is treated as zero).  The policy
    fails closed (raises) when ``diverse`` is selected without a similarity
    provider, so this helper is only reached with a provider present.
    """
    if similarity_provider is None:
        raise ValueError(
            "diverse (MMR) selection requires a similarity_provider; "
            "failing closed rather than silently degrading to non-MMR"
        )

    lambda_weight = 0.5  # balance quality vs. diversity

    def quality(asset: FactorAsset, recency_rank: int) -> float:
        decision = decisions.get(asset.factor_id)
        if decision is not None and decision.metadata:
            raw = decision.metadata.get("quality")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                value = float(raw)
                if value == value and value not in (float("inf"), float("-inf")):
                    return value
        # Fallback: higher recency (lower rank index) = higher quality.
        return float(len(assets) - recency_rank)

    def similarity_value(factor_a: str, factor_b: str) -> float:
        """Similarity magnitude for MMR.

        An ``UNKNOWN`` similarity (provider returning ``None``, or an artifact
        whose primary view is ``None``) MUST NOT be conflated with a computed
        zero.  In production a missing measurement is a hard failure — MMR
        would otherwise reward exactly the pairs it cannot assess.  This
        helper raises, forcing a caller to supply a real similarity or an
        explicit UNKNOWN policy; a bare ``float`` is treated as a computed
        similarity (abs-normalized).
        """
        result = similarity_provider(factor_a, factor_b)
        if isinstance(result, SimilarityArtifact):
            value = result.primary_value
        else:
            value = result
        if value is None:
            raise ValueError(
                "diverse (MMR) selection received an UNKNOWN similarity "
                f"(None) for pair ({factor_a!r}, {factor_b!r}); an unmeasured "
                "similarity must not be treated as zero. Supply a real "
                "similarity or an explicit UNKNOWN policy."
            )
        return abs(float(value))

    # Precompute quality for each asset (recency rank = index in input order).
    quality_by_id = {
        asset.factor_id: quality(asset, idx)
        for idx, asset in enumerate(assets)
    }

    remaining = list(assets)
    selected: list[FactorAsset] = []
    selected_ids: list[str] = []

    while remaining:
        best_asset = None
        best_score = float("-inf")
        for asset in remaining:
            q = quality_by_id[asset.factor_id]
            if selected_ids:
                max_sim = max(
                    similarity_value(asset.factor_id, s)
                    for s in selected_ids
                )
            else:
                max_sim = 0.0
            mmr = lambda_weight * q - (1.0 - lambda_weight) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_asset = asset
        if best_asset is None:
            break
        selected.append(best_asset)
        selected_ids.append(best_asset.factor_id)
        remaining.remove(best_asset)

    return selected


__all__ = ["FactorSetAssembler"]
