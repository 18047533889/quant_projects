"""Deterministic assembly of factor sets from registered assets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from factor_assets.contracts.asset import FactorAsset
from factor_assets.contracts.factor_set import FactorSet, FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.selection.policy import SelectionDecision


class FactorSetAssembler:
    """Build immutable :class:`FactorSet` values from factor asset metadata.

    Assembly is deliberately metadata-only.  Candidates are filtered using the
    fields represented by ``FactorSetSpec`` and the resulting IDs are sorted by
    ``factor_id`` before applying ``max_factors``.  No factor values or metric
    computations are performed here.
    """

    def assemble(
        self,
        spec: FactorSetSpec,
        candidates: Iterable[FactorAsset],
        *,
        selection_decisions: Optional[Iterable[SelectionDecision]] = None,
        selection_run_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> FactorSet:
        """Assemble a factor set from candidate assets.

        Args:
            spec: Selection criteria and output identity.
            candidates: Registered factor assets to consider.
            selection_decisions: Admission decisions for non-manual policies.
            selection_run_id: Optional provenance identifier for this run.
            created_at: Optional explicit creation timestamp, useful for replay.

        Raises:
            TypeError: If ``spec`` or a candidate is not the expected contract.
            ValueError: If no candidates match or ``max_factors`` is invalid.
        """
        if not isinstance(spec, FactorSetSpec):
            raise TypeError("spec must be a FactorSetSpec")
        if spec.max_factors is not None and spec.max_factors < 1:
            raise ValueError("max_factors must be >= 1")

        admitted_factor_ids: Optional[set[str]] = None
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
                    or decision.reason.value != "APPROVED"
                ):
                    raise ValueError(
                        "approved selection decisions require evidence references, "
                        "gate results, and an APPROVED reason"
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

        matched.sort(key=lambda asset: asset.factor_id)
        matched = self._apply_family_constraints(matched, spec.family_constraints)
        if spec.max_factors is not None:
            matched = matched[: spec.max_factors]
        if not matched:
            raise ValueError("no factor assets match the FactorSetSpec")

        factor_ids = tuple(asset.factor_id for asset in matched)
        families = tuple(sorted({asset.family for asset in matched if asset.family is not None}))
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        return FactorSet(
            set_id=spec.set_id,
            name=spec.name,
            factor_ids=factor_ids,
            created_at=timestamp,
            spec=spec,
            universe_ref=spec.universe_ref,
            frequency=spec.frequency,
            selection_run_id=selection_run_id,
            families=families,
            description=spec.description,
        )

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

__all__ = ["FactorSetAssembler"]
