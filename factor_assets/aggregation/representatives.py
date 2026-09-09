"""
Family representative selection strategies.

Selects representative factors from families based on:
- Maximum IC (information coefficient)
- Minimum intra-family correlation
- Equal weight from each subfamily

FA stores only selection metadata and factor references, not raw values.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Protocol
import math

@dataclass(frozen=True)
class ICEvidenceReceipt:
    factor_id: str
    value: float
    evidence_ref: str
    universe_ref: str
    window_ref: str
    snapshot_ref: str
    split_ref: str
    label_ref: str
    comparison_context_hash: str
    mature: bool
    def __post_init__(self):
        if not all((self.factor_id,self.evidence_ref,self.universe_ref,self.window_ref,self.snapshot_ref,self.split_ref,self.label_ref,self.comparison_context_hash)): raise ValueError("complete IC evidence identity/context required")
        if isinstance(self.value,bool) or not isinstance(self.value,(int,float)) or not math.isfinite(self.value): raise ValueError("IC evidence value must be a finite non-boolean number")
        if not self.mature: raise ValueError("IC evidence must be mature")


class RepresentativeSelectionMethod(Enum):
    """Methods for selecting family representatives."""
    MAX_IC = "max_ic"                      # Highest IC factor per family
    MIN_CORRELATION = "min_correlation"    # Lowest avg correlation to family
    EQUAL_WEIGHT = "equal_weight"          # Equal representation per subfamily
    FIRST = "first"                        # First factor (registration order)
    RANDOM = "random"                      # Random selection (with seed)


@dataclass(frozen=True)
class RepresentativeSelection:
    """
    Result of representative factor selection from a family.

    Contains selected factor IDs and selection metadata, not raw values.

    DLIB-FA-014: MAX_IC must bind to an EvaluationArtifactRef / window /
    snapshot / universe / split / label / evidence maturity — not a bare
    ``ICProvider.get_ic`` float.  ``ic_evidence_refs`` records the evidence
    reference that justified each selected representative's IC, so a
    representative is traceable to the evidence that picked it.
    """
    selection_id: str
    family: str
    method: RepresentativeSelectionMethod
    selected_factor_ids: tuple[str, ...]
    timestamp: str  # ISO 8601
    # Selection context
    candidate_factor_ids: tuple[str, ...] = ()
    universe_ref: Optional[str] = None
    lookback_periods: Optional[int] = None
    # Method-specific metadata
    ic_values: Optional[tuple[float, ...]] = None  # For MAX_IC
    avg_correlations: Optional[tuple[float, ...]] = None  # For MIN_CORRELATION
    subfamily_assignments: Optional[tuple[str, ...]] = None  # For EQUAL_WEIGHT
    random_seed: Optional[int] = None  # For RANDOM
    # Quality metrics
    selection_score: Optional[float] = None
    diversity_score: Optional[float] = None
    warnings: tuple[str, ...] = ()
    # DLIB-FA-014: evidence refs binding each selected representative to the
    # evidence that justified its selection (one per selected factor).
    ic_evidence_refs: tuple[Optional[str], ...] = ()
    # DLIB-FA-014: the evaluation window / snapshot / split the IC was measured
    # on (provenance for the MAX_IC binding).
    ic_window_ref: Optional[str] = None
    ic_snapshot_ref: Optional[str] = None
    ic_split_ref: Optional[str] = None
    selection_role: str = "CENTRAL"

    def __post_init__(self):
        if not self.selection_id:
            raise ValueError("selection_id is required")
        if not self.family:
            raise ValueError("family is required")
        if not self.selected_factor_ids:
            raise ValueError("selected_factor_ids cannot be empty")
        if not self.timestamp:
            raise ValueError("timestamp is required")

        if self.ic_values:
            if len(self.ic_values) != len(self.selected_factor_ids):
                raise ValueError("ic_values length must match selected_factor_ids")

        if self.avg_correlations:
            if len(self.avg_correlations) != len(self.selected_factor_ids):
                raise ValueError("avg_correlations length must match selected_factor_ids")

        if self.ic_evidence_refs:
            if len(self.ic_evidence_refs) != len(self.selected_factor_ids):
                raise ValueError(
                    "ic_evidence_refs length must match selected_factor_ids"
                )

    @property
    def num_selected(self) -> int:
        """Get number of selected representatives."""
        return len(self.selected_factor_ids)

    @property
    def num_candidates(self) -> int:
        """Get number of candidate factors considered."""
        return len(self.candidate_factor_ids)

    @property
    def selection_ratio(self) -> float:
        """Get ratio of selected to candidate factors."""
        if self.num_candidates == 0:
            return 0.0
        return self.num_selected / self.num_candidates

    @property
    def has_warnings(self) -> bool:
        """Check if selection produced warnings."""
        return len(self.warnings) > 0


class ICProvider(Protocol):
    """
    Protocol for providing IC (information coefficient) values.

    FA does not compute IC directly; it comes from QE or external provider.
    """

    def get_ic(
        self,
        factor_id: str,
        universe_ref: Optional[str] = None,
        lookback_periods: Optional[int] = None,
    ) -> Optional[ICEvidenceReceipt]:
        """
        Get IC value for a factor.

        Args:
            factor_id: Factor identifier
            universe_ref: Optional universe constraint
            lookback_periods: Optional lookback window

        Returns:
            IC value if available, None otherwise
        """
        ...


class CorrelationProvider(Protocol):
    """
    Protocol for providing correlation values between factors.

    FA does not compute correlations directly; delegates to similarity service.
    """

    def get_avg_correlation(
        self,
        factor_id: str,
        other_factor_ids: tuple[str, ...],
        universe_ref: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get average correlation of factor to a group of factors.

        Args:
            factor_id: Factor identifier
            other_factor_ids: Group of factors to correlate with
            universe_ref: Optional universe constraint

        Returns:
            Average absolute correlation if available, None otherwise
        """
        ...


@dataclass
class FamilyRepresentativeSelector:
    """
    Selects representative factors from families.

    Uses external providers for IC and correlation data.
    Does not compute or store raw factor values.
    """
    ic_provider: Optional[ICProvider] = None
    correlation_provider: Optional[CorrelationProvider] = None

    def select_representatives(
        self,
        family: str,
        factor_ids: tuple[str, ...],
        method: RepresentativeSelectionMethod,
        max_representatives: int = 1,
        universe_ref: Optional[str] = None,
        lookback_periods: Optional[int] = None,
        random_seed: Optional[int] = None,
        subfamily_assignments: Optional[Mapping[str,str]] = None,
        ic_window_ref: Optional[str] = None,
        ic_snapshot_ref: Optional[str] = None,
        ic_split_ref: Optional[str] = None,
        ic_label_ref: Optional[str] = None,
        comparison_context_hash: Optional[str] = None,
    ) -> RepresentativeSelection:
        """
        Select representative factors from a family.

        Args:
            family: Family name
            factor_ids: Candidate factor IDs
            method: Selection method
            max_representatives: Maximum number to select
            universe_ref: Optional universe constraint
            lookback_periods: Optional lookback window
            random_seed: Random seed for RANDOM method

        Returns:
            RepresentativeSelection with selected factors

        Raises:
            ValueError: If method requires provider that is not configured
        """
        if not factor_ids:
            raise ValueError("factor_ids cannot be empty")
        if max_representatives < 1:
            raise ValueError("max_representatives must be >= 1")
        if max_representatives > len(factor_ids):
            max_representatives = len(factor_ids)

        from datetime import datetime

        timestamp = datetime.utcnow().isoformat() + "Z"
        selection_id = f"{family}_{method.value}_{timestamp}"
        warnings = []

        if method == RepresentativeSelectionMethod.MAX_IC:
            selected, ic_values, warn = self._select_max_ic(
                factor_ids, max_representatives, universe_ref, lookback_periods,
                ic_window_ref, ic_snapshot_ref, ic_split_ref, ic_label_ref, comparison_context_hash
            )
            warnings.extend(warn)
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
                universe_ref=universe_ref,
                lookback_periods=lookback_periods,
                ic_values=ic_values,
                ic_evidence_refs=tuple(self._last_ic_refs),
                ic_window_ref=ic_window_ref, ic_snapshot_ref=ic_snapshot_ref, ic_split_ref=ic_split_ref,
                warnings=tuple(warnings),
            )

        elif method == RepresentativeSelectionMethod.MIN_CORRELATION:
            selected, avg_corrs, warn = self._select_min_correlation(
                factor_ids, max_representatives, universe_ref
            )
            warnings.extend(warn)
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
                universe_ref=universe_ref,
                avg_correlations=avg_corrs,
                selection_role="DIVERSITY",
                warnings=tuple(warnings),
            )

        elif method == RepresentativeSelectionMethod.EQUAL_WEIGHT:
            if not subfamily_assignments or set(subfamily_assignments) != set(factor_ids):
                raise ValueError("EQUAL_WEIGHT requires one subfamily assignment per factor")
            selected = self._select_equal_weight(factor_ids, max_representatives, subfamily_assignments)
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
                universe_ref=universe_ref,
                subfamily_assignments=tuple(subfamily_assignments[f] for f in selected),
                selection_role="SUBFAMILY_BALANCED",
            )

        elif method == RepresentativeSelectionMethod.FIRST:
            selected = factor_ids[:max_representatives]
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
            )

        elif method == RepresentativeSelectionMethod.RANDOM:
            import random
            # Use local Random instance to avoid polluting global RNG state
            rng = random.Random(random_seed)
            selected = tuple(rng.sample(list(factor_ids), max_representatives))
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
                random_seed=random_seed,
            )

        raise ValueError(f"Unknown method: {method}")

    def _select_max_ic(
        self,
        factor_ids: tuple[str, ...],
        max_representatives: int,
        universe_ref: Optional[str],
        lookback_periods: Optional[int],
        window_ref: Optional[str], snapshot_ref: Optional[str], split_ref: Optional[str],
        label_ref: Optional[str], comparison_context_hash: Optional[str],
    ) -> tuple[tuple[str, ...], tuple[float, ...], list[str]]:
        """Select factors with highest IC."""
        if self.ic_provider is None:
            raise ValueError("MAX_IC method requires ic_provider")

        warnings = []
        ic_data = []

        for fid in factor_ids:
            receipt = self.ic_provider.get_ic(fid, universe_ref, lookback_periods)
            if receipt is None:
                warnings.append(f"No IC data for {fid}, skipping")
                # Do NOT fill with 0.0 - skip factors without evidence
                continue
            if not isinstance(receipt, ICEvidenceReceipt): raise TypeError("ICProvider must return ICEvidenceReceipt, not a bare float")
            if (receipt.factor_id!=fid or receipt.universe_ref!=universe_ref or receipt.window_ref!=window_ref
                    or receipt.snapshot_ref!=snapshot_ref or receipt.split_ref!=split_ref or receipt.label_ref!=label_ref
                    or receipt.comparison_context_hash!=comparison_context_hash): raise ValueError("IC evidence context mismatch")
            ic_data.append((fid, receipt.value, receipt.evidence_ref))

        if not ic_data:
            raise ValueError(
                "No IC data available for any candidate factors. "
                "Cannot select representatives without evidence."
            )

        # Sort by descending absolute IC
        ic_data.sort(key=lambda x: (-x[1], x[0]))

        selected_ids = tuple(fid for fid, _, _ in ic_data[:max_representatives])
        selected_ics = tuple(ic for _, ic, _ in ic_data[:max_representatives])
        self._last_ic_refs = tuple(ref for _, _, ref in ic_data[:max_representatives])

        return selected_ids, selected_ics, warnings

    def _select_min_correlation(
        self,
        factor_ids: tuple[str, ...],
        max_representatives: int,
        universe_ref: Optional[str],
    ) -> tuple[tuple[str, ...], tuple[float, ...], list[str]]:
        """Select factors with minimum average intra-family correlation."""
        if self.correlation_provider is None:
            raise ValueError("MIN_CORRELATION method requires correlation_provider")

        warnings = []
        corr_data = []

        for fid in factor_ids:
            # Compute average correlation to all other factors in family
            other_ids = tuple(f for f in factor_ids if f != fid)
            avg_corr = self.correlation_provider.get_avg_correlation(
                fid, other_ids, universe_ref
            )
            if avg_corr is None:
                warnings.append(f"No correlation data for {fid}, skipping")
                # Do NOT fill with pessimistic default - skip factors without evidence
                continue
            corr_data.append((fid, avg_corr))

        if not corr_data:
            raise ValueError(
                "No correlation data available for any candidate factors. "
                "Cannot select representatives without evidence."
            )

        # Sort by ascending average correlation
        corr_data.sort(key=lambda x: x[1])

        selected_ids = tuple(fid for fid, _ in corr_data[:max_representatives])
        selected_corrs = tuple(corr for _, corr in corr_data[:max_representatives])

        return selected_ids, selected_corrs, warnings

    def _select_equal_weight(
        self,
        factor_ids: tuple[str, ...],
        max_representatives: int,
        assignments: Mapping[str,str],
    ) -> tuple[str, ...]:
        """Select evenly spaced factors for equal representation."""
        if max_representatives >= len(factor_ids):
            return factor_ids

        groups={}
        for fid in sorted(factor_ids): groups.setdefault(assignments[fid],[]).append(fid)
        selected=[]
        ordered=sorted(groups)
        while len(selected)<max_representatives:
            changed=False
            for group in ordered:
                if groups[group] and len(selected)<max_representatives:
                    selected.append(groups[group].pop(0)); changed=True
            if not changed: break
        return tuple(selected)
