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
from typing import Optional, Protocol


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
    ) -> Optional[float]:
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
                factor_ids, max_representatives, universe_ref, lookback_periods
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
                warnings=tuple(warnings),
            )

        elif method == RepresentativeSelectionMethod.EQUAL_WEIGHT:
            # For equal weight, select evenly spaced factors
            selected = self._select_equal_weight(factor_ids, max_representatives)
            return RepresentativeSelection(
                selection_id=selection_id,
                family=family,
                method=method,
                selected_factor_ids=selected,
                timestamp=timestamp,
                candidate_factor_ids=factor_ids,
                universe_ref=universe_ref,
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
            if random_seed is not None:
                random.seed(random_seed)
            selected = tuple(random.sample(list(factor_ids), max_representatives))
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
    ) -> tuple[tuple[str, ...], tuple[float, ...], list[str]]:
        """Select factors with highest IC."""
        if self.ic_provider is None:
            raise ValueError("MAX_IC method requires ic_provider")

        warnings = []
        ic_data = []

        for fid in factor_ids:
            ic = self.ic_provider.get_ic(fid, universe_ref, lookback_periods)
            if ic is None:
                warnings.append(f"No IC data for {fid}")
                ic = 0.0
            ic_data.append((fid, abs(ic)))  # Use absolute IC

        # Sort by descending absolute IC
        ic_data.sort(key=lambda x: x[1], reverse=True)

        selected_ids = tuple(fid for fid, _ in ic_data[:max_representatives])
        selected_ics = tuple(ic for _, ic in ic_data[:max_representatives])

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
                warnings.append(f"No correlation data for {fid}")
                avg_corr = 1.0  # Pessimistic default
            corr_data.append((fid, avg_corr))

        # Sort by ascending average correlation
        corr_data.sort(key=lambda x: x[1])

        selected_ids = tuple(fid for fid, _ in corr_data[:max_representatives])
        selected_corrs = tuple(corr for _, corr in corr_data[:max_representatives])

        return selected_ids, selected_corrs, warnings

    def _select_equal_weight(
        self,
        factor_ids: tuple[str, ...],
        max_representatives: int,
    ) -> tuple[str, ...]:
        """Select evenly spaced factors for equal representation."""
        if max_representatives >= len(factor_ids):
            return factor_ids

        # Select evenly spaced indices
        n = len(factor_ids)
        step = n / max_representatives
        indices = [int(i * step) for i in range(max_representatives)]

        return tuple(factor_ids[i] for i in indices)
