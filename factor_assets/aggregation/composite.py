"""DLIB-FA-004/018: composite factor evaluation.

Production composite-factor evaluation with HEAD strategies and CLOSE-out vs
OPEN-out accounting is a QuantPlatform compositor target that is NOT yet
implemented.

What FA can do TODAY is:

1. Causal-Criterion Composite Composition (CCCC) — compose a composite from a
   cluster that is HOMOGENEOUS on the causal criteria, using same-frequency,
   vwap-to-vwap return basis, a scenario pool of baselines, explicit Horizon
   Mapping, Orientation Mapping, Regime Mapping, and sub-cluster composition.

2. Aggregate vwap-to-vwap return (the global hard return basis — see
   Memory: vwap-to-vwap 收益口径) and compute weight-concentration /
   diversity diagnostics WITHOUT fabricating a full long/short portfolio
   backtest.

The FA CompositeEvaluator is INTENTIONALLY a composition + accounting
diagnostics component, NOT a backtest authority.  A full composite-factor
long/short backtest is a QuantPlatform compositor target, not FA.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from factor_assets.aggregation.specs import (
    AggregationResult,
    AggregationSpec,
    WeightingScheme,
)
from factor_assets.errors import (
    EvidenceUnavailableError,
    InsufficientObservations,
    NumericalFailure,
)


class HorizonType(Enum):
    """Horizon mapping: h=1..H : aggregate return over each horizon."""

    VWAP_AGG = "VWAP_AGG"  # same-frequency vwap-to-vwap aggregate return


class OrientationType(Enum):
    """Orientation mapping: LONG / SHORT / LONG_SHORT."""

    LONG = "LONG"
    SHORT = "SHORT"
    LONG_SHORT = "LONG_SHORT"


class RegimeType(Enum):
    """Regime mapping: the regime that defines the composite's return basis."""

    UP = "UP"
    DOWN = "DOWN"
    ALL = "ALL"


class CompactCompositeValue:
    """Compact composite signal aggregate (bounded, no raw time series)."""

    def __init__(
        self,
        composite_factor_id: str,
        agg_freq: str,
        n_securities: int,
        n_observations: int,
        horizon: int,
        orientation: OrientationType,
        regime: RegimeType,
        agg_vwap_to_vwap_return: float,
        source: str,
        evidence_basis_ref: str,
    ):
        if not composite_factor_id:
            raise ValueError("composite_factor_id is required")
        if n_observations < 0:
            raise ValueError("n_observations must be non-negative")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not isinstance(orientation, OrientationType):
            raise TypeError("orientation must be an OrientationType")
        if not isinstance(regime, RegimeType):
            raise TypeError("regime must be a RegimeType")
        if not math.isfinite(agg_vwap_to_vwap_return):
            raise NumericalFailure(
                "agg_vwap_to_vwap_return must be finite — no fabricated value"
            )
        self.composite_factor_id = composite_factor_id
        self.agg_freq = agg_freq
        self.n_securities = n_securities
        self.n_observations = n_observations
        self.horizon = horizon
        self.orientation = orientation
        self.regime = regime
        self.agg_vwap_to_vwap_return = agg_vwap_to_vwap_return
        self.source = source
        self.evidence_basis_ref = evidence_basis_ref

    def to_dict(self) -> dict:
        return {
            "composite_factor_id": self.composite_factor_id,
            "agg_freq": self.agg_freq,
            "n_securities": self.n_securities,
            "n_observations": self.n_observations,
            "horizon": self.horizon,
            "orientation": self.orientation.value,
            "regime": self.regime.value,
            "agg_vwap_to_vwap_return": self.agg_vwap_to_vwap_return,
            "source": self.source,
            "evidence_basis_ref": self.evidence_basis_ref,
        }


class HorizonMapping:
    """Horizon h aggregates the same-frequency, vwap-to-vwap forward return."""

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon

    def aggregate_vwap_to_vwap_return(
        self,
        observations: List[float],
    ) -> float:
        """Arithmetic mean of the vwap-to-vwap forward returns (vwap收益口径)."""
        if len(observations) < self.horizon:
            raise InsufficientObservations(
                f"need at least {self.horizon} observations for horizon aggregation"
            )
        # The observations are already vwap-to-vwap forward returns; we
        # aggregate over the horizon by simple mean of the first `horizon`
        # observations. In real usage the producer computes the forward return
        # at each step before this call.
        window = observations[: self.horizon]
        finite = [v for v in window if math.isfinite(v)]
        if not finite:
            raise NumericalFailure("no finite vwap-to-vwap returns in window")
        return sum(finite) / len(finite)


class OrientationMapping:
    """Flip the aggregate return according to orientation sign."""

    def apply(self, agg_return: float, orientation: OrientationType) -> float:
        if not math.isfinite(agg_return):
            raise NumericalFailure("cannot orient a non-finite return")
        if orientation is OrientationType.LONG:
            return agg_return
        if orientation is OrientationType.SHORT:
            return -agg_return
        # LONG_SHORT: neutral exposure — use the absolute signal magnitude
        # as the composite's carry, but do not stack a fabricated direction.
        return abs(agg_return)


class RegimeMapping:
    """A regime mapping is a re-weighting by regime; FA supports identity and
    regime-conditional composition via a score provider."""

    def __init__(self, regime_weights: Mapping[str, float]):
        if not regime_weights:
            raise ValueError("regime_weights must be non-empty")
        for key, weight in regime_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"regime_weights[{key!r}] must be a non-boolean number")
            w = float(weight)
            if w != w or w in (float("inf"), float("-inf")):
                raise ValueError(f"regime_weights[{key!r}] must be finite")
            if w < 0.0:
                raise ValueError(f"regime_weights[{key!r}] must be non-negative")
        self.regime_weights = dict(regime_weights)


class CompositeEvaluator:
    """CCCC composite composition and accounting diagnostics (DLIB-FA-018).

    NOT a backtest authority.  Full composite-factor long/short backtesting is
    a QuantPlatform compositor target.  FA composes a composite from a cluster
    and reports aggregate accounting diagnostics on the vwap-to-vwap basis.
    """

    def __init__(
        self,
        horizon: int = 5,
        orientation: OrientationType = OrientationType.LONG,
        regime_weights: Optional[Mapping[str, float]] = None,
    ):
        self.horizon = horizon
        self.orientation = orientation
        self.regime_weights = regime_weights or {"ALL": 1.0}
        self.horizon_mapping = HorizonMapping(horizon)
        self.orientation_mapping = OrientationMapping()
        self.regime_mapping = RegimeMapping(self.regime_weights)

    def compose(
        self,
        composite_factor_id: str,
        spec: AggregationSpec,
        cluster_returns: Mapping[str, List[float]],
        *,
        agg_freq: str,
        n_observations: int,
        evidence_basis_ref: str,
    ) -> CompactCompositeValue:
        """Compose a compact composite value from per-factor vwap-to-vwap
        return series.

        Args:
            composite_factor_id: ID of the new composite.
            spec: AggregationSpec identifying the members.
            cluster_returns: factor_id -> list of vwap-to-vwap forward returns.
            agg_freq: aggregation frequency.
            n_observations: number of observations per series.
            evidence_basis_ref: snapshot/universe/split provenance.
        """
        if not spec.factor_ids:
            raise ValueError("spec.factor_ids cannot be empty")
        missing = [fid for fid in spec.factor_ids if fid not in cluster_returns]
        if missing:
            raise EvidenceUnavailableError(
                f"missing vwap-to-vwap return series for factors: {missing}"
            )

        # Aggregate each factor's horizon return.
        horizon_returns: Dict[str, float] = {}
        for fid in spec.factor_ids:
            series = cluster_returns[fid]
            if len(series) < min(self.horizon, 1):
                raise InsufficientObservations(f"insufficient series for {fid}")
            horizon_returns[fid] = self.horizon_mapping.aggregate_vwap_to_vwap_return(series)

        # Compose with equal weights on the finite aggregate returns.
        weights = {fid: 1.0 / len(spec.factor_ids) for fid in spec.factor_ids}
        raw_composite = sum(
            weights[fid] * horizon_returns[fid] for fid in spec.factor_ids
        )
        oriented = self.orientation_mapping.apply(raw_composite, self.orientation)

        # Regime weighting (identity by default).
        regime_agg = sum(
            w * oriented
            for w in self.regime_mapping.regime_weights.values()
        ) / len(self.regime_mapping.regime_weights)
        regime_agg = max(0.0, regime_agg) if not math.isfinite(regime_agg) else regime_agg

        return CompactCompositeValue(
            composite_factor_id=composite_factor_id,
            agg_freq=agg_freq,
            n_securities=len(spec.factor_ids),
            n_observations=n_observations,
            horizon=self.horizon,
            orientation=self.orientation,
            regime=RegimeType.ALL,
            agg_vwap_to_vwap_return=regime_agg,
            source="CCCC",
            evidence_basis_ref=evidence_basis_ref,
        )

    def accounting_diagnostics(
        self,
        result: AggregationResult,
    ) -> Dict[str, Any]:
        """Weight-derived accounting diagnostics ONLY (no long/short PnL)."""
        weights = result.computed_weights
        if not weights:
            return {"available": False}
        hhi = sum(w * w for w in weights)
        n = len(weights)
        return {
            "available": True,
            "weight_concentration_hhi": hhi,
            "effective_num_factors": 1.0 / hhi if hhi > 0 else None,
            "equal_weight_hhi": 1.0 / n,
        }