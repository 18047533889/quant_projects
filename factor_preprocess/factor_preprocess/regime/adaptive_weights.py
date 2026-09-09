"""
Regime-dependent factor weighting.

Fits regime-specific factor weights and applies them causally based on
detected regime state. All operations preserve fold-safety.

Production integrity (FP-P0):
- FP-P0-01: the learned state is serializable into the canonical
  ``FittedState.learned_params`` typed payload via
  :func:`serialize_regime_weights` / :func:`deserialize_regime_weights`.
- FP-P0-02: ``method="sharpe"`` was a supervised aggregation
  (correlation(factor,target)/std) mislabelled as a Sharpe ratio. It is
  renamed to its true semantics ``target_corr_over_vol`` and gated as
  ``TransformKind.SUPERVISED_FITTED`` / admission ``RESEARCH_ONLY``.
- FP-P0-03: an unseen regime label must NOT silently pass through the
  raw factors. The default is :data:`UnknownRegimePolicy.FAIL`; the
  silent identity path is research-only.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Union, Mapping
from dataclasses import dataclass, field
import copy
from enum import Enum
from types import MappingProxyType

from factor_preprocess.errors import (
    InsufficientObservations,
    MissingFittedStateError,
    StaleFittedStateError,
    UnknownRegimeError,
    SupervisedTargetRequiredError,
    UnsupportedTargetError,
)
from factor_preprocess.contracts.policy import TransformKind

_WEIGHT_METHODS = ("equal", "volatility_inverse", "sharpe", "target_corr_over_vol")


class UnknownRegimePolicy(Enum):
    """How an unknown regime label is handled at apply time (FP-P0-03).

    Production-safe defaults never silently echo the raw factor value.
    """

    FAIL = "FAIL"                      # raise UnknownRegimeError (production default)
    FAIL_NAN = "FAIL_NAN"              # emit NaN for the unknown rows
    FALLBACK_GLOBAL = "FALLBACK_GLOBAL"  # apply fitted global weights
    IDENTITY_RESEARCH_ONLY = "IDENTITY_RESEARCH_ONLY"  # echo input (research only)


def _coerce_unknown_policy(value) -> UnknownRegimePolicy:
    if isinstance(value, UnknownRegimePolicy):
        return value
    if isinstance(value, str):
        try:
            return UnknownRegimePolicy(value.upper())
        except ValueError:
            raise ValueError(
                f"Invalid UnknownRegimePolicy: {value!r}. Valid: "
                f"{[p.value for p in UnknownRegimePolicy]}"
            )
    raise TypeError(
        f"unknown_regime_policy must be an UnknownRegimePolicy or string, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class RegimeWeightState:
    """
    Fitted regime-specific weights (FP-P0-01).

    Attributes
    ----------
    regime_weights : dict[int, np.ndarray]
        Mapping from regime label to factor weights.
        Each array has shape (n_factors,) and sums to 1.0.
    factor_names : list[str]
        Factor names in weight array order.
    n_regimes : int
        Number of regimes fitted.
    fit_window_start : pd.Timestamp
        Start of fitting window (for staleness checking).
    fit_window_end : pd.Timestamp
        End of fitting window.
    fit_method : str
        Canonical fitting method name. ``"sharpe"`` is stored as
        ``"target_corr_over_vol"``.
    target_required : bool
        True when the method requires a target/label series.
    transform_kind : TransformKind
        ``SUPERVISED_FITTED`` for target-driven methods, else ``FITTED``
        (a preprocessing-capable fit).
    admission : str
        ``RESEARCH_ONLY`` for supervised weighting, ``PRODUCTION`` for
        label-free weighting.
    global_weights : np.ndarray | None
        Optional global fallback weights used by
        ``UnknownRegimePolicy.FALLBACK_GLOBAL``.
    """

    regime_weights: Dict[int, np.ndarray]
    factor_names: list[str]
    n_regimes: int
    fit_window_start: pd.Timestamp
    fit_window_end: pd.Timestamp
    fit_method: str = "equal"
    target_required: bool = False
    transform_kind: TransformKind = field(
        default_factory=lambda: TransformKind.ROLLING
    )
    admission: str = "PRODUCTION"
    global_unqualified: Optional[np.ndarray] = None
    fallback_reasons: Mapping[int, str] = field(default_factory=dict)
    effective_counts: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    factor_reasons: Mapping[int, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.transform_kind is None:
            object.__setattr__(self, "transform_kind", TransformKind.ROLLING)
        if self.fit_method == "sharpe":
            object.__setattr__(self, "fit_method", "target_corr_over_vol")
        frozen_weights = {}
        for regime, weights in self.regime_weights.items():
            arr = np.array(weights, dtype=float, copy=True)
            arr.setflags(write=False)
            frozen_weights[int(regime)] = arr
        object.__setattr__(self, "regime_weights", MappingProxyType(frozen_weights))
        object.__setattr__(self, "factor_names", tuple(self.factor_names))
        if self.global_unqualified is not None:
            arr = np.array(self.global_unqualified, dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, "global_unqualified", arr)
        object.__setattr__(self, "fallback_reasons", MappingProxyType(dict(self.fallback_reasons)))
        object.__setattr__(self, "effective_counts", MappingProxyType({int(k): tuple(int(v) for v in values) for k, values in self.effective_counts.items()}))
        object.__setattr__(self, "factor_reasons", MappingProxyType({int(k): tuple(str(v) for v in values) for k, values in self.factor_reasons.items()}))


def serialize_regime_weights(state: RegimeWeightState) -> dict:
    """
    Serialize a RegimeWeightState into the canonical FittedState typed payload.

    The payload is a plain mapping of content-stable values so it can be stored
    in ``FittedState.learned_params`` and content-hashed deterministically
    (FP-P0-01).
    """
    return {
        "version": 2,
        "regime_weights": {int(r): np.asarray(w, dtype=float).tolist() for r, w in state.regime_weights.items()},
        "factor_names": list(state.factor_names),
        "n_regimes": int(state.n_regimes),
        "fit_window_start": state.fit_window_start.isoformat(),
        "fit_window_end": state.fit_window_end.isoformat(),
        "fit_method": state.fit_method,
        "target_required": bool(state.target_required),
        "transform_kind": state.transform_kind.value,
        "admission": state.admission,
        "global_unqualified": None if state.global_unqualified is None else np.asarray(state.global_unqualified, dtype=float).tolist(),
        "fallback_reasons": {str(k): v for k, v in state.fallback_reasons.items()},
        "effective_counts": {str(k): list(v) for k, v in state.effective_counts.items()},
        "factor_reasons": {str(k): list(v) for k, v in state.factor_reasons.items()},
    }


def deserialize_regime_weights(payload: dict) -> RegimeWeightState:
    """Rebuild a RegimeWeightState from a FittedState payload (FP-P0-01)."""
    if payload.get("version") != 2:
        raise ValueError(f"Unsupported regime weight payload version: {payload.get('version')}")
    required={"regime_weights","factor_names","n_regimes","fit_window_start","fit_window_end","fit_method","target_required","transform_kind","admission","global_unqualified","fallback_reasons"}
    missing=required-set(payload)
    if missing: raise ValueError(f"Incomplete regime weight payload: {sorted(missing)}")
    if payload["target_required"] and payload["admission"] != "RESEARCH_ONLY":
        raise ValueError("supervised state cannot be restored as production")
    return RegimeWeightState(
        regime_weights={int(k): np.array(v, dtype=float, copy=True) for k, v in payload["regime_weights"].items()},
        factor_names=list(payload["factor_names"]),
        n_regimes=int(payload["n_regimes"]),
        fit_window_start=pd.Timestamp(payload["fit_window_start"]),
        fit_window_end=pd.Timestamp(payload["fit_window_end"]),
        fit_method=str(payload.get("fit_method", "equal")),
        target_required=bool(payload.get("target_required", False)),
        transform_kind=TransformKind(payload.get("transform_kind", "rolling")),
        admission=str(payload.get("admission", "PRODUCTION")),
        global_unqualified=None if payload.get("global_unqualified") is None else np.array(payload["global_unqualified"], dtype=float, copy=True),
        fallback_reasons={int(k):str(v) for k,v in payload["fallback_reasons"].items()},
        effective_counts={int(k):tuple(v) for k,v in payload.get("effective_counts", {}).items()},
        factor_reasons={int(k):tuple(v) for k,v in payload.get("factor_reasons", {}).items()},
    )


def fit_regime_weights(
    factors: pd.DataFrame,
    regime_labels: pd.Series,
    target: Optional[pd.Series] = None,
    method: Literal["equal", "volatility_inverse", "sharpe", "target_corr_over_vol"] = "equal",
    time_col: str = "date",
    factor_cols: Optional[list[str]] = None,
    min_obs_per_regime: int = 30,
) -> RegimeWeightState:
    """
    Fit regime-specific factor weights on a training window.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor values with time_col and factor columns.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with factors.index.
    target : pd.Series, optional
        Target variable for supervised weighting.
        Required if method="target_corr_over_vol".
    method : str
        Weighting method:
        - "equal": 1/n_factors per factor (label-free)
        - "volatility_inverse": inverse volatility weighting (label-free)
        - "sharpe": deprecated alias for "target_corr_over_vol"
        - "target_corr_over_vol": supervised aggregation
          correlation(factor, target)/vol(factor) -- NOT a Sharpe ratio.
    time_col : str
        Time column
    factor_cols : list[str], optional
        Factor columns to weight. If None, uses all numeric columns except time_col.
    min_obs_per_regime : int
        Minimum observations per regime. Raises if any regime has fewer.

    Returns
    -------
    RegimeWeightState
        Fitted weights per regime.

    Raises
    ------
    InsufficientObservations
        If any regime has fewer than min_obs_per_regime observations.
    SupervisedTargetRequiredError
        If method="target_corr_over_vol" without a target (FP-P0-02).
    UnsupportedTargetError
        If a label-free method receives a target (FP-P0-02).

    Notes
    -----
    - The legacy name "sharpe" is NOT a real Sharpe ratio; it is
      correlation/vol supervised weighting (FP-P0-02). It is gated as
      SUPERVISED_FITTED / RESEARCH_ONLY so a plain production preprocess
      policy cannot silently use it.
    """
    if not factors.columns.is_unique:
        raise ValueError("duplicate factor names are not allowed")
    if not factors.index.is_unique or not regime_labels.index.is_unique:
        raise ValueError("duplicate sample keys are not allowed")
    if not regime_labels.index.equals(factors.index):
        raise ValueError("regime_labels index/order must exactly match factors.index")
    labels = regime_labels.dropna()
    if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or float(v) != int(v) for v in labels):
        raise ValueError("regime labels must be finite non-boolean integers")
    if target is not None and not target.index.equals(factors.index):
        raise ValueError("target index/order must exactly match factors.index")
    if method not in ["equal", "volatility_inverse", "sharpe", "target_corr_over_vol"]:
        raise ValueError(f"Unknown method: {method}")

    canonical_method = "target_corr_over_vol" if method == "sharpe" else method
    supervised = canonical_method == "target_corr_over_vol"

    if supervised and target is None:
        raise SupervisedTargetRequiredError(
            "target is required for method='target_corr_over_vol' (legacy "
            "'sharpe') — this is a supervised factor aggregation, not "
            "label-free preprocessing"
        )
    if not supervised and target is not None:
        raise UnsupportedTargetError(
            f"method='{method}' is label-free and does not accept a target"
        )

    # Verify sort order
    if not factors[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Select factor columns
    if factor_cols is None:
        factor_cols = [
            c for c in factors.columns
            if c != time_col and pd.api.types.is_numeric_dtype(factors[c])
        ]

    if len(factor_cols) == 0:
        raise ValueError("No factor columns found")

    # Extract time window
    fit_window_start = factors[time_col].min()
    fit_window_end = factors[time_col].max()

    # Get unique regimes
    unique_regimes = sorted([r for r in regime_labels.dropna().unique() if not np.isnan(r)])
    n_regimes = len(unique_regimes)

    if n_regimes == 0:
        raise InsufficientObservations("No valid regime labels found")

    # Fit weights per regime
    regime_weights = {}
    fallback_reasons = {}
    effective_counts = {}
    factor_reasons = {}

    for regime in unique_regimes:
        mask = (regime_labels == regime) & regime_labels.notna()
        regime_factors = factors.loc[mask, factor_cols].values

        finite_rows = np.isfinite(regime_factors).all(axis=1)
        n_obs = int(finite_rows.sum())
        counts = np.isfinite(regime_factors).sum(axis=0).astype(int)
        effective_counts[int(regime)] = tuple(counts.tolist())
        if n_obs < min_obs_per_regime:
            raise InsufficientObservations(
                f"Regime {regime} has only {n_obs} observations, "
                f"need at least {min_obs_per_regime}"
            )

        if canonical_method == "equal":
            weights = np.ones(len(factor_cols)) / len(factor_cols)

        elif canonical_method == "volatility_inverse":
            stds = np.nanstd(regime_factors, axis=0, ddof=1)
            valid_factor = (counts >= 2) & np.isfinite(stds) & (stds > 0)
            factor_reasons[int(regime)] = tuple(
                "ESTIMABLE" if valid else ("INSUFFICIENT_EFFECTIVE_OBSERVATIONS" if count < 2 else "ZERO_OR_NONFINITE_VOLATILITY")
                for valid, count in zip(valid_factor, counts)
            )
            if not valid_factor.any(): raise InsufficientObservations(f"Regime {regime} has no estimable factor volatility")
            stds = np.where(valid_factor, stds, np.inf)
            weights = 1.0 / stds
            weights = weights / np.sum(weights)

        else:  # target_corr_over_vol (supervised)
            regime_target = target.loc[mask].values
            if len(regime_target) != len(regime_factors):
                raise ValueError("Target and factors must be aligned")

            correlations = np.full(len(factor_cols), np.nan)
            stds = np.full(len(factor_cols), np.nan)

            for i in range(len(factor_cols)):
                factor_vals = regime_factors[:, i]
                valid_mask = np.isfinite(factor_vals) & np.isfinite(regime_target)

                if np.sum(valid_mask) >= 2:
                    correlations[i] = np.corrcoef(
                        factor_vals[valid_mask],
                        regime_target[valid_mask]
                    )[0, 1]
                    stds[i] = np.nanstd(factor_vals[valid_mask], ddof=1)

            estimable=np.isfinite(correlations)&np.isfinite(stds)&(stds>0)
            if not estimable.any(): raise InsufficientObservations(f"Regime {regime} has no estimable supervised factor")
            correlations=np.where(estimable,correlations,0.0); stds=np.where(estimable,stds,np.inf)
            weights = correlations / stds

            # Clip negative weights to zero
            weights = np.maximum(weights, 0.0)

            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Explicit non-supervised fallback: no learned positive signal.
                weights = np.ones(len(factor_cols)) / len(factor_cols)
                fallback_reasons[int(regime)] = "NO_POSITIVE_SUPERVISED_EVIDENCE_EQUAL_WEIGHT_FALLBACK"

        regime_weights[int(regime)] = weights

    # A supervised method is not a plain label-free preprocessing transform.
    transform_kind = (
        TransformKind.SUPERVISED_FITTED if supervised else TransformKind.ROLLING
    )
    admission = "RESEARCH_ONLY" if supervised else "PRODUCTION"

    # Global fallback weights: mean weight vector across fitted regimes.
    global_unqualified = None
    if len(regime_weights) > 0:
        stacked = np.vstack([regime_weights[r] for r in sorted(regime_weights)])
        global_unqualified = stacked.mean(axis=0)

    return RegimeWeightState(
        regime_weights=regime_weights,
        factor_names=factor_cols,
        n_regimes=n_regimes,
        fit_window_start=fit_window_start,
        fit_window_end=fit_window_end,
        fit_method=canonical_method,
        target_required=supervised,
        transform_kind=transform_kind,
        admission=admission,
        global_unqualified=global_unqualified,
        fallback_reasons=fallback_reasons,
        effective_counts=effective_counts,
        factor_reasons=factor_reasons,
    )


def regime_adaptive_weights(
    factors: pd.DataFrame,
    regime_labels: pd.Series,
    fitted_state: RegimeWeightState,
    time_col: str = "date",
    factor_cols: Optional[list[str]] = None,
    check_staleness: bool = True,
    unknown_regime_policy: Union[str, UnknownRegimePolicy] = UnknownRegimePolicy.FAIL,
    fallback_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Apply regime-specific weights to factors.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor values with time_col and factor columns.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with factors.index.
        NaN regime labels produce NaN weighted factors.
    fitted_state : RegimeWeightState
        Fitted weights from fit_regime_weights.
    time_col : str
        Time column
    factor_cols : list[str], optional
        Factor columns to weight. If None, uses fitted_state.factor_names.
    check_staleness : bool
        If True, raises if factors contain dates <= fitted_state.fit_window_end.
    unknown_regime_policy : str | UnknownRegimePolicy
        How to handle a regime label not seen at fit time (FP-P0-03).
        Default ``FAIL`` raises :class:`UnknownRegimeError`; the silent
        identity path is research-only.
    fallback_weights : np.ndarray, optional
        Weights used when policy is ``FALLBACK_GLOBAL``. If None, uses the
        state's fitted global weights.

    Returns
    -------
    pd.DataFrame
        Weighted factors with columns [time_col] + factor_cols.
        Each factor is multiplied by its regime-specific weight.

    Raises
    ------
    MissingFittedStateError
        If fitted_state is None.
    StaleFittedStateError
        If check_staleness=True and factors overlap with fit window.
    UnknownRegimeError
        If unknown_regime_policy is ``FAIL`` and an unknown regime is seen.
    """
    if not isinstance(regime_labels, pd.Series) and hasattr(regime_labels, "scope"):
        if (getattr(regime_labels,"scope",None)!="FILTERED_ASOF" or
                not getattr(regime_labels,"production_eligible",False) or
                getattr(regime_labels,"fit_end",None) is None or
                getattr(regime_labels,"decision_time",None) is None):
            raise ValueError("production regime weighting requires bound FILTERED_ASOF inference")
        regime_labels=pd.Series(np.asarray(regime_labels.states),index=factors.index)
    if fitted_state is None:
        raise MissingFittedStateError("fitted_state is required")

    policy = _coerce_unknown_policy(unknown_regime_policy)

    # Verify sort order
    if not factors[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Check staleness
    if check_staleness:
        min_date = factors[time_col].min()
        if min_date <= fitted_state.fit_window_end:
            raise StaleFittedStateError(
                f"Factors start at {min_date}, but fit window ends at "
                f"{fitted_state.fit_window_end}. This violates fold-safety."
            )

    # Select factor columns
    if factor_cols is None:
        factor_cols = fitted_state.factor_names

    if set(factor_cols) != set(fitted_state.factor_names):
        raise ValueError(
            f"factor_cols {factor_cols} does not match fitted factor names "
            f"{fitted_state.factor_names}"
        )

    factor_cols_ordered = list(fitted_state.factor_names)

    if not regime_labels.index.equals(factors.index):
        raise ValueError("regime_labels index/order must exactly match factors.index")
    factor_matrix = factors[factor_cols_ordered].to_numpy(dtype=float)
    weighted_matrix = np.full(factor_matrix.shape, np.nan, dtype=float)

    known_regimes = set(fitted_state.regime_weights.keys())

    if policy == UnknownRegimePolicy.FALLBACK_GLOBAL:
        if fallback_weights is None:
            fallback_weights = fitted_state.global_unqualified
        if fallback_weights is None or len(fallback_weights) != len(factor_cols_ordered):
            raise ValueError(
                "FALLBACK_GLOBAL requires fallback_weights with length "
                f"{len(factor_cols_ordered)}, got "
                f"{None if fallback_weights is None else len(fallback_weights)}"
            )
        fallback_weights = np.asarray(fallback_weights, dtype=float)

    for i in range(len(factors)):
        regime = regime_labels.iloc[i]

        if pd.isna(regime):
            # NaN regime -> NaN output
            continue

        if isinstance(regime,(bool,np.bool_)) or not np.isfinite(regime) or float(regime)!=int(regime):
            raise ValueError(f"invalid regime label {regime!r}")
        regime_int = int(regime)

        if regime_int in known_regimes:
            weights = fitted_state.regime_weights[regime_int]
            weighted_matrix[i, :] = factor_matrix[i, :] * weights
            continue

        # Unknown regime handling (FP-P0-03) -- never default to silent echo.
        if policy == UnknownRegimePolicy.FAIL:
            raise UnknownRegimeError(
                f"Regime label {regime_int} was not seen at fit time "
                f"(known: {sorted(known_regimes)}). "
                f"Refusing to silently pass through. Set an explicit "
                f"unknown_regime_policy (FAIL_NAN / FALLBACK_GLOBAL) to proceed."
            )
        elif policy == UnknownRegimePolicy.FAIL_NAN:
            continue  # weighted_matrix already NaN
        elif policy == UnknownRegimePolicy.FALLBACK_GLOBAL:
            weighted_matrix[i, :] = factor_matrix[i, :] * fallback_weights
        elif policy == UnknownRegimePolicy.IDENTITY_RESEARCH_ONLY:
            weighted_matrix[i, :] = factor_matrix[i, :]
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled policy {policy}")

    result = factors[[time_col]].copy()
    for j, col in enumerate(factor_cols_ordered):
        result[col] = weighted_matrix[:, j]

    return result
