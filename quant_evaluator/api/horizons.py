"""Public predictive-IC evaluation across explicit forward-label horizons.

This module also hosts the multi-horizon convenience layer:

- :func:`build_forward_return_label_bundles` builds compound forward-return
  ``LabelBundle`` panels from a (T, N) single-period return panel;
- :func:`summarize_horizons` condenses a :class:`HorizonEvaluationBundle`
  into per-horizon IC summary statistics;
- :func:`evaluate_factor_multi_horizon` chains both with
  :func:`evaluate_horizons` in a single call.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex, stable_hash
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact
from quant_evaluator.contracts.statistical_evidence import HorizonCurveEvidence


def _frozen_mask(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    frozen = np.array(array, copy=True)
    frozen.flags.writeable = False
    return frozen


@dataclass(frozen=True)
class HorizonEvaluationBundle:
    """Immutable multi-horizon result retaining every daily IC and sample mask."""

    factor_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    sample_policy: str
    as_of: Any
    daily_ic_artifacts: Mapping[int, SeriesMetricArtifact]
    curve_evidence: Mapping[str, HorizonCurveEvidence]
    maturity_masks: Mapping[int, np.ndarray]
    evaluation_masks: Mapping[int, np.ndarray]
    common_mask: np.ndarray
    maturity_counts: Mapping[int, int]
    sample_counts: Mapping[int, int]
    label_content_refs: Mapping[int, str]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ic_method: str = "spearman"

    def __post_init__(self) -> None:
        factors, horizons = tuple(self.factor_ids), tuple(self.horizons)
        if not factors or not horizons:
            raise ValueError("factor_ids and horizons cannot be empty")
        if self.ic_method not in {"spearman", "pearson"}:
            raise ValueError("ic_method must be spearman or pearson")
        if set(self.daily_ic_artifacts) != set(horizons):
            raise ValueError("daily IC artifacts must cover every horizon")
        first = self.daily_ic_artifacts[horizons[0]]
        t = first.values.shape[0]
        n = np.asarray(self.common_mask).shape[1]
        artifacts = {}
        for horizon in horizons:
            artifact = self.daily_ic_artifacts[horizon]
            if not isinstance(artifact, SeriesMetricArtifact) or artifact.values.shape != (t, len(factors)):
                raise ValueError("every horizon requires a full aligned SeriesMetricArtifact")
            artifacts[horizon] = artifact
        maturity = {h: _frozen_mask(self.maturity_masks[h], (t,), f"maturity_masks[{h}]")
                    for h in horizons}
        evaluation = {h: _frozen_mask(self.evaluation_masks[h], (t, n), f"evaluation_masks[{h}]")
                      for h in horizons}
        object.__setattr__(self, "factor_ids", factors)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "daily_ic_artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "curve_evidence", MappingProxyType(dict(self.curve_evidence)))
        object.__setattr__(self, "maturity_masks", MappingProxyType(maturity))
        object.__setattr__(self, "evaluation_masks", MappingProxyType(evaluation))
        object.__setattr__(self, "common_mask", _frozen_mask(self.common_mask, (t, n), "common_mask"))
        object.__setattr__(self, "maturity_counts", MappingProxyType(dict(self.maturity_counts)))
        object.__setattr__(self, "sample_counts", MappingProxyType(dict(self.sample_counts)))
        object.__setattr__(self, "label_content_refs", MappingProxyType(dict(self.label_content_refs)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _same_axis(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object))


def evaluate_horizons(
    factors: FactorBatch,
    labels: Mapping[int, LabelBundle],
    *,
    as_of: Any,
    sample_policy: str = "common",
    min_assets: int = 10,
    min_periods: int = 20,
    zero_tolerance: float = 1e-12,
    backend: Any = None,
    ic_method: str = "spearman",
) -> HorizonEvaluationBundle:
    """Evaluate predictive IC curves against explicit label bundles.

    The default ic_method is Spearman; Pearson is required for the registered
    ic_decay formula and is recorded in the returned bundle.

    ``common`` intersects label validity and label-time maturity at the T×N
    cell level before every leaf evaluation. ``per_horizon`` preserves each
    label's own eligible sample and marks the curves as non-comparable samples.
    """
    if not isinstance(factors, FactorBatch):
        raise TypeError("factors must be a FactorBatch")
    if ic_method not in {"spearman", "pearson"}:
        raise ValueError("ic_method must be spearman or pearson")
    if sample_policy not in {"common", "per_horizon"}:
        raise ValueError("sample_policy must be 'common' or 'per_horizon'")
    if as_of is None:
        raise ValueError("as_of is required")
    if isinstance(min_assets, bool) or not isinstance(min_assets, int) or min_assets < 2:
        raise ValueError("min_assets must be an integer >= 2")
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be a positive integer")
    if not isinstance(labels, Mapping) or len(labels) < 2:
        raise ValueError("labels must contain at least two horizons")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in labels):
        raise ValueError("horizons must be unique positive integers")
    horizons = tuple(sorted(labels))
    if factors.time_axis.values is None or factors.asset_axis.values is None:
        raise ValueError("factor time and asset axes require explicit coordinates")
    t, n = factors.num_times, factors.num_assets

    decision_axis = observation_axis = asset_axis = None
    own_masks: dict[int, np.ndarray] = {}
    maturity_masks: dict[int, np.ndarray] = {}
    label_refs: dict[int, str] = {}
    for horizon in horizons:
        label = labels[horizon]
        if not isinstance(label, LabelBundle) or label.horizon != horizon:
            raise ValueError(f"label mapping key {horizon} must match LabelBundle.horizon")
        if label.values.shape != (t, n) or label.asset_axis is None or label.asset_axis.values is None:
            raise ValueError("every label must be a bound T×N panel")
        if not label.observation_time:
            raise ValueError("every label requires an explicit observation_time axis")
        if not _same_axis(label.observation_time, factors.time_axis.values):
            raise ValueError("label observation axis differs from factor time axis")
        if not _same_axis(label.asset_axis.values, factors.asset_axis.values):
            raise ValueError("label asset axis differs from factor asset axis")
        if decision_axis is None:
            decision_axis = label.decision_time
            observation_axis = label.observation_time
            asset_axis = label.asset_axis.values
        elif (not _same_axis(label.decision_time, decision_axis)
              or not _same_axis(label.observation_time, observation_axis)
              or not _same_axis(label.asset_axis.values, asset_axis)):
            raise ValueError("all horizon labels must share decision, observation and asset axes")
        try:
            mature = np.asarray([end <= as_of for end in label.label_end_time], dtype=bool)
        except TypeError as exc:
            raise ValueError("as_of and label_end_time must be explicitly comparable") from exc
        validity = np.isfinite(label.values)
        if label.validity is not None:
            validity &= label.validity
        maturity_masks[horizon] = mature
        own_masks[horizon] = validity & mature[:, None]
        label_refs[horizon] = label.content_hash

    common_mask = np.logical_and.reduce([own_masks[h] for h in horizons])
    selected_masks = {
        h: common_mask if sample_policy == "common" else own_masks[h]
        for h in horizons
    }
    artifacts: dict[int, SeriesMetricArtifact] = {}
    from quant_evaluator.runtime.evaluator import evaluate
    series_metric = "pearson_ic_series" if ic_method == "pearson" else "rank_ic_series"
    for horizon in horizons:
        selected_label = replace(labels[horizon], validity=selected_masks[horizon])
        result = evaluate(
            factors, selected_label, metrics=(series_metric,), backend=backend,
            metric_parameters={series_metric: {"min_assets": min_assets}},
        )
        artifacts[horizon] = result.artifacts[series_metric]

    from quant_evaluator.metrics.statistical_evidence import build_horizon_curve_evidence
    curves: dict[str, HorizonCurveEvidence] = {}
    for factor_index, factor_id in enumerate(factors.factor_ids):
        curves[factor_id] = build_horizon_curve_evidence(
            {h: artifacts[h].values[:, factor_index] for h in horizons},
            label_refs=label_refs, min_periods=min_periods,
            zero_tolerance=zero_tolerance,
        )
    return HorizonEvaluationBundle(
        factor_ids=tuple(factors.factor_ids), horizons=horizons,
        sample_policy=sample_policy, as_of=as_of,
        daily_ic_artifacts=artifacts, curve_evidence=curves,
        maturity_masks=maturity_masks, evaluation_masks=selected_masks,
        common_mask=common_mask,
        maturity_counts={h: int(maturity_masks[h].sum()) for h in horizons},
        sample_counts={h: int(selected_masks[h].sum()) for h in horizons},
        label_content_refs=label_refs,
        ic_method=ic_method,
        provenance={
            "ic_method": ic_method,
            "sample_policy": sample_policy,
            "samples_comparable_across_horizons": sample_policy == "common",
            "min_assets": min_assets,
            "min_periods": min_periods,
            "zero_tolerance": curves[factors.factor_ids[0]].zero_tolerance,
            "backend": getattr(backend, "value", backend),
            "factor_value_hash": factors.value_hash,
        },
    )


# ---------------------------------------------------------------------------
# Multi-horizon forward-return labels and summaries
# ---------------------------------------------------------------------------

DEFAULT_MULTI_HORIZONS = (1, 3, 5, 7, 10, 20)

_BUILDER_KWARGS = frozenset({"time_axis_values", "asset_axis_values", "validity", "execution_delay"})
_SUMMARY_KWARGS = frozenset({"icir_min_periods", "hac", "hac_max_lag", "annualization_periods", "per_horizon_portfolio_metrics"})
_EVALUATE_KWARGS = frozenset({"as_of", "sample_policy", "min_assets", "min_periods", "zero_tolerance", "backend", "ic_method"})

_ROW_STAT_FIELDS = (
    "mean_ic", "ic_std", "rank_icir", "annualized_ir", "positive_ic_ratio",
    "sample_counts", "t_stat", "p_value", "t_stat_hac", "se_hac",
)


def _validated_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)) or isinstance(horizons, (bool, int, np.integer)):
        raise ValueError("horizons must be a sequence of unique positive integers")
    seen: set[int] = set()
    for value in horizons:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError("horizons must be unique positive integers")
        horizon = int(value)
        if horizon in seen:
            raise ValueError("horizons must be unique positive integers")
        seen.add(horizon)
    if not seen:
        raise ValueError("horizons must not be empty")
    return tuple(sorted(seen))


def _validated_return_panel(daily_returns: Any) -> np.ndarray:
    panel = np.asarray(daily_returns)
    if panel.ndim != 2:
        raise ValueError("daily_returns must be a (T, N) single-period return panel")
    kind = panel.dtype.kind
    if kind == "b":
        raise ValueError("daily_returns must be a real floating panel; boolean returns are rejected")
    if kind == "c":
        raise ValueError("daily_returns must be a real floating panel; complex returns are rejected")
    if kind == "O":
        raise ValueError("daily_returns must be a real floating panel; object panels are rejected")
    if kind != "f":
        # Integer (and other non-float) panels are rejected fail-closed:
        # single-period returns are fractional, and an integer panel almost
        # always signals a units bug (e.g. price levels passed as returns).
        raise ValueError(
            "daily_returns must be a floating panel; integer return panels are "
            "rejected because single-period returns are fractional and an integer "
            "panel almost always signals a units bug. Cast explicitly with "
            ".astype(np.float64) if the values are truly fractional."
        )
    if panel.dtype != np.float64:
        panel = panel.astype(np.float64)
    return panel


def _time_value_at(values: tuple, index: int) -> Any:
    """Return ``values[index]``, synthesizing strictly-increasing tail
    coordinates by linear extrapolation beyond the axis end.

    Immature label rows (window beyond the panel end) still need valid
    ``label_end_time`` coordinates (``label_start_time < label_end_time`` is
    enforced by :class:`LabelBundle`); the synthesized value only exists to
    keep the causal chain orderable and is always paired with a NaN label, so
    it can never contribute an evaluated sample.
    """
    size = len(values)
    if 0 <= index < size:
        return values[index]
    if size < 2:
        raise ValueError(
            "time_axis_values needs at least two coordinates to synthesize "
            "label_end_time for immature rows"
        )
    try:
        step = values[-1] - values[-2]
        if step == 0:
            raise ValueError("time_axis_values must be strictly increasing")
        return values[-1] + step * (index - (size - 1))
    except TypeError as exc:
        raise ValueError(
            "immature-tail label_end_time coordinates require numeric or "
            "datetime-like time axis values supporting subtraction/addition"
        ) from exc


def build_forward_return_label_bundles(
    daily_returns: Any,
    horizons: Iterable[int] = DEFAULT_MULTI_HORIZONS,
    *,
    time_axis_values: Any,
    asset_axis_values: Any,
    validity: Any = None,
    execution_delay: int = 1,
) -> dict[int, LabelBundle]:
    """Build compound forward-return label bundles for arbitrary horizons.

    Formula (fixed): for decision index ``t`` and horizon ``H``,

        label_H(t) = prod_{k=1..H} (1 + r(t+k)) - 1

    where ``r(t)`` is the single-period return spanning ``t -> t+1`` supplied
    by the caller (close-to-close or vwap-to-vwap, provenance only).  The
    label window is therefore ``[t+1, t+1+H]``: strictly causal with a
    one-period execution delay embedded by construction.  ``label_H`` uses
    only returns inside its own window — no data outside ``[t+1, t+1+H]``.

    NaN semantics (fail-closed, never silently zero-filled):

    - any non-finite (NaN or inf) single-period return inside the window
      makes the whole label NaN;
    - a window extending past the panel end (``t+1+H > T-1``) makes the
      label NaN (insufficient maturity);
    - an explicitly invalid cell in ``validity`` (boolean (T, N) mask over
      the *single-period* return panel) is treated exactly like a NaN return
      and propagates to every label whose window covers it.

    For ``H == 1`` the label is the exact single-period return ``r(t+1)``
    (mathematically identical to the compound formula, without the
    ``(1+r)-1`` floating-point round-trip).

    ``execution_delay`` is recorded provenance (the panel alignment above
    already embeds a one-period delay); it is validated and stored on each
    :class:`LabelBundle` but does not re-index the window.

    Timing axes: ``decision_time``/``observation_time`` are
    ``time_axis_values`` verbatim (must match the factor time axis for
    :func:`evaluate_horizons`).  ``label_start_time``/``execution_time`` are
    ``time_axis_values[i+1]`` and ``label_end_time`` is
    ``time_axis_values[i+1+H]``; for immature rows the tail coordinates are
    synthesized by linear extrapolation of the last axis step (always paired
    with a NaN label, so they can never enter an evaluated sample).

    Requires at least two coordinates on ``time_axis_values`` and a strictly
    increasing axis (enforced by :class:`LabelBundle` causality checks).
    Returns ``{H: LabelBundle}`` sorted by horizon with
    ``LabelBundle.horizon == H`` for every key.  The builder accepts a single
    horizon; :func:`evaluate_horizons` itself requires at least two.
    """
    returns = _validated_return_panel(daily_returns)
    t, n = returns.shape
    horizons = _validated_horizons(horizons)
    times = tuple(time_axis_values)
    if len(times) != t:
        raise ValueError(f"time_axis_values must have {t} coordinates, got {len(times)}")
    assets = np.asarray(asset_axis_values)
    if assets.ndim != 1 or assets.shape[0] != n:
        raise ValueError(f"asset_axis_values must be a 1-D sequence of {n} coordinates")
    if isinstance(execution_delay, bool) or not isinstance(execution_delay, (int, np.integer)) or execution_delay < 0:
        raise ValueError("execution_delay must be a nonnegative integer")
    if validity is not None:
        mask = np.asarray(validity)
        if mask.shape != (t, n) or mask.dtype.kind != "b":
            raise ValueError("validity must be a boolean (T, N) mask over daily_returns")
        returns = np.where(mask, returns, np.nan)
    # NaN and +-inf single-period returns propagate as NaN, never as 0.
    returns = np.where(np.isfinite(returns), returns, np.nan)

    decision = times
    observation = times
    start = tuple(_time_value_at(times, i + 1) for i in range(t))
    execution = start
    asset_axis = AxisRef(
        "asset",
        "str" if assets.dtype.kind in "US" else str(assets.dtype),
        n,
        assets,
    )
    bundles: dict[int, LabelBundle] = {}
    for horizon in horizons:
        labels = np.full((t, n), np.nan, dtype=np.float64)
        if horizon == 1:
            if t > 1:
                labels[:-1] = returns[1:]
        else:
            growth = np.ones((t, n), dtype=np.float64)
            for k in range(1, horizon + 1):
                shifted = np.full((t, n), np.nan, dtype=np.float64)
                if k < t:
                    shifted[: t - k] = returns[k:]
                growth = growth * (1.0 + shifted)
            labels = growth - 1.0
        end = tuple(_time_value_at(times, i + 1 + horizon) for i in range(t))
        bundles[horizon] = LabelBundle(
            target_id=f"forward_return_h{horizon}",
            values=labels,
            horizon=horizon,
            execution_delay=int(execution_delay),
            decision_time=decision,
            observation_time=observation,
            execution_time=execution,
            label_start_time=start,
            label_end_time=end,
            asset_axis=asset_axis,
            source_ref=f"forward_return_labels:h{horizon}",
            metadata={
                "formula": "label_H(t) = prod_{k=1..H} (1 + r(t+k)) - 1",
                "label_window": "[t+1, t+1+H]",
                "single_period_return_span": "r(t) spans t -> t+1",
                "execution_delay_periods": int(execution_delay),
                "validity_mask_applied": validity is not None,
            },
        )
    return bundles


def _frozen_stat_array(value: Any, name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class HorizonSummaryRow:
    """Per-horizon IC summary statistics; every array has shape (F,).

    Conventions (aligned with ``quant_evaluator.metrics.ic_summary``):

    - ``mean_ic`` / ``ic_std``: finite-value mean and sample std (ddof=1) of
      the daily IC series; NaN when fewer than 1 / 2 finite values exist.
    - ``rank_icir``: computed by ``compute_icir`` — mean / std(ddof=1) over
      finite values, NaN when fewer than ``icir_min_periods`` finite values
      or exactly zero dispersion.
    - ``annualized_ir``: ``rank_icir * sqrt(annualization_periods)``
      (default sqrt(252)); assumes equally spaced 252 trading days per year
      and that the daily IC series is on that grid.
    - ``positive_ic_ratio``: fraction of finite daily IC values > 0.
    - ``sample_counts``: number of finite daily IC values (int64).
    - ``t_stat`` / ``p_value``: two-tailed one-sample t-test of
      H0: mean(IC)=0, computed by ``compute_ic_tstat``.
    - ``t_stat_hac`` / ``se_hac``: HAC (Newey-West, Bartlett kernel)
      robust t-statistic and standard error from
      ``quant_evaluator.metrics.robustness.compute_hac_tstat`` (the single
      library HAC kernel; no second implementation), only when requested.
    """

    mean_ic: np.ndarray
    ic_std: np.ndarray
    rank_icir: np.ndarray
    annualized_ir: np.ndarray
    positive_ic_ratio: np.ndarray
    sample_counts: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    t_stat_hac: Optional[np.ndarray] = None
    se_hac: Optional[np.ndarray] = None


@dataclass(frozen=True, eq=False)
class MultiHorizonSummary:
    """Frozen, content-hashed multi-horizon IC summary.

    ``rows`` maps each horizon to a :class:`HorizonSummaryRow`;
    ``cumulative_ic[h]`` is the (T, F) cumulative-IC curve: the running sum
    of the daily IC values where NaN days are skipped (``nancumsum``
    semantics — a NaN day contributes nothing and does not reset the curve),
    matching the NaN handling of ``metrics.ic_summary``.

    ``per_horizon_portfolio_metrics`` is an optional caller-injected carrier
    (never computed here): ``{horizon: {metric_id: float}}`` for per-horizon
    long-short portfolio metrics.  See :func:`evaluate_horizons` /
    module docstring for the documented recipe using the existing
    ``evaluate`` facade with ``max_underwater_duration``,
    ``mean_underwater_duration`` and ``time_to_recovery``.

    Equality and hashing are content-based (``content_hash`` is a canonical
    SHA-256 digest over every payload array, stable across processes).
    """

    horizons: tuple[int, ...]
    factor_ids: tuple[str, ...]
    rows: Mapping[int, HorizonSummaryRow]
    cumulative_ic: Mapping[int, np.ndarray]
    annualization_periods: int
    icir_min_periods: int
    per_horizon_portfolio_metrics: Optional[Mapping[int, Mapping[str, float]]] = None
    content_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        horizons = tuple(self.horizons)
        factor_ids = tuple(self.factor_ids)
        if not horizons or not factor_ids:
            raise ValueError("horizons and factor_ids cannot be empty")
        rows = dict(self.rows)
        if set(rows) != set(horizons):
            raise ValueError("rows must cover exactly every horizon")
        frozen_rows: dict[int, HorizonSummaryRow] = {}
        for horizon in horizons:
            row = rows[horizon]
            if not isinstance(row, HorizonSummaryRow):
                raise ValueError(f"rows[{horizon}] must be a HorizonSummaryRow")
            frozen_rows[horizon] = HorizonSummaryRow(**{
                name: self._freeze_row_field(getattr(row, name), name)
                for name in _ROW_STAT_FIELDS
            })
        cumulative = {}
        for horizon in horizons:
            curve = np.array(self.cumulative_ic[horizon], copy=True)
            if curve.ndim != 2 or curve.shape[1] != len(factor_ids):
                raise ValueError(f"cumulative_ic[{horizon}] must have shape (T, F)")
            curve.flags.writeable = False
            cumulative[horizon] = curve
        portfolio = None
        if self.per_horizon_portfolio_metrics is not None:
            portfolio = {}
            for key, metrics in dict(self.per_horizon_portfolio_metrics).items():
                horizon = int(key)
                if horizon not in horizons:
                    raise ValueError("per_horizon_portfolio_metrics keys must be evaluated horizons")
                frozen_metrics = {}
                for metric_id, value in dict(metrics).items():
                    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                        raise ValueError("per_horizon_portfolio_metrics values must be real numbers")
                    frozen_metrics[str(metric_id)] = float(value)
                portfolio[horizon] = MappingProxyType(frozen_metrics)
            portfolio = MappingProxyType(portfolio)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "rows", MappingProxyType(frozen_rows))
        object.__setattr__(self, "cumulative_ic", MappingProxyType(cumulative))
        object.__setattr__(self, "per_horizon_portfolio_metrics", portfolio)
        identity = {
            "horizons": list(horizons),
            "factor_ids": list(factor_ids),
            "annualization_periods": self.annualization_periods,
            "icir_min_periods": self.icir_min_periods,
            "rows": {
                str(horizon): {name: getattr(frozen_rows[horizon], name) for name in _ROW_STAT_FIELDS}
                for horizon in horizons
            },
            "cumulative_ic": {str(horizon): cumulative[horizon] for horizon in horizons},
            "per_horizon_portfolio_metrics": (
                {str(k): v for k, v in portfolio.items()}
                if portfolio is not None else None
            ),
        }
        object.__setattr__(self, "content_hash", stable_content_hex(tag="MultiHorizonSummary.v1", fields=identity))

    @staticmethod
    def _freeze_row_field(value: Any, name: str) -> Any:
        if value is None:
            return None
        return _frozen_stat_array(value, name)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        return stable_hash(self.content_hash)


def summarize_horizons(
    bundle: HorizonEvaluationBundle,
    *,
    icir_min_periods: int = 20,
    hac: bool = False,
    hac_max_lag: int = 5,
    annualization_periods: int = 252,
    per_horizon_portfolio_metrics: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> MultiHorizonSummary:
    """Summarize a :class:`HorizonEvaluationBundle` per horizon.

    All statistics reuse the existing ``metrics.ic_summary`` /
    ``metrics.robustness`` kernels — no second implementation.  See
    :class:`HorizonSummaryRow` for the exact conventions (ddof=1 sample
    statistics over finite daily IC values, sqrt(252) annualization assuming
    equally spaced 252 trading days, nancumsum cumulative-IC curve).

    ``hac=True`` adds HAC (Newey-West, Bartlett kernel, ``hac_max_lag``)
    t-statistics via ``compute_hac_tstat`` (requires at least
    ``hac_max_lag + 10`` contiguous finite IC observations; NaN otherwise).
    """
    if not isinstance(bundle, HorizonEvaluationBundle):
        raise TypeError("bundle must be a HorizonEvaluationBundle")
    if isinstance(icir_min_periods, bool) or not isinstance(icir_min_periods, (int, np.integer)) or icir_min_periods < 2:
        raise ValueError("icir_min_periods must be an integer >= 2 (sample std requires ddof=1)")
    if isinstance(hac_max_lag, bool) or not isinstance(hac_max_lag, (int, np.integer)) or hac_max_lag < 0:
        raise ValueError("hac_max_lag must be a nonnegative integer")
    if isinstance(annualization_periods, bool) or not isinstance(annualization_periods, (int, np.integer)) or annualization_periods <= 0:
        raise ValueError("annualization_periods must be a positive integer")
    if not isinstance(hac, bool):
        raise ValueError("hac must be a boolean")

    from quant_evaluator.metrics.ic_summary import compute_ic_tstat, compute_icir
    from quant_evaluator.metrics.robustness import compute_hac_tstat

    sqrt_periods = float(np.sqrt(float(annualization_periods)))
    rows: dict[int, HorizonSummaryRow] = {}
    cumulative: dict[int, np.ndarray] = {}
    for horizon in bundle.horizons:
        values = np.asarray(bundle.daily_ic_artifacts[horizon].values, dtype=np.float64)
        finite = np.isfinite(values)
        counts = finite.sum(axis=0).astype(np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            safe = np.where(finite, values, 0.0)
            sums = safe.sum(axis=0)
            mean_ic = np.divide(sums, counts, out=np.full(counts.shape, np.nan, dtype=np.float64), where=counts > 0)
            ic_std = np.nanstd(np.where(finite, values, np.nan), axis=0, ddof=1)
        rank_icir = compute_icir(values, min_periods=int(icir_min_periods))
        with np.errstate(invalid="ignore"):
            annualized_ir = rank_icir * sqrt_periods
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            positive = np.divide(
                (np.where(finite, values, 0.0) > 0).sum(axis=0).astype(np.float64),
                counts.astype(np.float64),
                out=np.full(counts.shape, np.nan, dtype=np.float64),
                where=counts > 0,
            )
        t_stat, p_value = compute_ic_tstat(values, min_periods=int(icir_min_periods))
        t_stat_hac = se_hac = None
        if hac:
            t_stat_hac, se_hac = compute_hac_tstat(values, max_lag=int(hac_max_lag))
        rows[horizon] = HorizonSummaryRow(
            mean_ic=mean_ic, ic_std=ic_std, rank_icir=rank_icir,
            annualized_ir=annualized_ir, positive_ic_ratio=positive,
            sample_counts=counts, t_stat=t_stat, p_value=p_value,
            t_stat_hac=t_stat_hac, se_hac=se_hac,
        )
        cumulative[horizon] = np.nancumsum(values, axis=0)
    return MultiHorizonSummary(
        horizons=bundle.horizons,
        factor_ids=bundle.factor_ids,
        rows=rows,
        cumulative_ic=cumulative,
        annualization_periods=int(annualization_periods),
        icir_min_periods=int(icir_min_periods),
        per_horizon_portfolio_metrics=per_horizon_portfolio_metrics,
    )


def evaluate_factor_multi_horizon(
    factors: FactorBatch,
    daily_returns: Any,
    horizons: Iterable[int] = DEFAULT_MULTI_HORIZONS,
    **kwargs: Any,
) -> tuple[HorizonEvaluationBundle, MultiHorizonSummary]:
    """One-call multi-horizon evaluation: labels -> evaluate_horizons -> summary.

    ``horizons`` defaults to ``(1, 3, 5, 7, 10, 20)``.  Keyword arguments are
    routed fail-closed:

    - to :func:`build_forward_return_label_bundles`: ``validity``,
      ``execution_delay``; ``time_axis_values`` / ``asset_axis_values``
      default to the factor batch's own axis coordinates when omitted;
    - to :func:`summarize_horizons`: ``icir_min_periods`` (default: the
      ``min_periods`` passed to :func:`evaluate_horizons`, itself defaulting
      to 20), ``hac``, ``hac_max_lag``, ``annualization_periods``,
      ``per_horizon_portfolio_metrics``;
    - to :func:`evaluate_horizons`: ``as_of`` (required), ``sample_policy``,
      ``min_assets``, ``min_periods``, ``zero_tolerance``, ``backend``.

    Any other keyword raises ``TypeError``.  Returns
    ``(HorizonEvaluationBundle, MultiHorizonSummary)``.  Requires at least
    two horizons (:func:`evaluate_horizons` contract); for a single horizon
    use :func:`build_forward_return_label_bundles` plus ``evaluate``.

    Multi-horizon long-short underwater recipe (documented usage of the
    EXISTING kernels — nothing new is computed here; the results are carried
    structurally via ``per_horizon_portfolio_metrics``)::

        bundle, summary = evaluate_factor_multi_horizon(
            factors, daily_returns, horizons=(1, 5, 10), as_of=as_of)

        from quant_evaluator.runtime.evaluator import evaluate
        portfolio_metrics = {}
        for h in bundle.horizons:
            # Build the per-horizon long-short daily PnL series (T,) for
            # horizon h (e.g. from the factor panel decile spread executed
            # with the same one-period delay), then reuse the existing
            # underwater kernels through the evaluate facade:
            result = evaluate(
                factors, labels[h],
                metrics=("max_underwater_duration",
                         "mean_underwater_duration",
                         "time_to_recovery"),
                portfolio_returns=long_short_pnl_for_horizon_h,
            )
            portfolio_metrics[h] = {
                metric_id: float(result.artifacts[metric_id].values)
                for metric_id in ("max_underwater_duration",
                                  "mean_underwater_duration",
                                  "time_to_recovery")
            }
        summary = summarize_horizons(
            bundle, per_horizon_portfolio_metrics=portfolio_metrics)
    """
    builder_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _BUILDER_KWARGS}
    summary_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _SUMMARY_KWARGS}
    unknown = set(kwargs) - _EVALUATE_KWARGS
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    horizons = _validated_horizons(horizons)
    if len(horizons) < 2:
        raise ValueError(
            "evaluate_factor_multi_horizon requires at least two horizons "
            "(evaluate_horizons contract); use "
            "build_forward_return_label_bundles + evaluate for single-horizon runs"
        )
    time_axis_values = builder_kwargs.pop("time_axis_values", None)
    if time_axis_values is None:
        if factors.time_axis is None or factors.time_axis.values is None:
            raise ValueError("factors require explicit time axis coordinates when time_axis_values is omitted")
        time_axis_values = factors.time_axis.values
    asset_axis_values = builder_kwargs.pop("asset_axis_values", None)
    if asset_axis_values is None:
        if factors.asset_axis is None or factors.asset_axis.values is None:
            raise ValueError("factors require explicit asset axis coordinates when asset_axis_values is omitted")
        asset_axis_values = factors.asset_axis.values
    if "as_of" not in kwargs:
        raise ValueError("as_of is required")
    labels = build_forward_return_label_bundles(
        daily_returns, horizons,
        time_axis_values=time_axis_values,
        asset_axis_values=asset_axis_values,
        **builder_kwargs,
    )
    bundle = evaluate_horizons(factors, labels, **kwargs)
    if summary_kwargs.get("icir_min_periods") is None:
        summary_kwargs["icir_min_periods"] = kwargs.get("min_periods", 20)
    summary = summarize_horizons(bundle, **summary_kwargs)
    return bundle, summary


__all__ = [
    "HorizonEvaluationBundle",
    "HorizonSummaryRow",
    "MultiHorizonSummary",
    "DEFAULT_MULTI_HORIZONS",
    "evaluate_horizons",
    "build_forward_return_label_bundles",
    "summarize_horizons",
    "evaluate_factor_multi_horizon",
]
