"""M02 out-of-sample model delta kernels (QE-EXT-SPEC-1.0, extension_plan.md Sec. 7.2).

True OOS ablation evidence.  SEMANTIC SEPARATION (Sec. 2 / Sec. 7.2): the
existing ``metrics/interactions/conditional.py`` incremental IC is a
*same-day descriptive* decomposition of realised cross-sections; the
kernels here consume upstream *out-of-sample predictions* plus a typed
fit manifest and measure genuine out-of-sample model increments.  Both
are kept; neither may stand in for the other.

Fail-closed inputs (Sec. 1.3 item 5): every kernel requires a typed
:class:`OOSFitManifest`; a missing manifest raises ``TypeError`` and a
manifest whose training window touches the decision window is
INVALID_EVIDENCE (``OOS_WINDOW_LEAKAGE``, strict ``<`` gate, Sec. 7.2
v1).  Strings alone never certify OOS-ness.

Common-sample rule (Sec. 6.1 rule 2): cross-sectional metrics allow
per-day pairwise-finite assets with at least ``min_assets`` valid
assets; every result carries per-day valid-asset counts.  When a
candidate has missing predictions the common support shrinks and the
baseline leg is recomputed on the *same* common support (Sec. 7.2
tests).

New metric IDs start at version ``1.0.0`` (Sec. 3.2).  Registry binding
and facade-builder wiring are NOT done here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from quant_evaluator.metrics.extension.paired_performance import (
    METRIC_VERSION,
    STATUS_INSUFFICIENT,
    STATUS_INVALID,
    STATUS_OK,
    PairedMetricResult,
    paired_net_sharpe_delta,
    real_matrix,
)

__all__ = [
    "OOSFitManifest",
    "OOSDeltaResult",
    "OOSAblationTaskResult",
    "oos_rank_ic_delta",
    "oos_r2_gain",
    "oos_daily_mse_improvement",
    "oos_ablation_summary",
]


# ---------------------------------------------------------------------------
# typed fit manifest (Sec. 7.2 OOS gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OOSFitManifest:
    """Typed, hashable OOS fit manifest (Sec. 7.2 input contract).

    ``max_training_label_end`` must be strictly less than the first
    decision time id (v1 gate, Sec. 7.2).  ``decision_time_ids`` must be
    non-empty and strictly increasing; its length must match the
    prediction time axis.
    """
    model_ref: str
    fit_scope_ref: str
    max_training_label_end: float
    decision_time_ids: tuple

    def __post_init__(self):
        for name in ("model_ref", "fit_scope_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string reference")
        value = self.max_training_label_end
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError("max_training_label_end must be a finite comparable value")
        ids = tuple(self.decision_time_ids)
        if not ids:
            raise ValueError("decision_time_ids must be non-empty")
        try:
            strictly_increasing = all(
                ids[i] < ids[i + 1] for i in range(len(ids) - 1))
        except TypeError:
            strictly_increasing = False
        if not strictly_increasing:
            raise ValueError("decision_time_ids must be strictly increasing and comparable")
        object.__setattr__(self, "decision_time_ids", ids)


def _require_manifest(manifest, role: str) -> Optional[str]:
    """Fail-closed manifest handling.  Returns a leakage reason or None."""
    if manifest is None:
        raise TypeError(
            f"{role} requires a typed OOSFitManifest; a missing manifest is "
            "not OOS evidence (Sec. 1.3 item 5, Sec. 7.2)")
    if not isinstance(manifest, OOSFitManifest):
        raise TypeError(f"{role} must be an OOSFitManifest, got {type(manifest).__name__}")
    if not (manifest.max_training_label_end < manifest.decision_time_ids[0]):
        return "OOS_WINDOW_LEAKAGE"  # v1 strict '<' (Sec. 7.2)
    return None


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OOSDeltaResult:
    """Typed result for one M02 delta metric.

    ``summary`` is the headline scalar over valid days (NaN when none);
    ``daily`` is the (T,) daily series (NaN for invalid days) or None for
    global statistics.  ``diagnostics`` always carries per-day valid
    asset counts when a daily decomposition exists (Sec. 6.1 rule 2).
    """
    metric_id: str
    summary: float
    daily: Optional[np.ndarray]
    status: str
    reason_code: str
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OOSAblationTaskResult:
    """Per-ablation-task outputs (Sec. 7.2 ``oos_ablation_summary``)."""
    task_id: str
    rank_ic_delta: OOSDeltaResult
    daily_mse_improvement: OOSDeltaResult
    r2_gain: OOSDeltaResult
    net_sharpe_delta: Optional[PairedMetricResult]
    economic_status: str


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------

def _oos_inputs(labels, baseline_pred, candidate_pred, validity):
    y = real_matrix(labels, "labels", ndim=2)
    p0 = real_matrix(baseline_pred, "baseline_pred", ndim=2)
    p1 = real_matrix(candidate_pred, "candidate_pred", ndim=2)
    if not (y.shape == p0.shape == p1.shape):
        raise ValueError("labels, baseline_pred and candidate_pred must share one (T, N) shape")
    mask = np.isfinite(y) & np.isfinite(p0) & np.isfinite(p1)
    if validity is not None:
        raw = np.asarray(validity)
        if raw.dtype.kind not in "bfiu" or raw.shape != y.shape:
            raise ValueError("validity must be a real/boolean array shaped like the labels")
        mask = mask & raw.astype(bool)
    return y, p0, p1, mask


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    dx = x - x.mean()
    dy = y - y.mean()
    denom = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if not math.isfinite(denom) or denom <= 0.0:
        return math.nan
    return float((dx * dy).sum() / denom)


def _daily_rank_ic(pred_row: np.ndarray, y_row: np.ndarray,
                   mask_row: np.ndarray, min_assets: int):
    """Spearman (average-tie ranks, Pearson-on-ranks) on one day's common
    support.  Returns ``(ic, n_assets)``; NaN when the support is too
    small or a rank vector is constant."""
    n_assets = int(mask_row.sum())
    if n_assets < min_assets:
        return math.nan, n_assets
    ranks_pred = rankdata(pred_row[mask_row])
    ranks_y = rankdata(y_row[mask_row])
    return _pearson(ranks_pred, ranks_y), n_assets


def _leakage_result(metric_id: str, t_len: int, which: str) -> OOSDeltaResult:
    return OOSDeltaResult(
        metric_id=metric_id, summary=math.nan,
        daily=np.full(t_len, np.nan) if t_len else None,
        status=STATUS_INVALID, reason_code="OOS_WINDOW_LEAKAGE",
        diagnostics={"leaky_manifest": which, "metric_version": METRIC_VERSION},
    )


def _manifests_or_raise(manifest_baseline, manifest_candidate):
    leak_b = _require_manifest(manifest_baseline, "baseline manifest")
    leak_c = _require_manifest(manifest_candidate, "candidate manifest")
    if leak_b:
        return "baseline"
    if leak_c:
        return "candidate"
    return None


def _check_time_axis(manifest_baseline, manifest_candidate, t_len: int):
    if len(manifest_baseline.decision_time_ids) != t_len:
        raise ValueError("baseline manifest decision axis must match the prediction time axis")
    if len(manifest_candidate.decision_time_ids) != t_len:
        raise ValueError("candidate manifest decision axis must match the prediction time axis")


# ---------------------------------------------------------------------------
# M02 kernels
# ---------------------------------------------------------------------------

def oos_rank_ic_delta(labels, baseline_pred, candidate_pred,
                      manifest_baseline: OOSFitManifest,
                      manifest_candidate: OOSFitManifest, *,
                      validity=None, min_assets: int = 50) -> OOSDeltaResult:
    """Sec. 7.2 ``oos_rank_ic_delta``.

    On each day's *common support*, Spearman IC of baseline-vs-y and
    candidate-vs-y separately, differenced, then equal-weight averaged
    over valid days.  Returns the daily delta plus each leg's daily IC
    and the per-day valid-asset counts.  This is a true OOS ablation --
    NOT the same-day descriptive incremental IC of
    ``metrics/interactions/conditional.py``.
    """
    metric_id = "oos_rank_ic_delta"
    if (isinstance(min_assets, bool) or not isinstance(min_assets, (int, np.integer))
            or min_assets < 1):
        raise ValueError("min_assets must be a positive integer")
    leak = _manifests_or_raise(manifest_baseline, manifest_candidate)
    y, p0, p1, mask = _oos_inputs(labels, baseline_pred, candidate_pred, validity)
    t_len = y.shape[0]
    _check_time_axis(manifest_baseline, manifest_candidate, t_len)
    if leak:
        return _leakage_result(metric_id, t_len, leak)

    daily = np.full(t_len, np.nan, dtype=np.float64)
    ic_baseline = np.full(t_len, np.nan, dtype=np.float64)
    ic_candidate = np.full(t_len, np.nan, dtype=np.float64)
    n_assets = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        ic_b, n_t = _daily_rank_ic(p0[t], y[t], mask[t], min_assets)
        n_assets[t] = n_t
        if not math.isfinite(ic_b):
            continue
        ic_c, _ = _daily_rank_ic(p1[t], y[t], mask[t], min_assets)
        if math.isfinite(ic_c):
            ic_baseline[t] = ic_b
            ic_candidate[t] = ic_c
            daily[t] = ic_c - ic_b
    valid = np.isfinite(daily)
    status = STATUS_OK if valid.any() else STATUS_INSUFFICIENT
    reason = "OK" if valid.any() else "OBSERVATIONS_TOO_FEW"
    return OOSDeltaResult(
        metric_id=metric_id,
        summary=float(daily[valid].mean()) if valid.any() else math.nan,
        daily=daily, status=status, reason_code=reason,
        diagnostics={"metric_version": METRIC_VERSION,
                     "ic_baseline": ic_baseline, "ic_candidate": ic_candidate,
                     "n_assets": n_assets,
                     "valid_days": int(valid.sum()),
                     "semantics": "true_oos_ablation_spearman"},
    )


def oos_daily_mse_improvement(labels, baseline_pred, candidate_pred,
                              manifest_baseline: OOSFitManifest,
                              manifest_candidate: OOSFitManifest, *,
                              validity=None, min_assets: int = 50) -> OOSDeltaResult:
    """Sec. 7.2 ``oos_daily_mse_improvement``.

    ``d_t = mean_i[(y - p0)^2 - (y - p1)^2]`` on each day's common
    support; the summary equal-weight averages the valid daily values --
    days with more assets are never secretly up-weighted (Sec. 7.2).
    """
    metric_id = "oos_daily_mse_improvement"
    if (isinstance(min_assets, bool) or not isinstance(min_assets, (int, np.integer))
            or min_assets < 1):
        raise ValueError("min_assets must be a positive integer")
    leak = _manifests_or_raise(manifest_baseline, manifest_candidate)
    y, p0, p1, mask = _oos_inputs(labels, baseline_pred, candidate_pred, validity)
    t_len = y.shape[0]
    _check_time_axis(manifest_baseline, manifest_candidate, t_len)
    if leak:
        return _leakage_result(metric_id, t_len, leak)

    daily = np.full(t_len, np.nan, dtype=np.float64)
    n_assets = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        mask_row = mask[t]
        n_t = int(mask_row.sum())
        n_assets[t] = n_t
        if n_t < min_assets:
            continue
        err_b = (y[t][mask_row] - p0[t][mask_row]) ** 2
        err_c = (y[t][mask_row] - p1[t][mask_row]) ** 2
        daily[t] = float((err_b - err_c).mean())
    valid = np.isfinite(daily)
    status = STATUS_OK if valid.any() else STATUS_INSUFFICIENT
    reason = "OK" if valid.any() else "OBSERVATIONS_TOO_FEW"
    return OOSDeltaResult(
        metric_id=metric_id,
        summary=float(daily[valid].mean()) if valid.any() else math.nan,
        daily=daily, status=status, reason_code=reason,
        diagnostics={"metric_version": METRIC_VERSION, "n_assets": n_assets,
                     "valid_days": int(valid.sum()),
                     "aggregation": "equal_weight_daily_means"},
    )


def oos_r2_gain(labels, baseline_pred, candidate_pred, reference_pred,
                manifest_baseline: OOSFitManifest,
                manifest_candidate: OOSFitManifest, *,
                validity=None, min_assets: int = 50) -> OOSDeltaResult:
    """Sec. 7.2 ``oos_r2_gain``.

    Requires a *frozen* reference prediction ``p_ref`` (zero prediction
    or training-period mean -- fitted on the training window upstream,
    never on the test window).  ``SSE_j = mean_t mean_i (y - p_j)^2`` and
    ``SST_ref = mean_t mean_i (y - p_ref)^2`` over the common support;
    the result is ``(SSE_0 - SSE_1) / SST_ref``.  It is deliberately NOT
    ``(SSE_0 - SSE_1) / SSE_0``.  ``SST_ref == 0`` is insufficient.
    Per-day means over assets are averaged with equal day weights
    (``mean_t mean_i``, Sec. 7.2).
    """
    metric_id = "oos_r2_gain"
    if (isinstance(min_assets, bool) or not isinstance(min_assets, (int, np.integer))
            or min_assets < 1):
        raise ValueError("min_assets must be a positive integer")
    leak = _manifests_or_raise(manifest_baseline, manifest_candidate)
    if reference_pred is None:
        raise TypeError(
            "oos_r2_gain requires a frozen reference prediction p_ref "
            "(Sec. 7.2); a missing reference is not a zero reference")
    y, p0, p1, mask = _oos_inputs(labels, baseline_pred, candidate_pred, validity)
    pr = real_matrix(reference_pred, "reference_pred", ndim=2)
    if pr.shape != y.shape:
        raise ValueError("reference_pred must share the (T, N) label shape")
    mask = mask & np.isfinite(pr)
    t_len = y.shape[0]
    _check_time_axis(manifest_baseline, manifest_candidate, t_len)
    if leak:
        return _leakage_result(metric_id, t_len, leak)

    day_sse0 = np.full(t_len, np.nan, dtype=np.float64)
    day_sse1 = np.full(t_len, np.nan, dtype=np.float64)
    day_sst = np.full(t_len, np.nan, dtype=np.float64)
    for t in range(t_len):
        mask_row = mask[t]
        if int(mask_row.sum()) < min_assets:
            continue
        day_sse0[t] = float(((y[t][mask_row] - p0[t][mask_row]) ** 2).mean())
        day_sse1[t] = float(((y[t][mask_row] - p1[t][mask_row]) ** 2).mean())
        day_sst[t] = float(((y[t][mask_row] - pr[t][mask_row]) ** 2).mean())
    included = np.isfinite(day_sst)
    if not included.any():
        return OOSDeltaResult(
            metric_id=metric_id, summary=math.nan, daily=None,
            status=STATUS_INSUFFICIENT, reason_code="OBSERVATIONS_TOO_FEW",
            diagnostics={"metric_version": METRIC_VERSION})
    sse0 = float(day_sse0[included].mean())
    sse1 = float(day_sse1[included].mean())
    sst_ref = float(day_sst[included].mean())
    if sst_ref == 0.0:
        return OOSDeltaResult(
            metric_id=metric_id, summary=math.nan, daily=None,
            status=STATUS_INSUFFICIENT, reason_code="R2_DENOMINATOR_ZERO",
            diagnostics={"metric_version": METRIC_VERSION,
                         "sse_baseline": sse0, "sse_candidate": sse1,
                         "sst_reference": sst_ref})
    return OOSDeltaResult(
        metric_id=metric_id, summary=(sse0 - sse1) / sst_ref, daily=None,
        status=STATUS_OK, reason_code="OK",
        diagnostics={"metric_version": METRIC_VERSION,
                     "sse_baseline": sse0, "sse_candidate": sse1,
                     "sst_reference": sst_ref,
                     "included_days": int(included.sum())},
    )


def oos_ablation_summary(labels, tasks,
                         manifest_baseline: OOSFitManifest,
                         manifest_candidate: OOSFitManifest, *,
                         validity=None, min_assets: int = 50,
                         reference_predictions=None,
                         paired_returns=None, risk_free_daily=None,
                         periods_per_year: float = 252.0,
                         min_periods: int = 60) -> tuple:
    """Sec. 7.2 ``oos_ablation_summary``.

    ``tasks`` is a sequence of ``(task_id, baseline_pred, candidate_pred)``
    with unique non-empty string ids (each candidate maps 1:1 to an
    upstream ablation/new-addition task id).  Each task receives
    ``oos_rank_ic_delta`` / ``oos_daily_mse_improvement`` /
    ``oos_r2_gain``.  ``reference_predictions`` is either one shared
    ``(T, N)`` frozen reference or a mapping ``task_id -> (T, N)``; tasks
    without a reference get ``r2_gain.status = NOT_COMPUTED`` with
    ``REFERENCE_PREDICTION_MISSING`` (never a silently re-fitted mean).

    ``paired_returns=(candidate_net, baseline_net)`` optionally attaches
    the M01 net-Sharpe increment; without real paired portfolio
    trajectories the economic increment stays ``NOT_COMPUTED`` -- a
    missing trajectory never becomes a fabricated portfolio improvement.
    Group/family ablations use complete upstream feature families only
    (Sec. 7.2); re-training importance proxies (e.g. SHAP) are not a
    substitute and are not computed here.
    """
    if not tasks:
        raise ValueError("tasks must be a non-empty sequence")
    seen = set()
    for task in tasks:
        if not (isinstance(task, (tuple, list)) and len(task) == 3):
            raise ValueError("each task must be (task_id, baseline_pred, candidate_pred)")
        task_id = task[0]
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task ids must be non-empty strings")
        if task_id in seen:
            raise ValueError(f"duplicate ablation task id: {task_id!r}")
        seen.add(task_id)

    if paired_returns is not None:
        if not (isinstance(paired_returns, (tuple, list)) and len(paired_returns) == 2):
            raise ValueError("paired_returns must be (candidate_net, baseline_net)")
        sharpe_delta = paired_net_sharpe_delta(
            paired_returns[0], paired_returns[1],
            risk_free_daily=risk_free_daily, periods_per_year=periods_per_year,
            min_periods=min_periods)
        economic_status = sharpe_delta.status
    else:
        sharpe_delta = None
        economic_status = "NOT_COMPUTED"

    results = []
    for task_id, base_pred, cand_pred in tasks:
        rank_ic = oos_rank_ic_delta(labels, base_pred, cand_pred,
                                    manifest_baseline, manifest_candidate,
                                    validity=validity, min_assets=min_assets)
        mse = oos_daily_mse_improvement(labels, base_pred, cand_pred,
                                        manifest_baseline, manifest_candidate,
                                        validity=validity, min_assets=min_assets)
        if reference_predictions is None:
            r2 = OOSDeltaResult(metric_id="oos_r2_gain", summary=math.nan,
                                daily=None, status="NOT_COMPUTED",
                                reason_code="REFERENCE_PREDICTION_MISSING",
                                diagnostics={"metric_version": METRIC_VERSION})
        elif isinstance(reference_predictions, dict):
            r2 = oos_r2_gain(labels, base_pred, cand_pred,
                             reference_predictions[task_id],
                             manifest_baseline, manifest_candidate,
                             validity=validity, min_assets=min_assets)
        else:
            r2 = oos_r2_gain(labels, base_pred, cand_pred,
                             reference_predictions,
                             manifest_baseline, manifest_candidate,
                             validity=validity, min_assets=min_assets)
        results.append(OOSAblationTaskResult(
            task_id=task_id, rank_ic_delta=rank_ic,
            daily_mse_improvement=mse, r2_gain=r2,
            net_sharpe_delta=sharpe_delta, economic_status=economic_status,
        ))
    return tuple(results)
