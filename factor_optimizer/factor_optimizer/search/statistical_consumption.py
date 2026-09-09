"""FO consumption of QE statistical evidence bound to durable campaign state."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
from quant_evaluator.contracts.statistical_evidence import (
    DSREvidence,
    HorizonCurveEvidence,
    PBOEvidence,
    RegimeEvidence,
    RetentionEvidence,
)
from quant_evaluator.metrics.statistical_evidence import (
    build_dsr_evidence,
    build_horizon_curve_evidence,
    build_pbo_evidence,
    build_regime_evidence,
    build_retention_evidence,
)


@dataclass(frozen=True)
class SearchStatisticalEvidence:
    """Auditable statistical bundle consumed by an FO search decision."""

    campaign_id: str
    hypothesis_summary: Mapping[str, int]
    dsr: DSREvidence
    pbo: PBOEvidence
    retention: RetentionEvidence
    horizon_curve: HorizonCurveEvidence
    regime: RegimeEvidence


def build_search_statistical_evidence(
    store: SQLiteCampaignStore,
    campaign_id: str,
    candidate_returns: np.ndarray,
    family_sharpes: Sequence[float],
    *,
    selected_candidate_index: int,
    trial_ledger_ref: str,
    common_cost_spec_ref: str,
    returns_frequency: str,
    annualization_factor: float,
    train_series: Sequence[float],
    validation_series: Sequence[float],
    train_value: float,
    validation_value: float,
    horizon_ic_series: Mapping[int, Sequence[float]],
    horizon_label_refs: Mapping[int, str],
    regime_values: Sequence[float],
    regime_labels: Sequence[int],
    regime_available_times: Sequence[object],
    regime_kind: str,
    regime_state_ref: Optional[str] = None,
    family_sharpe_scale: str = "raw_periodic",
    pbo_splits: int = 6,
    near_zero: float = 1e-3,
    bootstrap_block_length: int = 10,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 0,
    min_periods: int = 30,
) -> SearchStatisticalEvidence:
    """Build the statistical bundle from a complete durable search family.

    The matrix columns and Sharpe family must correspond one-for-one to the
    durable unique effective specifications. This prevents a selected winner
    from being evaluated as though it were the full searched family.
    """
    if not isinstance(store, SQLiteCampaignStore):
        raise TypeError("store must be a SQLiteCampaignStore")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id is required")
    matrix = np.asarray(candidate_returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("candidate_returns must be a T×C matrix")
    summary = dict(store.hypothesis_family_summary(campaign_id))
    expected = summary["unique_effective_spec_count"]
    if expected < 2:
        raise ValueError("statistical search evidence requires at least two durable effective specs")
    if matrix.shape[1] != expected or len(family_sharpes) != expected:
        raise ValueError("candidate matrix and Sharpe family must cover every durable effective spec")
    if (not isinstance(selected_candidate_index, int) or isinstance(selected_candidate_index, bool)
            or not 0 <= selected_candidate_index < expected):
        raise ValueError("selected_candidate_index is out of range")

    dsr = build_dsr_evidence(
        matrix[:, selected_candidate_index],
        family_sharpes=family_sharpes,
        effective_trial_count=expected,
        trial_ledger_ref=trial_ledger_ref,
        returns_frequency=returns_frequency,
        annualization_factor=annualization_factor,
        family_sharpe_scale=family_sharpe_scale,
        min_periods=min_periods,
    )
    pbo = build_pbo_evidence(
        matrix,
        n_splits=pbo_splits,
        candidate_universe_complete=True,
        common_cost_spec_ref=common_cost_spec_ref,
        purpose="diagnostic",
    )
    retention = build_retention_evidence(
        train_series,
        validation_series,
        train_value=train_value,
        validation_value=validation_value,
        near_zero=near_zero,
        cost_basis="net",
        block_length=bootstrap_block_length,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        min_periods=min_periods,
    )
    horizon = build_horizon_curve_evidence(
        horizon_ic_series, label_refs=horizon_label_refs, min_periods=min_periods,
    )
    regime = build_regime_evidence(
        regime_values,
        regime_labels,
        regime_available_times,
        regime_kind=regime_kind,
        state_ref=regime_state_ref,
        min_periods=min_periods,
    )
    return SearchStatisticalEvidence(
        campaign_id, summary, dsr, pbo, retention, horizon, regime,
    )
