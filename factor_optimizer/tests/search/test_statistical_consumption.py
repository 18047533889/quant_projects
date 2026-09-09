import numpy as np
import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
from factor_optimizer.search.statistical_consumption import build_search_statistical_evidence


def _store(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "campaign.sqlite3")
    for index in range(3):
        store.record_hypothesis_attempt(
            campaign_id="campaign", proposal_id=f"proposal-{index}",
            effective_spec_hash=f"spec-{index}", evaluation_intent_hash="intent",
            horizon=index + 1, executed=True, has_pvalue=True,
        )
    return store


def _kwargs():
    t = 60
    x = np.arange(t, dtype=float)
    matrix = np.column_stack([
        0.001 + np.sin(x / 4) * 0.01,
        0.002 + np.cos(x / 5) * 0.01,
        0.0015 + np.sin(x / 7) * 0.01,
    ])
    return dict(
        candidate_returns=matrix, family_sharpes=(0.10, 0.12, 0.11),
        selected_candidate_index=1, trial_ledger_ref="ledger:campaign",
        common_cost_spec_ref="cost:v1", returns_frequency="daily",
        annualization_factor=252, train_series=matrix[:, 0],
        validation_series=matrix[:, 1], train_value=0.03, validation_value=0.02,
        horizon_ic_series={1: np.full(t, 0.04), 2: np.full(t, 0.02), 3: np.full(t, 0.01)},
        horizon_label_refs={1: "label:1", 2: "label:2", 3: "label:3"},
        regime_values=matrix[:, 1], regime_labels=np.repeat([0, 1], t // 2),
        regime_available_times=tuple(range(t)), regime_kind="online_causal",
        regime_state_ref="regime-state:v1", pbo_splits=6,
        bootstrap_repetitions=50, min_periods=20,
    )


def test_fo_consumes_complete_durable_family_into_typed_qe_bundle(tmp_path):
    evidence = build_search_statistical_evidence(_store(tmp_path), "campaign", **_kwargs())
    assert evidence.hypothesis_summary["unique_effective_spec_count"] == 3
    assert evidence.dsr.status == "VALID"
    assert evidence.pbo.status == "VALID"
    assert evidence.retention.status == "VALID"
    assert evidence.horizon_curve.fit_status == "VALID"
    assert evidence.regime.status == "VALID"


def test_fo_rejects_winner_only_or_partial_family_before_qe(tmp_path):
    kwargs = _kwargs()
    kwargs["candidate_returns"] = kwargs["candidate_returns"][:, :1]
    kwargs["family_sharpes"] = (0.12,)
    kwargs["selected_candidate_index"] = 0
    with pytest.raises(ValueError, match="cover every durable effective spec"):
        build_search_statistical_evidence(_store(tmp_path), "campaign", **kwargs)
