"""
R61-FI-029 — adversarial fit-apply guard tests (plan §29).

Five adversarial scenarios must each be constructed and asserted to be
rejected by a guard:

1. future append
2. split leakage
3. fit state reused across wrong snapshot/universe
4. wrong label access
5. wrong as-of date
"""
import datetime

import numpy as np
import pytest

from factor_preprocess.contracts.fit_apply import (
    CausalityClass,
    FitScope,
    ApplicationSplit,
    TreatmentFitApplyDeclaration,
    assert_prefix_invariant,
    assert_fit_state_context_match,
    assert_label_access_legal,
)
from factor_preprocess.errors import (
    InvalidContractError,
    TimingContractError,
    StaleFittedStateError,
    SnapshotMismatchError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _declaration(**overrides):
    kwargs = dict(
        treatment_id="cs_rank",
        causality_class=CausalityClass.CROSS_SECTIONAL_CAUSAL,
        fit_scope=FitScope.CROSS_SECTIONAL_DATE,
        fit_split=ApplicationSplit.TRAIN,
        state_identity="stateless",
        application_scope=ApplicationSplit.PRODUCTION,
    )
    kwargs.update(overrides)
    return TreatmentFitApplyDeclaration(**kwargs)


def _causal_trailing_mean(values):
    """Prefix-invariant trailing (causal) mean — output at t uses rows <= t."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.mean(values[max(0, i - 2): i + 1])
    return out


def _centered_mean(values):
    """Non-causal centered mean — output at t uses rows t-1..t+1."""
    return np.convolve(np.asarray(values, dtype=float), np.ones(3) / 3, mode="same")


# ---------------------------------------------------------------------------
# 1. future append
# ---------------------------------------------------------------------------


def test_future_append_causal_transform_is_prefix_invariant():
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(size=40))
    assert_prefix_invariant(_causal_trailing_mean, values, split_index=15)
    assert_prefix_invariant(_causal_trailing_mean, values, split_index=30)


def test_future_append_centered_transform_rejected():
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(size=40))
    with pytest.raises(TimingContractError):
        assert_prefix_invariant(_centered_mean, values, split_index=15)


def test_future_append_identity_and_nan_preserving_ok():
    rng = np.random.default_rng(3)
    values = rng.normal(size=25)
    values[3] = np.nan
    assert_prefix_invariant(lambda v: np.asarray(v, dtype=float), values, split_index=12)


# ---------------------------------------------------------------------------
# 2. split leakage
# ---------------------------------------------------------------------------


def test_split_leakage_fitted_on_test_applied_to_production_rejected():
    decl = _declaration(
        treatment_id="scaler",
        causality_class=CausalityClass.TRAIN_FITTED,
        fit_scope=FitScope.TRAIN_ONLY,
        fit_split=ApplicationSplit.TEST,  # fitted on TEST...
        application_scope=ApplicationSplit.PRODUCTION,  # ...applied to PROD
        state_identity="scaler_state_v1",
    )
    with pytest.raises(TimingContractError):
        decl.assert_no_split_leakage()


def test_train_fitted_applied_to_production_is_legal():
    decl = _declaration(
        treatment_id="scaler",
        causality_class=CausalityClass.TRAIN_FITTED,
        fit_scope=FitScope.TRAIN_ONLY,
        fit_split=ApplicationSplit.TRAIN,
        application_scope=ApplicationSplit.PRODUCTION,
        state_identity="scaler_state_v1",
    )
    decl.assert_no_split_leakage()
    assert decl.application_scope.is_evaluation_valid


def test_full_sample_research_fit_never_evaluation_valid():
    decl = _declaration(
        treatment_id="hp_smoother",
        causality_class=CausalityClass.OFFLINE_ONLY,
        fit_scope=FitScope.FULL_SAMPLE_RESEARCH,
        fit_split=ApplicationSplit.FULL_SAMPLE_RESEARCH,
        application_scope=ApplicationSplit.FULL_SAMPLE_RESEARCH,
        state_identity="hp_full",
    )
    decl.validate_self_consistent()
    assert decl.fit_scope.is_research_only
    with pytest.raises(TimingContractError):
        # Even a "TEST" application of a full-sample fit is leakage: the test
        # rows were part of the fit sample.
        _declaration(
            treatment_id="hp_smoother",
            causality_class=CausalityClass.OFFLINE_ONLY,
            fit_scope=FitScope.FULL_SAMPLE_RESEARCH,
            fit_split=ApplicationSplit.FULL_SAMPLE_RESEARCH,
            application_scope=ApplicationSplit.TEST,
            state_identity="hp_full",
        ).assert_no_split_leakage()


# ---------------------------------------------------------------------------
# 3. fit state reused across wrong snapshot / universe
# ---------------------------------------------------------------------------


def test_fit_state_reused_across_wrong_data_snapshot_rejected():
    with pytest.raises(SnapshotMismatchError):
        assert_fit_state_context_match(
            state_identity="scaler_v1",
            state_data_snapshot_ref="snapshot_20260801",
            apply_data_snapshot_ref="snapshot_20260901",
            state_universe_ref="ashare_full",
            apply_universe_ref="ashare_full",
            state_feature_order=["f1"],
            apply_feature_order=["f1"],
        )


def test_fit_state_reused_across_wrong_universe_rejected():
    with pytest.raises(StaleFittedStateError):
        assert_fit_state_context_match(
            state_identity="scaler_v1",
            state_data_snapshot_ref="snapshot_20260801",
            apply_data_snapshot_ref="snapshot_20260801",
            state_universe_ref="ashare_500",
            apply_universe_ref="ashare_full",
            state_feature_order=["f1"],
            apply_feature_order=["f1"],
        )


def test_fit_state_reused_across_wrong_feature_order_rejected():
    with pytest.raises(StaleFittedStateError):
        assert_fit_state_context_match(
            state_identity="scaler_v1",
            state_data_snapshot_ref="snapshot_20260801",
            apply_data_snapshot_ref="snapshot_20260801",
            state_universe_ref="ashare_full",
            apply_universe_ref="ashare_full",
            state_feature_order=["f1", "f2"],
            apply_feature_order=["f2", "f1"],
        )


def test_fit_state_context_match_same_context_passes():
    assert_fit_state_context_match(
        state_identity="scaler_v1",
        state_data_snapshot_ref="snapshot_20260801",
        apply_data_snapshot_ref="snapshot_20260801",
        state_universe_ref="ashare_full",
        apply_universe_ref="ashare_full",
        state_feature_order=["f1", "f2"],
        apply_feature_order=["f1", "f2"],
    )


# ---------------------------------------------------------------------------
# 4. wrong label access
# ---------------------------------------------------------------------------


def test_label_knowledge_after_asof_rejected():
    with pytest.raises(TimingContractError):
        assert_label_access_legal(
            label_knowledge_time=datetime.date(2026, 9, 5),
            asof_timestamp=datetime.date(2026, 9, 3),
            label_name="vwap_to_vwap_h10",
        )


def test_label_knowledge_at_or_before_asof_legal():
    assert_label_access_legal(
        label_knowledge_time=datetime.date(2026, 9, 3),
        asof_timestamp=datetime.date(2026, 9, 3),
    )
    assert_label_access_legal(
        label_knowledge_time=datetime.date(2026, 9, 1),
        asof_timestamp=datetime.date(2026, 9, 3),
    )


# ---------------------------------------------------------------------------
# 5. wrong as-of date
# ---------------------------------------------------------------------------


def test_apply_asof_before_fit_window_end_rejected():
    decl = _declaration(
        treatment_id="scaler",
        causality_class=CausalityClass.TRAIN_FITTED,
        fit_scope=FitScope.TRAIN_ONLY,
        fit_split=ApplicationSplit.TRAIN,
        application_scope=ApplicationSplit.PRODUCTION,
        state_identity="scaler_v1",
    )
    with pytest.raises(TimingContractError):
        decl.assert_asof_legal(
            asof_timestamp=datetime.datetime(2026, 8, 1),
            fit_end_timestamp=datetime.datetime(2026, 8, 15),
        )


def test_apply_asof_after_fit_window_end_legal():
    decl = _declaration(
        treatment_id="scaler",
        causality_class=CausalityClass.TRAIN_FITTED,
        fit_scope=FitScope.TRAIN_ONLY,
        fit_split=ApplicationSplit.TRAIN,
        application_scope=ApplicationSplit.PRODUCTION,
        state_identity="scaler_v1",
    )
    decl.assert_asof_legal(
        asof_timestamp=datetime.datetime(2026, 9, 1),
        fit_end_timestamp=datetime.datetime(2026, 8, 15),
    )
    decl.assert_asof_legal(
        asof_timestamp=datetime.datetime(2026, 8, 15),
        fit_end_timestamp=datetime.datetime(2026, 8, 15),
    )


# ---------------------------------------------------------------------------
# declaration completeness / self-consistency
# ---------------------------------------------------------------------------


def test_declaration_requires_all_five_fields():
    with pytest.raises(InvalidContractError):
        _declaration(state_identity="")  # state_identity required
    with pytest.raises(InvalidContractError):
        _declaration(treatment_id="")  # treatment_id required


def test_offline_only_cannot_apply_to_production():
    with pytest.raises(InvalidContractError):
        _declaration(
            treatment_id="hp",
            causality_class=CausalityClass.OFFLINE_ONLY,
            fit_scope=FitScope.FULL_SAMPLE_RESEARCH,
            fit_split=ApplicationSplit.FULL_SAMPLE_RESEARCH,
            application_scope=ApplicationSplit.PRODUCTION,
            state_identity="hp_full",
        ).validate_self_consistent()


def test_train_only_scope_cannot_declare_evaluation_fit_split():
    with pytest.raises(InvalidContractError):
        _declaration(
            treatment_id="scaler",
            causality_class=CausalityClass.TRAIN_FITTED,
            fit_scope=FitScope.TRAIN_ONLY,
            fit_split=ApplicationSplit.TEST,  # contradictory
            application_scope=ApplicationSplit.PRODUCTION,
            state_identity="scaler_v1",
        ).validate_self_consistent()


def test_cs_rank_declaration_matches_plan_example():
    # plan §29: cs_rank -> no fitted state, cross-sectional current-date.
    decl = _declaration()
    assert decl.causality_class is CausalityClass.CROSS_SECTIONAL_CAUSAL
    assert decl.fit_scope is FitScope.CROSS_SECTIONAL_DATE
    assert decl.fit_split is ApplicationSplit.TRAIN
    assert decl.state_identity == "stateless"
    decl.validate_self_consistent()
    decl.assert_no_split_leakage()
