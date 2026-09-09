"""
Tests for the residual-IC conditional-novelty producer (QE wiring).

Pins the producer behind SelectionPolicy.make_decision's
novelty_score / residual_ic parameters (FA-P0-3's SHADOW /
SHADOWED_COEXIST chain): QE compute_incremental_ic → signals →
EvidenceResult.secondary_metrics → make_decision.
"""

import numpy as np
import pytest

import factor_assets.adapters.residual_novelty as rn_module
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

from factor_assets.adapters.residual_novelty import (
    DEFAULT_MIN_PERIODS,
    INCREMENTAL_IC_MEAN_KEY,
    NOVELTY_SCORE_KEY,
    NUM_VALID_PERIODS_KEY,
    RESIDUAL_IC_KEY,
    TOTAL_IC_MEAN_KEY,
    ResidualICNoveltyProducer,
    evidence_result_from_signals,
    novelty_inputs_from_evidence,
)
from factor_assets.novelty.provider import EvidenceResult
from factor_assets.selection import SelectionPolicy
from factor_assets.selection.gates import GateEvaluation, GateResult


def _make_batch(T=80, N=60, seed=7):
    """Build a batch whose factor 0 is the signal and factor 1 is a
    noisier copy of it (high-similarity admitted factor); factor 2 is the
    candidate — the label plus independent noise relative to factor 1."""
    rng = np.random.default_rng(seed)
    label = rng.normal(size=(T, N))
    base_signal = label + 0.5 * rng.normal(size=(T, N))
    redundant = base_signal + 0.2 * rng.normal(size=(T, N))
    candidate = label + 1.5 * rng.normal(size=(T, N))
    values = np.stack([base_signal, redundant, candidate], axis=2)
    batch = FactorBatch(
        factor_ids=("base", "redundant", "candidate"),
        time_axis=AxisRef(name="time", dtype="int64", size=T),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N),
        values=values,
    )
    bundle = LabelBundle(
        target_id="return",
        values=label,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    return batch, bundle


def _passing_gate(factor_id="F001"):
    return GateEvaluation(
        gate_name="test_gate",
        factor_id=factor_id,
        result=GateResult.PASS,
        timestamp="2024-01-01T00:00:00Z",
    )


class TestComputeSignals:
    def test_default_admission_path_rejects_same_date_descriptive_evidence(self):
        batch, labels = _make_batch()
        producer = ResidualICNoveltyProducer()
        assert producer.estimation_scope == "SAME_DATE_DESCRIPTIVE"
        assert producer.compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        ) is None

    def test_returns_all_signal_keys(self):
        batch, labels = _make_batch()
        producer = ResidualICNoveltyProducer(allow_same_date_descriptive=True)
        signals = producer.compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert signals is not None
        for key in (
            RESIDUAL_IC_KEY,
            INCREMENTAL_IC_MEAN_KEY,
            TOTAL_IC_MEAN_KEY,
            NOVELTY_SCORE_KEY,
            NUM_VALID_PERIODS_KEY,
        ):
            assert key in signals
        assert signals[NUM_VALID_PERIODS_KEY] >= DEFAULT_MIN_PERIODS

    def test_novelty_score_in_unit_interval(self):
        batch, labels = _make_batch()
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert 0.0 <= signals[NOVELTY_SCORE_KEY] <= 1.0

    def test_redundant_factor_scores_low(self):
        # Factor 1 is a noisy copy of base factor 0: most of its IC is
        # explained, so residual novelty must be low.
        batch, labels = _make_batch()
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=1
        )
        assert signals[NOVELTY_SCORE_KEY] < 0.5

    def test_independent_factor_scores_higher_than_redundant(self):
        batch, labels = _make_batch()
        producer = ResidualICNoveltyProducer(allow_same_date_descriptive=True)
        redundant = producer.compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=1
        )
        independent = producer.compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert (
            independent[NOVELTY_SCORE_KEY]
            > redundant[NOVELTY_SCORE_KEY]
        )

    def test_insufficient_periods_fails_closed(self):
        # Every period below min_assets → all-NaN incremental IC → None.
        batch, labels = _make_batch(T=5, N=3)
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert signals is None

    def test_absent_total_ic_fails_closed(self):
        # A corrupt/absent total-IC series (finite incremental IC but no
        # finite total IC) must NOT produce a maximal novelty score — the
        # signal is absent entirely (fail-closed, never fail-open).
        producer = ResidualICNoveltyProducer(allow_same_date_descriptive=True)
        batch, labels = _make_batch()
        # Simulate QE returning NaN total_ic: patch via a thin wrapper is
        # overkill; instead check the consistency gate directly with a
        # fabricated series through map-free path — use a producer whose
        # method we intercept.
        class _NaNTotalProducer(ResidualICNoveltyProducer):
            def compute_signals(self, factor_batch, label_bundle, base_factor_indices, test_factor_idx):
                import quant_evaluator.metrics as qm
                real = qm.compute_incremental_ic
                def patched(fb, lb, base_factor_indices, test_factor_idx, method, min_assets):
                    incr, base, total = real(
                        fb, lb,
                        base_factor_indices=base_factor_indices,
                        test_factor_idx=test_factor_idx,
                        method=method,
                        min_assets=min_assets,
                    )
                    return incr, base, np.full_like(total, np.nan)
                import factor_assets.adapters.residual_novelty as rn
                original = rn._qe_compute_incremental_ic
                rn._qe_compute_incremental_ic = patched
                try:
                    return super().compute_signals(
                        factor_batch, label_bundle,
                        base_factor_indices=base_factor_indices,
                        test_factor_idx=test_factor_idx,
                    )
                finally:
                    rn._qe_compute_incremental_ic = original

        signals = _NaNTotalProducer().compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert signals is None  # no total-IC evidence → no signal at all

    def test_min_periods_gate(self):
        # T=10 valid periods < min_periods=20 → None; min_periods=5 → signal.
        batch, labels = _make_batch(T=10)
        producer = ResidualICNoveltyProducer(allow_same_date_descriptive=True)
        assert (
            producer.compute_signals(
                batch, labels, base_factor_indices=(0,), test_factor_idx=2
            )
            is None
        )
        lenient = ResidualICNoveltyProducer(min_periods=5, allow_same_date_descriptive=True)
        assert (
            lenient.compute_signals(
                batch, labels, base_factor_indices=(0,), test_factor_idx=2
            )
            is not None
        )

    def test_num_valid_periods_counts_both_finite(self):
        # num_valid_periods must count periods where BOTH incremental and
        # total IC are finite (the mask behind the means), not just
        # finite-increment periods — otherwise the reported evidence
        # overstates the periods the ratio actually used.
        class _PartialNaNTotalProducer(ResidualICNoveltyProducer):
            def compute_signals(self, factor_batch, label_bundle, base_factor_indices, test_factor_idx):
                import quant_evaluator.metrics as qm
                real = qm.compute_incremental_ic
                def patched(fb, lb, base_factor_indices, test_factor_idx, method, min_assets):
                    incr, base, total = real(
                        fb, lb,
                        base_factor_indices=base_factor_indices,
                        test_factor_idx=test_factor_idx,
                        method=method,
                        min_assets=min_assets,
                    )
                    # Half the total-IC series NaN: both_finite < finite_incr
                    total = total.copy()
                    total[::2] = np.nan
                    return incr, base, total
                import factor_assets.adapters.residual_novelty as rn
                original = rn._qe_compute_incremental_ic
                rn._qe_compute_incremental_ic = patched
                try:
                    return super().compute_signals(
                        factor_batch, label_bundle,
                        base_factor_indices=base_factor_indices,
                        test_factor_idx=test_factor_idx,
                    )
                finally:
                    rn._qe_compute_incremental_ic = original

        batch, labels = _make_batch(T=20)
        signals = _PartialNaNTotalProducer(
            min_periods=5, allow_same_date_descriptive=True,
        ).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert signals is not None
        # Reported count equals usable periods (both series finite), not
        # the larger finite-increment count.
        assert signals[NUM_VALID_PERIODS_KEY] < 20
        assert signals[NUM_VALID_PERIODS_KEY] >= 5

    def test_empty_base_indices_rejected(self):
        batch, labels = _make_batch()
        with pytest.raises(ValueError, match="non-empty"):
            ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
                batch, labels, base_factor_indices=(), test_factor_idx=2
            )

    def test_self_residualization_rejected(self):
        batch, labels = _make_batch()
        with pytest.raises(ValueError, match="self-residualization"):
            ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
                batch, labels, base_factor_indices=(0, 2), test_factor_idx=2
            )

    def test_constructor_validates(self):
        with pytest.raises(ValueError, match="min_periods"):
            ResidualICNoveltyProducer(min_periods=0)
        with pytest.raises(ValueError, match="method"):
            ResidualICNoveltyProducer(method="kendall")
        with pytest.raises(ValueError, match="min_assets"):
            ResidualICNoveltyProducer(min_assets=1)


class TestMapNoveltyScore:
    def test_abs_ratio_bounded(self):
        assert ResidualICNoveltyProducer.map_novelty_score(0.05, 0.10) == pytest.approx(0.5)

    def test_clamped_to_one(self):
        assert ResidualICNoveltyProducer.map_novelty_score(0.8, 0.1) == 1.0

    def test_zero_residual_is_zero(self):
        assert ResidualICNoveltyProducer.map_novelty_score(0.0, 0.1) == 0.0

    def test_nonfinite_residual_is_zero(self):
        assert ResidualICNoveltyProducer.map_novelty_score(None, 0.1) == 0.0
        assert ResidualICNoveltyProducer.map_novelty_score(float("nan"), 0.1) == 0.0

    def test_absent_total_uses_residual_presence(self):
        assert ResidualICNoveltyProducer.map_novelty_score(0.03, None) == 1.0
        assert ResidualICNoveltyProducer.map_novelty_score(0.0, None) == 0.0

    def test_near_zero_total_ic_does_not_blows_up_ratio(self):
        # total_ic just above the old 1e-12 eps used to take the ratio
        # branch: any residual noise produced a full-novelty score.  With
        # the scaled floor, a degenerate denominator falls back to
        # residual-presence semantics: a noise-level residual (1e-8) is
        # NOT novel signal, while a real residual (0.05) with a degenerate
        # denominator is fully novel — never a 10.0 ratio clamped to 1.0.
        assert ResidualICNoveltyProducer.map_novelty_score(1e-8, 1e-9) == 0.0
        assert ResidualICNoveltyProducer.map_novelty_score(0.05, 1e-9) == 1.0
        # Just above the floor the ratio branch applies again.
        assert ResidualICNoveltyProducer.map_novelty_score(0.05, 0.10) == pytest.approx(0.5)

    def test_eps_guard_is_scaled_to_ic_magnitudes(self):
        # The floor must sit far above float noise but far below real ICs.
        assert rn_module._TOTAL_IC_EPS >= 1e-6
        assert rn_module._TOTAL_IC_EPS <= 1e-2


class TestEvidenceRoundTrip:
    def test_signals_wrapped_as_evidence_result(self):
        batch, labels = _make_batch()
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        evidence = evidence_result_from_signals(
            signals,
            factor_id="F001",
            evaluation_run_id="run-001",
            timestamp="2026-08-19T00:00:00Z",
        )
        assert evidence.primary_metric_name == RESIDUAL_IC_KEY
        assert evidence.primary_metric_value == signals[RESIDUAL_IC_KEY]
        assert evidence.secondary_metrics[NOVELTY_SCORE_KEY] == signals[NOVELTY_SCORE_KEY]

    def test_round_trip_extracts_inputs(self):
        batch, labels = _make_batch()
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        evidence = evidence_result_from_signals(
            signals,
            factor_id="F001",
            evaluation_run_id="run-001",
            timestamp="2026-08-19T00:00:00Z",
        )
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        assert novelty_score == signals[NOVELTY_SCORE_KEY]
        assert residual_ic == signals[RESIDUAL_IC_KEY]

    def test_missing_residual_key_rejected(self):
        with pytest.raises(ValueError, match="residual_ic"):
            evidence_result_from_signals(
                {"novelty_score": 0.5},
                factor_id="F001",
                evaluation_run_id="run-001",
                timestamp="2026-08-19T00:00:00Z",
            )

    def test_none_evidence_yields_absent_inputs(self):
        assert novelty_inputs_from_evidence(None) == (None, None)

    def test_unavailable_evidence_yields_absent_inputs(self):
        evidence = EvidenceResult(
            factor_id="F001",
            evaluation_run_id="run-001",
            evidence_id="e-1",
            timestamp="2026-08-19T00:00:00Z",
            available=False,
        )
        assert novelty_inputs_from_evidence(evidence) == (None, None)

    def test_nonfinite_signals_reported_absent(self):
        evidence = EvidenceResult(
            factor_id="F001",
            evaluation_run_id="run-001",
            evidence_id="e-1",
            timestamp="2026-08-19T00:00:00Z",
            available=True,
            primary_metric_name=RESIDUAL_IC_KEY,
            primary_metric_value=float("nan"),
            secondary_metrics={
                NOVELTY_SCORE_KEY: float("nan"),
                RESIDUAL_IC_KEY: float("nan"),
            },
        )
        assert novelty_inputs_from_evidence(evidence) == (None, None)

    def test_non_numeric_signal_reported_absent(self):
        # A corrupt record storing a string must be treated as absent,
        # not crash the extraction.
        evidence = EvidenceResult(
            factor_id="F001",
            evaluation_run_id="run-001",
            evidence_id="e-1",
            timestamp="2026-08-19T00:00:00Z",
            available=True,
            primary_metric_name=RESIDUAL_IC_KEY,
            primary_metric_value=0.2,
            secondary_metrics={NOVELTY_SCORE_KEY: "0.5", RESIDUAL_IC_KEY: 0.2},
        )
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        assert novelty_score is None
        assert residual_ic == 0.2

    def test_out_of_range_novelty_score_reported_absent(self):
        evidence = EvidenceResult(
            factor_id="F001",
            evaluation_run_id="run-001",
            evidence_id="e-1",
            timestamp="2026-08-19T00:00:00Z",
            available=True,
            primary_metric_name=RESIDUAL_IC_KEY,
            primary_metric_value=0.2,
            secondary_metrics={NOVELTY_SCORE_KEY: 1.5, RESIDUAL_IC_KEY: 0.2},
        )
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        assert novelty_score is None
        assert residual_ic == 0.2


class TestPolicyIntegration:
    """End-to-end: producer signals → evidence → make_decision."""

    def _policy(self):
        return SelectionPolicy(
            policy_name="test_policy",
            policy_version="1.0",
            similarity_threshold=0.7,
        )

    def _evidence(self, seed=7):
        batch, labels = _make_batch(seed=seed)
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=2
        )
        assert signals is not None
        return evidence_result_from_signals(
            signals,
            factor_id="F001",
            evaluation_run_id="run-001",
            timestamp="2026-08-19T00:00:00Z",
        )

    def test_independent_factor_coexists(self):
        evidence = self._evidence()
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        decision = self._policy().make_decision(
            factor_id="F001",
            gate_evaluations=[_passing_gate()],
            evidence_refs=(evidence.evidence_id,),
            max_similarity=0.85,  # above threshold → novelty decides
            novelty_score=novelty_score,
            residual_ic=residual_ic,
            novelty_refs=(evidence.evidence_id,),
        )
        assert decision.approved
        assert decision.reason.value == "SHADOWED_COEXIST"

    def test_redundant_factor_shadows(self):
        # With a nonzero novelty floor, a mostly-redundant factor (a noisy
        # copy of the base) must be recorded as SHADOW, never admitted.
        batch, labels = _make_batch()
        signals = ResidualICNoveltyProducer(allow_same_date_descriptive=True).compute_signals(
            batch, labels, base_factor_indices=(0,), test_factor_idx=1
        )
        assert signals is not None
        assert signals[NOVELTY_SCORE_KEY] < 0.5
        evidence = evidence_result_from_signals(
            signals,
            factor_id="F001",
            evaluation_run_id="run-001",
            timestamp="2026-08-19T00:00:00Z",
        )
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        policy = SelectionPolicy(
            policy_name="test_policy",
            policy_version="1.0",
            similarity_threshold=0.7,
            min_novelty_score=0.5,
        )
        decision = policy.make_decision(
            factor_id="F001",
            gate_evaluations=[_passing_gate()],
            evidence_refs=(evidence.evidence_id,),
            max_similarity=0.85,
            novelty_score=novelty_score,
            residual_ic=residual_ic,
            novelty_refs=(evidence.evidence_id,),
        )
        assert decision.reason.value == "SHADOW"
        assert not decision.approved

    def test_independent_factor_meets_nonzero_floor(self):
        # The independent candidate must clear a 0.5 novelty floor and
        # coexist despite high similarity to the admitted base factor.
        evidence = self._evidence()
        novelty_score, residual_ic = novelty_inputs_from_evidence(evidence)
        assert novelty_score >= 0.5
        policy = SelectionPolicy(
            policy_name="test_policy",
            policy_version="1.0",
            similarity_threshold=0.7,
            min_novelty_score=0.5,
        )
        decision = policy.make_decision(
            factor_id="F001",
            gate_evaluations=[_passing_gate()],
            evidence_refs=(evidence.evidence_id,),
            max_similarity=0.85,
            novelty_score=novelty_score,
            residual_ic=residual_ic,
            novelty_refs=(evidence.evidence_id,),
        )
        assert decision.approved
        assert decision.reason.value == "SHADOWED_COEXIST"

    def test_absent_signal_still_rejects_on_similarity(self):
        decision = self._policy().make_decision(
            factor_id="F001",
            gate_evaluations=[_passing_gate()],
            evidence_refs=("EVD_001",),
            max_similarity=0.85,
            novelty_refs=("SIM_001",),
        )
        assert not decision.approved
        assert decision.reason.value == "REJECTED_SIMILARITY"
