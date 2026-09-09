import pytest

from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
from factor_assets.clustering.incremental import PairwiseEvidenceStatus
from quant_evaluator.metrics.interactions.pairwise import (
    PairwiseCorrelationArtifact,
    PairwiseMeasurementStatus,
    PairwiseWindowEvidence,
)
from factor_assets.similarity.exact import QEPairwiseSimilarity, SimilarityMethod, SimilarityMeasurementStatus


def _qe(*, value=0.0, status=PairwiseMeasurementStatus.COMPUTED,
        count=40, window="250d", universe="ASHARE"):
    return PairwiseCorrelationArtifact(
        "FX", "F1", value, status, count, 1, (count,), "pearson",
        window, universe, "sample-1",
    )


def test_real_qe_adapter_preserves_computed_zero_and_signed_value():
    provider = QEEvidenceProvider()
    zero = provider.create_pairwise_evidence(
        _qe(), expected_factor_ids=("F1", "FX"),
        expected_window_ref="250d", expected_universe_ref="ASHARE",
    )
    assert zero.status is PairwiseEvidenceStatus.CERTIFIED
    assert zero.signed_similarity == 0.0
    assert zero.pair_count == 40

    reverse = provider.create_pairwise_evidence(
        _qe(value=-0.99), expected_factor_ids=("FX", "F1"),
        expected_window_ref="250d", expected_universe_ref="ASHARE",
    )
    assert reverse.signed_similarity == -0.99


def test_real_qe_adapter_keeps_unknown_and_rejects_incomparable_identity():
    provider = QEEvidenceProvider()
    unknown = _qe(value=None, status=PairwiseMeasurementStatus.INSUFFICIENT_DATA,
                  count=1)
    adapted = provider.create_pairwise_evidence(
        unknown, expected_factor_ids=("FX", "F1"),
        expected_window_ref="250d", expected_universe_ref="ASHARE",
    )
    assert adapted.status is PairwiseEvidenceStatus.UNMEASURED
    assert adapted.signed_similarity is None
    with pytest.raises(ValueError, match="window_ref is not comparable"):
        provider.create_pairwise_evidence(
            _qe(), expected_factor_ids=("FX", "F1"),
            expected_window_ref="60d", expected_universe_ref="ASHARE",
        )
    with pytest.raises(ValueError, match="factor IDs"):
        provider.create_pairwise_evidence(
            _qe(), expected_factor_ids=("FX", "OTHER"),
            expected_window_ref="250d", expected_universe_ref="ASHARE",
        )


def test_legacy_qe_similarity_consumer_refuses_aggregate_only_bypass():
    consumer = QEPairwiseSimilarity()
    legacy = consumer._convert_qe_result({
        "factor_id_a": "FX", "factor_id_b": "F1", "correlation": 0.99,
        "sample_size": 1000, "timestamp": "t", "universe_ref": "ASHARE",
        "period_start": "a", "period_end": "b",
    }, SimilarityMethod.PEARSON)
    assert legacy.measurement_status is SimilarityMeasurementStatus.UNKNOWN
    assert legacy.similarity_score is None

    window = PairwiseWindowEvidence(
        "250d", -0.95, PairwiseMeasurementStatus.COMPUTED, 100, 25,
        (4,) * 25, (-0.99, -0.90),
    )
    typed = PairwiseCorrelationArtifact(
        "FX", "F1", -0.95, PairwiseMeasurementStatus.COMPUTED,
        100, 25, (4,) * 25, "pearson", "250d", "ASHARE", "S", (window,),
    )
    converted = consumer._convert_qe_result(typed, SimilarityMethod.PEARSON)
    assert converted.measurement_status is SimilarityMeasurementStatus.COMPUTED_VALUE
    assert converted.similarity_score == -0.95
