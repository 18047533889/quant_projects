import pytest
from factor_assets.library import CandidateEvaluationRef, PromotionGate, PromotionDecision, PromotionReasonCode


@pytest.mark.parametrize('fields,reason', [
    ({'label_maturity': False}, PromotionReasonCode.LABEL_NOT_MATURE),
    ({'return_basis': 'close_to_close'}, PromotionReasonCode.RETURN_BASIS_WRONG),
    ({'rank_ic': .001}, PromotionReasonCode.RANK_IC_BELOW_THRESHOLD),
])
def test_soft_duplicate_does_not_override_hard_rejection(fields, reason):
    candidate = CandidateEvaluationRef(**({'candidate_ref': 'candidate', 'rank_ic': .1} | fields))
    decision = PromotionGate(reject_duplicates=False).evaluate('library', candidate, library_member_refs=['existing'], similarity_fn=lambda a, b: .9)
    assert decision.decision is PromotionDecision.REJECT
    assert reason in decision.reason_codes


@pytest.mark.parametrize('maturity', ['false', 'true', 0, 1, None])
def test_deserialization_does_not_coerce_maturity(maturity):
    with pytest.raises(TypeError, match='label_maturity'):
        CandidateEvaluationRef.from_dict({'candidate_ref': 'candidate', 'rank_ic': .1, 'label_maturity': maturity})


@pytest.mark.parametrize('basis', ['', None, 0])
def test_deserialization_does_not_invent_return_basis(basis):
    with pytest.raises(ValueError, match='return_basis'):
        CandidateEvaluationRef.from_dict({'candidate_ref': 'candidate', 'return_basis': basis})


def test_valid_candidate_mapping_still_uses_omitted_defaults():
    candidate = CandidateEvaluationRef.from_dict({'candidate_ref': 'candidate', 'rank_ic': .1, 'label_maturity': False})
    assert candidate.label_maturity is False
    assert candidate.return_basis == 'vwap_to_vwap'
