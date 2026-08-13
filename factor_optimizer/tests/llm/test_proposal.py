"""Tests for LLM proposal generator."""

from datetime import datetime

from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from factor_optimizer.llm.proposal import (
    ProposalGenerator,
    ProposalRequest,
    ProposalResponse,
)
from factor_optimizer.llm.prompts import PromptRegistry, PromptTemplate, create_default_registry
from factor_optimizer.llm.records import RecordStore


def test_proposal_request_creation():
    """Test basic proposal request creation."""
    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001", "type": "momentum"},
        performance_metrics={"sharpe": 1.2, "ic": 0.05},
        search_objective="maximize sharpe ratio",
        available_mutations=["parameter_tune", "operator_swap"],
        context={"budget": "low"},
    )

    assert request.parent_factor_info["factor_id"] == "f_001"
    assert request.search_objective == "maximize sharpe ratio"
    assert len(request.available_mutations) == 2
    assert request.context["budget"] == "low"


def test_proposal_request_serialization():
    """Test request serialization."""
    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    data = request.to_dict()
    assert data["parent_factor_info"]["factor_id"] == "f_001"
    assert data["performance_metrics"]["sharpe"] == 1.2


def test_proposal_response_validation():
    """Test proposal response validation."""
    from factor_optimizer.llm.records import LLMRecord

    # Valid response with proposals
    valid_response = ProposalResponse(
        request_id="req_001",
        proposals=[
            CandidateMutation(
                mutation_id="m_001",
                mutation_spec_version="1.0.0",
                parent_factor_ids=["f_001"],
                mutation_type="test",
                parameters={},
            )
        ],
        llm_record=LLMRecord(
            record_id="rec_001",
            timestamp=datetime.now(),
            model="test",
            model_version=None,
            prompt_template_id="test",
            prompt_template_version="1.0.0",
            prompt_hash="hash",
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            response_hash="hash",
        ),
        raw_response="{}",
        validation_errors=[],
    )

    assert valid_response.is_valid()

    # Invalid response with errors
    invalid_response = ProposalResponse(
        request_id="req_002",
        proposals=[],
        llm_record=valid_response.llm_record,
        raw_response="{}",
        validation_errors=["Error 1"],
    )

    assert not invalid_response.is_valid()


def test_proposal_generator_initialization():
    """Test proposal generator initialization."""
    generator = ProposalGenerator()

    assert generator.prompt_registry is not None
    assert generator.record_store is not None
    assert generator.model == "mock-model"


def test_proposal_generator_custom_initialization():
    """Test proposal generator with custom components."""
    registry = PromptRegistry()
    store = RecordStore()

    generator = ProposalGenerator(
        prompt_registry=registry,
        record_store=store,
        model="custom-model",
    )

    assert generator.prompt_registry is registry
    assert generator.record_store is store
    assert generator.model == "custom-model"


def test_generate_proposals_basic():
    """Test basic proposal generation."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001", "description": "momentum factor"},
        performance_metrics={"sharpe": 1.2, "ic": 0.05},
        search_objective="maximize sharpe ratio",
        available_mutations=["parameter_tune", "operator_swap"],
    )

    response = generator.generate_proposals(
        request=request,
        parent_factor_ids=["f_001"],
    )

    # Check response structure
    assert response.request_id is not None
    assert len(response.proposals) > 0
    assert response.llm_record is not None
    assert response.raw_response is not None

    # Check that proposals are valid CandidateMutation objects
    for proposal in response.proposals:
        assert isinstance(proposal, CandidateMutation)
        assert proposal.mutation_id is not None
        assert proposal.mutation_type in request.available_mutations
        assert proposal.parent_factor_ids == ["f_001"]


def test_generate_proposals_with_template_version():
    """Test proposal generation with specific template version."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    response = generator.generate_proposals(
        request=request,
        template_id="mutation_proposal",
        template_version="1.0.0",
    )

    assert response.llm_record.prompt_template_id == "mutation_proposal"
    assert response.llm_record.prompt_template_version == "1.0.0"


def test_generate_proposals_records_llm_call():
    """Test that proposal generation records the LLM call."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    # Initially no records
    assert len(generator.get_records()) == 0

    response = generator.generate_proposals(request=request)

    # Should have one record now
    records = generator.get_records()
    assert len(records) == 1

    record = records[0]
    assert record.model == "mock-model"
    assert record.prompt_template_id == "mutation_proposal"
    assert record.prompt_hash is not None
    assert record.response_hash is not None


def test_generate_proposals_token_tracking():
    """Test token usage tracking."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    # Initially zero tokens
    usage = generator.get_token_usage()
    assert usage["total_tokens"] == 0

    # Generate proposals
    generator.generate_proposals(request=request)

    # Should have token usage now
    usage = generator.get_token_usage()
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] > 0


def test_generate_proposals_multiple_calls():
    """Test multiple proposal generation calls."""
    generator = ProposalGenerator()

    request1 = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test 1",
        available_mutations=["mutation_1"],
    )

    request2 = ProposalRequest(
        parent_factor_info={"factor_id": "f_002"},
        performance_metrics={"sharpe": 1.5},
        search_objective="test 2",
        available_mutations=["mutation_2"],
    )

    response1 = generator.generate_proposals(request=request1)
    response2 = generator.generate_proposals(request=request2)

    # Should have unique request IDs
    assert response1.request_id != response2.request_id

    # Should have two records
    records = generator.get_records()
    assert len(records) == 2


def test_generate_proposals_with_multiple_mutations():
    """Test proposal generation with multiple available mutations."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1", "mutation_2", "mutation_3"],
    )

    response = generator.generate_proposals(request=request)

    # Mock implementation generates up to 2 proposals
    assert len(response.proposals) <= 2

    # Each proposal should use one of the available mutations
    for proposal in response.proposals:
        assert proposal.mutation_type in request.available_mutations


def test_generate_proposals_hypothesis_and_signatures():
    """Test that proposals include hypothesis and expected signatures."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="maximize sharpe ratio",
        available_mutations=["parameter_tune"],
    )

    response = generator.generate_proposals(request=request)

    # Check that proposals have hypotheses and signatures
    for proposal in response.proposals:
        assert proposal.mechanism_hypothesis is not None
        assert len(proposal.mechanism_hypothesis) > 0
        assert isinstance(proposal.expected_signatures, list)


def test_generate_proposals_provenance():
    """Test that proposals have proper provenance metadata."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    response = generator.generate_proposals(request=request)

    for proposal in response.proposals:
        assert proposal.producer == "llm_proposal_generator"
        assert proposal.producer_version == "0.1.0"
        assert proposal.created_at is not None
        assert isinstance(proposal.created_at, datetime)


def test_generate_proposals_latency_measurement():
    """Test that LLM call latency is measured."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    response = generator.generate_proposals(request=request)

    # Should have latency recorded
    assert response.llm_record.latency_ms is not None
    assert response.llm_record.latency_ms >= 0


def test_generate_proposals_hash_consistency():
    """Test that identical prompts produce identical hashes."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    response1 = generator.generate_proposals(request=request)
    response2 = generator.generate_proposals(request=request)

    # Same request should produce same prompt hash
    assert response1.llm_record.prompt_hash == response2.llm_record.prompt_hash

    # But different response hashes (mock generates deterministic but unique responses)
    # Note: In this mock implementation, responses are actually identical
    # In real implementation with actual LLM, responses would differ


def test_mock_llm_call_deterministic():
    """Test that mock LLM call is deterministic for same input."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1", "mutation_2"],
    )

    response1 = generator.generate_proposals(request=request)
    response2 = generator.generate_proposals(request=request)

    # Mock should produce identical raw responses
    assert response1.raw_response == response2.raw_response


def test_generate_proposals_with_empty_mutations():
    """Test proposal generation with empty mutation list."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=[],  # Empty list
    )

    response = generator.generate_proposals(request=request)

    # Should still succeed but produce no proposals
    assert len(response.proposals) == 0


def test_generate_proposals_default_parent_factor_ids():
    """Test that default parent factor IDs are used when not provided."""
    generator = ProposalGenerator()

    request = ProposalRequest(
        parent_factor_info={"factor_id": "f_001"},
        performance_metrics={"sharpe": 1.2},
        search_objective="test",
        available_mutations=["mutation_1"],
    )

    response = generator.generate_proposals(request=request)  # No parent_factor_ids

    for proposal in response.proposals:
        assert proposal.parent_factor_ids == ["unknown_factor"]
