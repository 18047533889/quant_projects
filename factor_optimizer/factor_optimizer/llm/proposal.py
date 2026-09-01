"""Structured LLM proposal generator with schema validation."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from factor_optimizer.llm.prompts import PromptRegistry, PromptTemplate, create_default_registry
from factor_optimizer.llm.records import LLMRecord, RecordStore, compute_hash


@dataclass
class ProposalRequest:
    """
    Request for LLM-generated mutation proposals.

    Attributes:
        parent_factor_info: Information about the parent factor(s)
        performance_metrics: Current performance data
        search_objective: What we're trying to optimize for
        available_mutations: List of legal mutation types
        context: Additional context for the LLM
    """

    parent_factor_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    search_objective: str
    available_mutations: List[str]
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "parent_factor_info": self.parent_factor_info,
            "performance_metrics": self.performance_metrics,
            "search_objective": self.search_objective,
            "available_mutations": self.available_mutations,
            "context": self.context,
        }


@dataclass
class ProposalResponse:
    """
    Response from LLM proposal generation.

    Attributes:
        request_id: Unique identifier for this request
        proposals: List of generated candidate mutations
        llm_record: Record of the LLM call for reproducibility
        raw_response: Raw LLM response text
        validation_errors: Any errors during validation
    """

    request_id: str
    proposals: List[CandidateMutation]
    llm_record: LLMRecord
    raw_response: str
    validation_errors: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if response has valid proposals."""
        return len(self.proposals) > 0 and len(self.validation_errors) == 0


class ProposalGenerator:
    """
    Generate structured mutation proposals using LLMs.

    This is a mock implementation that doesn't make real LLM calls.
    Production usage would integrate with actual LLM APIs.
    """

    def __init__(
        self,
        prompt_registry: Optional[PromptRegistry] = None,
        record_store: Optional[RecordStore] = None,
        model: str = "mock-model",
    ):
        """
        Initialize the proposal generator.

        Args:
            prompt_registry: Registry of prompt templates (uses default if None)
            record_store: Store for LLM call records (creates new if None)
            model: Model identifier to use
        """
        self.prompt_registry = prompt_registry or create_default_registry()
        self.record_store = record_store or RecordStore()
        self.model = model

    def generate_proposals(
        self,
        request: ProposalRequest,
        template_id: str = "mutation_proposal",
        template_version: Optional[str] = None,
        parent_factor_ids: Optional[List[str]] = None,
    ) -> ProposalResponse:
        """
        Generate mutation proposals for the given request.

        Args:
            request: The proposal request
            template_id: ID of the prompt template to use
            template_version: Specific template version (latest if None)
            parent_factor_ids: Factor IDs being mutated

        Returns:
            ProposalResponse with generated proposals and LLM record
        """
        request_id = str(uuid.uuid4())
        parent_factor_ids = parent_factor_ids or ["unknown_factor"]

        # Get the prompt template
        template = self.prompt_registry.get(template_id, template_version)

        # Render the prompt
        system_prompt, user_prompt = template.get_full_prompt(
            parent_factor_info=json.dumps(request.parent_factor_info, indent=2),
            performance_metrics=json.dumps(request.performance_metrics, indent=2),
            search_objective=request.search_objective,
            available_mutations=", ".join(request.available_mutations),
        )

        # Compute prompt hash
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        prompt_hash = compute_hash(full_prompt)

        # Mock LLM call (real implementation would call actual API)
        start_time = datetime.now(timezone.utc)
        raw_response = self._mock_llm_call(request, template)
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Compute response hash
        response_hash = compute_hash(raw_response)

        # Parse and validate response
        proposals, validation_errors = self._parse_and_validate(
            raw_response, parent_factor_ids, template
        )

        # Create LLM record
        llm_record = LLMRecord(
            record_id=str(uuid.uuid4()),
            timestamp=start_time,
            model=self.model,
            model_version=None,
            prompt_template_id=template.template_id,
            prompt_template_version=template.version,
            prompt_hash=prompt_hash,
            input_tokens=len(full_prompt.split()),  # Mock token count
            output_tokens=len(raw_response.split()),  # Mock token count
            latency_ms=latency_ms,
            response_hash=response_hash,
            metadata=template.model_hints,
        )

        # Store the record
        self.record_store.add(llm_record)

        return ProposalResponse(
            request_id=request_id,
            proposals=proposals,
            llm_record=llm_record,
            raw_response=raw_response,
            validation_errors=validation_errors,
        )

    def _mock_llm_call(self, request: ProposalRequest, template: PromptTemplate) -> str:
        """
        Mock LLM call that returns a deterministic response.

        Real implementation would call actual LLM API.
        """
        # Generate a mock response based on available mutations
        mock_proposals = []
        for i, mutation_type in enumerate(request.available_mutations[:2], 1):
            proposal = {
                "mutation_type": mutation_type,
                "parameters": {"mock_param": f"value_{i}"},
                "mechanism_hypothesis": f"Hypothesis {i}: This mutation might improve {request.search_objective}",
                "expected_signatures": [f"improved_{request.search_objective}"],
            }
            mock_proposals.append(proposal)

        return json.dumps({"proposals": mock_proposals}, indent=2)

    def _parse_and_validate(
        self,
        raw_response: str,
        parent_factor_ids: List[str],
        template: PromptTemplate,
    ) -> tuple[List[CandidateMutation], List[str]]:
        """
        Parse LLM response and validate against schema.

        Returns:
            (proposals, validation_errors)
        """
        validation_errors = []
        proposals = []

        try:
            # Parse JSON
            response_data = json.loads(raw_response)

            # Basic schema validation (real implementation would use jsonschema)
            if "proposals" not in response_data:
                validation_errors.append("Missing 'proposals' field in response")
                return proposals, validation_errors

            if not isinstance(response_data["proposals"], list):
                validation_errors.append("'proposals' must be an array")
                return proposals, validation_errors

            # Convert to CandidateMutation objects
            for i, proposal_data in enumerate(response_data["proposals"]):
                try:
                    mutation = CandidateMutation(
                        mutation_id=str(uuid.uuid4()),
                        mutation_spec_version="1.0.0",
                        parent_factor_ids=parent_factor_ids,
                        mutation_type=proposal_data["mutation_type"],
                        parameters=proposal_data["parameters"],
                        mechanism_hypothesis=proposal_data.get("mechanism_hypothesis"),
                        expected_signatures=proposal_data.get("expected_signatures", []),
                        created_at=datetime.now(timezone.utc),
                        producer="llm_proposal_generator",
                        producer_version="0.1.0",
                    )
                    proposals.append(mutation)
                except (TypeError, ValueError, KeyError, AttributeError) as e:
                    validation_errors.append(f"Proposal {i}: {str(e)}")

        except json.JSONDecodeError as e:
            validation_errors.append(f"Invalid JSON: {str(e)}")
        except (TypeError, ValueError, KeyError, RuntimeError, OSError) as e:
            validation_errors.append(f"Unexpected error: {str(e)}")

        return proposals, validation_errors

    def get_records(self) -> List[LLMRecord]:
        """Get all LLM call records."""
        return self.record_store.list_all()

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage statistics."""
        return self.record_store.total_tokens()
