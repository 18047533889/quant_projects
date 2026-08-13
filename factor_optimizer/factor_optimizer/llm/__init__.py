"""
LLM-powered mutation proposal generation.

Core components:
    - proposal: Structured proposal generator with schema validation
    - prompts: Versioned prompt templates
    - records: Model/prompt/hash recording for reproducibility
"""

from factor_optimizer.llm.proposal import ProposalGenerator, ProposalRequest, ProposalResponse
from factor_optimizer.llm.prompts import PromptTemplate, PromptRegistry
from factor_optimizer.llm.records import LLMRecord, RecordStore

__all__ = [
    "ProposalGenerator",
    "ProposalRequest",
    "ProposalResponse",
    "PromptTemplate",
    "PromptRegistry",
    "LLMRecord",
    "RecordStore",
]
