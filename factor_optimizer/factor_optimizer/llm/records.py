"""LLM call recording for reproducibility and audit."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMRecord:
    """
    Record of a single LLM API call for reproducibility.

    Attributes:
        record_id: Unique identifier for this record
        timestamp: When the call was made
        model: Model identifier (e.g., "claude-opus-4", "gpt-4")
        model_version: Specific version/snapshot if available
        prompt_template_id: ID of the prompt template used
        prompt_template_version: Version of the prompt template
        prompt_hash: SHA256 hash of the final rendered prompt
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: API call latency in milliseconds
        response_hash: SHA256 hash of the response text
        metadata: Additional context (temperature, top_p, etc.)
    """

    record_id: str
    timestamp: datetime
    model: str
    model_version: Optional[str]
    prompt_template_id: str
    prompt_template_version: str
    prompt_hash: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: Optional[float]
    response_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields."""
        if not self.record_id:
            raise ValueError("record_id is required")
        if not self.model:
            raise ValueError("model is required")
        if not self.prompt_template_id:
            raise ValueError("prompt_template_id is required")
        if not self.prompt_hash:
            raise ValueError("prompt_hash is required")
        if not self.response_hash:
            raise ValueError("response_hash is required")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "model_version": self.model_version,
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_version": self.prompt_template_version,
            "prompt_hash": self.prompt_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "response_hash": self.response_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMRecord":
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RecordStore:
    """
    In-memory store for LLM call records.

    Production usage would persist to disk/database.
    """

    def __init__(self):
        self._records: List[LLMRecord] = []
        self._index_by_id: Dict[str, LLMRecord] = {}

    def add(self, record: LLMRecord) -> None:
        """Add a record to the store."""
        if record.record_id in self._index_by_id:
            raise ValueError(f"Duplicate record_id: {record.record_id}")
        self._records.append(record)
        self._index_by_id[record.record_id] = record

    def get(self, record_id: str) -> Optional[LLMRecord]:
        """Retrieve a record by ID."""
        return self._index_by_id.get(record_id)

    def list_all(self) -> List[LLMRecord]:
        """List all records in chronological order."""
        return list(self._records)

    def list_by_model(self, model: str) -> List[LLMRecord]:
        """List records for a specific model."""
        return [r for r in self._records if r.model == model]

    def list_by_template(self, template_id: str) -> List[LLMRecord]:
        """List records for a specific prompt template."""
        return [r for r in self._records if r.prompt_template_id == template_id]

    def total_tokens(self) -> Dict[str, int]:
        """Calculate total token usage."""
        total_input = sum(r.input_tokens or 0 for r in self._records)
        total_output = sum(r.output_tokens or 0 for r in self._records)
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._index_by_id.clear()

    def export_json(self) -> str:
        """Export all records as JSON."""
        return json.dumps([r.to_dict() for r in self._records], indent=2)

    def import_json(self, json_str: str) -> int:
        """Import records from JSON. Returns count of imported records."""
        data = json.loads(json_str)
        count = 0
        for item in data:
            record = LLMRecord.from_dict(item)
            if record.record_id not in self._index_by_id:
                self.add(record)
                count += 1
        return count
