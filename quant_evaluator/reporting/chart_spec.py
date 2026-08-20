"""
Chart specification dataclass for quant_evaluator reporting.

Defines immutable structure for chart metadata, data, and content-based deduplication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import uuid
from datetime import datetime, timezone


RENDERER_VERSION = "1.0.0"
CHART_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """Immutable specification for a single chart/visualization.

    The content_hash covers all fields that define chart identity and content.
    Two ChartSpecs with the same content_hash are semantically identical.

    Attributes:
        chart_id: Unique identifier for this chart spec (UUID4)
        title: Chart title
        chart_type: Type of chart (line, bar, scatter, heatmap)
        x_label: X-axis label
        y_label: Y-axis label
        x_semantics: Semantic meaning of X axis (e.g., "time", "category", "value")
        y_semantics: Semantic meaning of Y axis (e.g., "value", "count", "percentage")
        renderer: Rendering backend identifier (e.g., "matplotlib", "plotly")
        renderer_version: Version of the renderer used
        source_artifact_refs: References to source artifacts that produced this chart
        parameters: Additional chart parameters (e.g., line width, color scheme)
        data: Dictionary containing series, timestamps, and other chart data
        chart_version: Version of this chart specification schema
        content_hash: SHA-256 hash of all identity fields for deduplication
        created_at: UTC ISO-8601 timestamp of creation
    """
    chart_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    chart_type: str = ""  # line, bar, scatter, heatmap
    x_label: str = ""
    y_label: str = ""
    x_semantics: str = "value"
    y_semantics: str = "value"
    renderer: str = "matplotlib"
    renderer_version: str = RENDERER_VERSION
    source_artifact_refs: Tuple[str, ...] = ()
    parameters: Tuple[Tuple[str, Any], ...] = ()
    data: Dict[str, Any] = field(default_factory=dict)
    chart_version: str = CHART_VERSION
    content_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        """Compute content hash after initialization if not provided."""
        # frozen=True means we cannot use object.__setattr__ directly.
        # We need to use the __init_subclass__ or re-init pattern.
        # For frozen dataclasses, we use object.__setattr__ in a controlled way.
        if not self.content_hash:
            computed = self._compute_hash()
            object.__setattr__(self, "content_hash", computed)

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of all identity fields.

        This covers:
        - title, chart_type, x_label, y_label
        - x_semantics, y_semantics
        - renderer, renderer_version
        - source_artifact_refs
        - parameters
        - chart_version
        - data

        Returns:
            Hex digest of SHA-256 hash
        """
        identity = {
            "title": self.title,
            "chart_type": self.chart_type,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_semantics": self.x_semantics,
            "y_semantics": self.y_semantics,
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "source_artifact_refs": self.source_artifact_refs,
            "parameters": self.parameters,
            "chart_version": self.chart_version,
            "data": self.data,
        }
        serialized = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert ChartSpec to dictionary.

        Returns:
            Dictionary representation of the chart spec
        """
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "chart_type": self.chart_type,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_semantics": self.x_semantics,
            "y_semantics": self.y_semantics,
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "source_artifact_refs": list(self.source_artifact_refs),
            "parameters": list(self.parameters),
            "data": self.data,
            "chart_version": self.chart_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChartSpec":
        """Create ChartSpec from dictionary.

        Args:
            d: Dictionary containing chart spec fields

        Returns:
            ChartSpec instance
        """
        src_refs = d.get("source_artifact_refs", [])
        params = d.get("parameters", [])
        # Convert list of lists to tuples for immutability
        if params and isinstance(params[0], list):
            params = [tuple(p) for p in params]

        return cls(
            chart_id=d.get("chart_id", str(uuid.uuid4())),
            title=d.get("title", ""),
            chart_type=d.get("chart_type", ""),
            x_label=d.get("x_label", ""),
            y_label=d.get("y_label", ""),
            x_semantics=d.get("x_semantics", "value"),
            y_semantics=d.get("y_semantics", "value"),
            renderer=d.get("renderer", "matplotlib"),
            renderer_version=d.get("renderer_version", RENDERER_VERSION),
            source_artifact_refs=tuple(src_refs),
            parameters=tuple(tuple(p) if isinstance(p, list) else p for p in params),
            data=d.get("data", {}),
            chart_version=d.get("chart_version", CHART_VERSION),
            content_hash=d.get("content_hash", ""),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChartSpec":
        """Create ChartSpec from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ChartSpec instance
        """
        d = json.loads(json_str)
        return cls.from_dict(d)

    def to_json(self, indent: int = 2) -> str:
        """Convert ChartSpec to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def replace(self, **changes: Any) -> "ChartSpec":
        """Create a new ChartSpec with the given fields replaced.

        This is the canonical way to "mutate" a frozen dataclass.

        Args:
            **changes: Fields to replace

        Returns:
            New ChartSpec instance with updated fields
        """
        # Build new dict from current state, apply changes
        current = self.to_dict()
        current.update(changes)
        # Force recompute of content_hash
        current["content_hash"] = ""
        return self.from_dict(current)
