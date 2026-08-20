"""
Chart specification dataclass for quant_evaluator reporting.

Defines the structure for chart metadata, data, and content-based deduplication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import hashlib
import json


@dataclass
class ChartSpec:
    """Specification for a single chart/visualization.

    Attributes:
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        chart_type: Type of chart (line, bar, scatter, heatmap)
        data: Dictionary containing series, timestamps, and other chart data
        content_hash: SHA-256 hash of serialized data for deduplication
    """
    title: str
    x_label: str
    y_label: str
    chart_type: str  # line, bar, scatter, heatmap
    data: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        """Compute content hash after initialization if not provided."""
        if not self.content_hash:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of serialized data for content deduplication.

        Returns:
            Hex digest of SHA-256 hash
        """
        # Create a stable serialization of the data
        serialized = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert ChartSpec to dictionary.

        Returns:
            Dictionary representation of the chart spec
        """
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "chart_type": self.chart_type,
            "data": self.data,
            "content_hash": self.content_hash
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChartSpec":
        """Create ChartSpec from dictionary.

        Args:
            d: Dictionary containing chart spec fields

        Returns:
            ChartSpec instance
        """
        return cls(
            title=d["title"],
            x_label=d["x_label"],
            y_label=d["y_label"],
            chart_type=d["chart_type"],
            data=d.get("data", {}),
            content_hash=d.get("content_hash", "")
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
