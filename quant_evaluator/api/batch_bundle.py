"""Columnar batch evaluation bundle (spec §44).

The main return of ``evaluate_many`` is columnar, not a list of 500 Python
objects: ``scalar_metrics`` (F,), ``series_metrics`` (T,F), ``vector_metrics``
(Q,F).  ``for_factor`` projects one factor's view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class BatchEvaluationBundle:
    """Columnar batch evaluation result (spec §44)."""

    factor_ids: Tuple[str, ...]
    label_id: str
    scalar_metrics: Dict[str, np.ndarray] = field(default_factory=dict)   # (F,)
    series_metrics: Dict[str, np.ndarray] = field(default_factory=dict)   # (T,F)
    vector_metrics: Dict[str, np.ndarray] = field(default_factory=dict)   # (Q,F)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def for_factor(self, factor_id: str) -> Dict[str, Any]:
        """Project a single factor's scalar/vector view."""
        try:
            idx = self.factor_ids.index(factor_id)
        except ValueError:
            raise KeyError(factor_id) from None
        out: Dict[str, Any] = {}
        for k, v in self.scalar_metrics.items():
            out[k] = float(v[idx]) if v.ndim else float(v)
        for k, v in self.vector_metrics.items():
            out[k] = v[:, idx] if v.ndim == 2 else v
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_ids": list(self.factor_ids),
            "label_id": self.label_id,
            "scalar_metrics": {k: v.tolist() for k, v in self.scalar_metrics.items()},
            "vector_metrics": {k: v.tolist() for k, v in self.vector_metrics.items()},
            "metadata": self.metadata,
        }
