"""
QE-P0-2: reporting layer must not recompute metrics inline.

The tear-sheet previously inlined ``np.nanmean(quantile_returns, axis=0)``
(time-averaging quantile returns) inside the renderer.  Reporting may only
render pre-computed values; any aggregation must live in the kernel layer and
be delegated to, never recomputed inline.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant_evaluator.reporting.tear_sheet as tear_sheet
from quant_evaluator.reporting.tear_sheet import EvaluationResult, generate_tear_sheet


def test_tear_sheet_does_not_inline_nanmean():
    """The tear-sheet source must not inline a quantile-returns aggregation."""
    src = open(tear_sheet.__file__).read()
    assert "np.nanmean(evaluation_result.quantile_returns" not in src, (
        "reporting layer must not recompute quantile returns inline"
    )


def test_quantile_returns_bar_renders_precomputed_mean():
    """The quantile-returns panel renders a pre-computed mean, not a recompute."""
    import numpy as np

    result = EvaluationResult()
    # Pre-computed time-averaged quantile returns (n_quantiles, F).
    result.quantile_returns = np.array([[-0.02, -0.01], [0.0, 0.01], [0.02, 0.03]])
    result.quantile_names = ["Q1", "Q2", "Q3"]
    panels = generate_tear_sheet(result)
    spec = panels["quantile_returns_bar"]
    assert spec.data["values"] == [[-0.02, -0.01], [0.0, 0.01], [0.02, 0.03]]
