"""
QE-P0-04: coverage compiler must not count comment/docstring mentions as hits.

The compiler's ``_mentioned`` helper treats ANY substring occurrence of a
metric_id / function name as a test/report/artifact binding — including inside
comments, docstrings, and dead code.  That makes the READY gate a false
positive.  A comment-only mention must NOT count as a real binding.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_evaluator.metrics.coverage_compiler import _mentioned


def test_comment_only_mention_is_not_a_binding():
    """A metric_id appearing only in a comment must NOT count as a hit."""
    text = "# pearson_ic is computed by compute_daily_ic (see docs)"
    assert _mentioned(text, "pearson_ic") is False, (
        "comment-only mention must not count as a real binding"
    )
    assert _mentioned(text, "compute_daily_ic") is False


def test_docstring_only_mention_is_not_a_binding():
    """A metric_id appearing only in a docstring must NOT count as a hit."""
    text = '"""pearson_ic: Pearson correlation between factor and returns."""'
    assert _mentioned(text, "pearson_ic") is False


def test_real_code_mention_is_a_binding():
    """A metric_id in actual executable code still counts as a hit."""
    text = 'result = get_metric("pearson_ic")'
    assert _mentioned(text, "pearson_ic") is True
