"""Warning regressions for book-tail scalar and degenerate-beta paths."""

import warnings

import numpy as np
import pytest

from quant_evaluator.contracts.evidence_status import EvidenceStatus
from quant_evaluator.metrics.extension.book_risk import (
    book_conditional_loss,
    book_downside_beta,
    book_tail_loss_delta,
)


@pytest.mark.parametrize("backend", ["cpu_reference", "cpu_fast"])
def test_scalar_book_tail_has_no_numpy_conversion_warning(backend):
    book = np.array([-0.3, -0.2, -0.1, 0.0])
    candidate = np.array([0.1, 0.4, 0.8, 1.6])
    kwargs = dict(
        backend=backend, tail_confidence=0.625,
        min_tail_mass=0.1, min_periods=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for kernel in (book_conditional_loss, book_tail_loss_delta):
            result = kernel(book, candidate, **kwargs)
            assert result["status"] is EvidenceStatus.COMPUTED
            assert isinstance(result["value"], float)


@pytest.mark.parametrize("backend", ["cpu_reference", "cpu_fast"])
def test_degenerate_book_beta_fails_closed_without_divide_warning(backend):
    book = np.full(200, 0.0005)
    candidate = np.column_stack((np.full(200, 0.001), np.full(200, -0.002)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = book_downside_beta(
            book, candidate, backend=backend,
            min_tail_mass=1.0, min_periods=10,
        )
    assert result["status"] is EvidenceStatus.INVALID_EVIDENCE
    assert result["diagnostics"]["reason_detail"] == (
        "baseline_weighted_variance_degenerate"
    )
    assert result["diagnostics"]["degenerate_columns"] == [0, 1]
