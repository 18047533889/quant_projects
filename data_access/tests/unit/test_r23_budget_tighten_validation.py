"""Budget tightening must validate before conversion hides bad requests."""
from decimal import Decimal
from fractions import Fraction

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.query_budget import QueryBudget


@pytest.mark.parametrize("field", QueryBudget._INT_FIELDS)
@pytest.mark.parametrize("bad", [True, False, 1.5, Decimal("1.5"), Fraction(3, 2), float("inf"), float("nan"), -1, 0])
def test_integer_tighten_rejects_invalid_original_values(field, bad):
    budget = QueryBudget(**{field: 100})
    with pytest.raises(ValidationError):
        budget.tighten(**{field: bad})
    assert getattr(budget, field) == 100


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -1, 0])
def test_elapsed_tighten_cannot_hide_invalid_values_with_min(bad):
    with pytest.raises(ValidationError):
        QueryBudget(max_elapsed_ms=100).tighten(max_elapsed_ms=bad)


@pytest.mark.parametrize("field", QueryBudget._BOOL_FIELDS)
@pytest.mark.parametrize("bad", ["false", "true", 0, 1, [], {}])
def test_boolean_tighten_does_not_apply_truthiness(field, bad):
    with pytest.raises(ValidationError):
        QueryBudget(**{field: True}).tighten(**{field: bad})


def test_valid_tightening_remains_monotone_and_preserves_other_limits():
    original = QueryBudget(max_rows=10, max_result_bytes=200, max_elapsed_ms=50,
                           require_columns=True, max_scan_files=5)
    actual = original.tighten(max_rows=20, max_result_bytes=100, max_elapsed_ms=25,
                              require_columns=False, require_time_range=True)
    assert actual == QueryBudget(max_rows=10, max_result_bytes=100, max_elapsed_ms=25,
                                 require_columns=True, require_time_range=True, max_scan_files=5)
    assert original.max_result_bytes == 200
    assert original.tighten(max_rows=None) == original
