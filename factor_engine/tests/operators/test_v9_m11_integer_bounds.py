"""Receipt bounds apply to scalar integers as well as containers and text."""
from factor_engine.cleaned_operators.ts_model import _rolling_core as rc


def test_huge_integer_diagnostic_values_and_keys_are_not_retained():
    huge = 1 << 100000
    status = rc.FitStatus(False, "singular", (("value", huge), (huge, 2)))
    details = dict(status.details)
    assert details["value"] == "<oversized_integer>"
    assert details["<oversized_integer>"] == 2
    assert len(repr(status)) < 512


def test_huge_integer_scope_cannot_create_owned_receipt():
    huge = 1 << 100000
    scope = rc.FitScope(canonical="op", instrument=huge, window_start=huge)
    with rc.fit_receipt_scope(scope):
        sanitized = rc.current_fit_scope()
        assert sanitized.instrument is None
        assert sanitized.window_start is None
        assert sanitized.scope_kind == "kernel_only"


def test_normal_integer_diagnostics_and_nanosecond_coordinates_are_preserved():
    assert dict(rc.FitStatus(False, "singular", (("count", 123),)).details) == {"count": 123}
    for value in (0, -1, 2**63 - 1, -(2**63)):
        with rc.fit_receipt_scope(rc.FitScope(window_start=value)):
            assert rc.current_fit_scope().window_start == value
