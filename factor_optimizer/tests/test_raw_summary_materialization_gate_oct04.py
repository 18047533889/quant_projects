"""Direct contract checks for refusing unusable optimizer materializations."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_HELPER_PATH = Path(__file__).with_name("test_raw_summary_cache_benchmark_oct04.py")
_SPEC = importlib.util.spec_from_file_location("raw_summary_benchmark_fixtures", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


@pytest.mark.parametrize("failure", ["materialization_error", "failed_status"])
def test_result_signature_rejects_materialization_failure_before_fingerprinting(failure):
    """A failed selected plan must not count as a successful A/B result."""
    batch, labels, _, _ = _HELPERS._cohort()
    result = _HELPERS._fake_result(
        batch, labels, _HELPERS.BENCH._default_config())
    factor = result.factors[batch.factor_ids[0]]
    if failure == "materialization_error":
        factor.materialization_error = "RuntimeError: selected plan could not execute"
    else:
        factor.status = "materialization_failed"

    automatic_audit = _HELPERS.BENCH._load_audit_modules()[0]
    with pytest.raises(ValueError, match="materialization failure"):
        _HELPERS.BENCH._result_signature(
            result, batch, result.split, automatic_audit)
