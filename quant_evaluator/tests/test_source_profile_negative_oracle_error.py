"""Reject negative absolute errors in source-profile oracle receipts."""
import json

import pytest

from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report
from quant_evaluator.tests.test_source_profile_report_reader_oct04 import (
    _payload as _pearson_v1_payload,
    _payload_v2 as _pearson_v2_payload,
)
from quant_evaluator.tests.test_source_linear_shape_report_reader_oct04 import (
    _payload as _linear_shape_payload,
)


_FIXTURES = (
    ("pearson_v1", _pearson_v1_payload, False),
    ("pearson_v2", _pearson_v2_payload, True),
    ("linear_shape", _linear_shape_payload, True),
)
_NEGATIVE_ERRORS = (-1.0, -5e-324)
_NONNEGATIVE_ERRORS = (0.0, -0.0)


def _parse(payload):
    return parse_source_profile_report(json.dumps(payload, allow_nan=False))


def _oracle_metric(payload, section, run_index, metric):
    if section == "oracle_reports":
        return payload[section][run_index]["metrics"][metric]
    return payload[section]["oracle_report"]["metrics"][metric]


@pytest.mark.parametrize("name,payload_factory,has_default_auto", _FIXTURES, ids=[
    "pearson-v1", "pearson-v2", "linear-shape",
])
@pytest.mark.parametrize("section,run_index", (
    ("oracle_reports", 0), ("oracle_reports", 1),
    ("oracle_reports", 2), ("oracle_reports", 3),
    ("auto_verification", None), ("default_auto_verification", None),
))
@pytest.mark.parametrize("error", _NONNEGATIVE_ERRORS, ids=["zero", "negative-zero"])
def test_existing_f48_reports_accept_nonnegative_oracle_errors(
        name, payload_factory, has_default_auto, section, run_index, error):
    # Defect guarded against: rejecting exact matches or signed zero as errors.
    if section == "default_auto_verification" and not has_default_auto:
        pytest.skip("v1 has no default-auto receipt")
    payload = payload_factory()
    metric = payload["metric_ids"][0]
    _oracle_metric(payload, section, run_index, metric)["max_abs_error"] = error
    assert _parse(payload).kind == payload["kind"]


@pytest.mark.parametrize("name,payload_factory,has_default_auto", _FIXTURES, ids=[
    "pearson-v1", "pearson-v2", "linear-shape",
])
@pytest.mark.parametrize("section,run_index", (
    ("oracle_reports", 0), ("oracle_reports", 1),
    ("oracle_reports", 2), ("oracle_reports", 3),
    ("auto_verification", None), ("default_auto_verification", None),
))
@pytest.mark.parametrize("error", _NEGATIVE_ERRORS, ids=["negative-one", "negative-subnormal"])
def test_all_oracle_receipts_reject_negative_max_abs_error(
        name, payload_factory, has_default_auto, section, run_index, error):
    # Defect guarded against: any oracle path accepting a negative absolute error.
    if section == "default_auto_verification" and not has_default_auto:
        pytest.skip("v1 has no default-auto receipt")
    payload = payload_factory()
    metric = payload["metric_ids"][0]
    _oracle_metric(payload, section, run_index, metric)["max_abs_error"] = error
    with pytest.raises(ValueError):
        _parse(payload)
