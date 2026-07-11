# -*- coding: utf-8
"""Evidence schema v2：参数域 supported_calls。"""
from __future__ import annotations

from backend.operator_evidence_schema import (
    evidence_schema_version,
    operator_evidence_record,
    parameter_domain_verified,
    supported_calls_match,
)
from planner.logical_plan import PlanNode


def test_evidence_schema_version_is_v2():
    assert evidence_schema_version() >= 2


def test_scale_supported_calls_to_one_only():
    rec = operator_evidence_record("scale")
    assert rec is not None
    assert rec["supported_calls"]["to"] == [1]


def test_scale_to_two_not_in_parameter_domain():
    node = PlanNode(
        op="scale",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 2.0}, inputs=[]),
        ],
    )
    assert not supported_calls_match(node, operator_evidence_record("scale")["supported_calls"])
    assert parameter_domain_verified("scale", node) is False


def test_cs_regression_modes_in_domain():
    for mode in (0, 1, 2):
        node = PlanNode(
            op="cs_regression",
            inputs=[
                PlanNode(op="column", attrs={"name": "y"}, inputs=[]),
                PlanNode(op="column", attrs={"name": "x"}, inputs=[]),
                PlanNode(op="literal", attrs={"value": mode}, inputs=[]),
            ],
        )
        assert parameter_domain_verified("cs_regression", node)
