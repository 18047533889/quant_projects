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


def test_extended_scale_has_no_production_evidence():
    assert operator_evidence_record("scale") is None


def test_scale_is_fail_closed_for_production():
    node = PlanNode(
        op="scale",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 2.0}, inputs=[]),
        ],
    )
    assert parameter_domain_verified("scale", node, production=True) is False


def test_supported_calls_helper_rejects_out_of_domain_literal():
    node = PlanNode(
        op="scale",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 2.0}, inputs=[]),
        ],
    )
    assert not supported_calls_match(node, {"to": [1]})
