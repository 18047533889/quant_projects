# -*- coding: utf-8 -*-
"""Round-14 P2-35: ParamRole split — estimator-resolution params separated
from economic/horizon search dimensions.

The kernel audits found that estimator-tuning knobs (``bins``, ``n_segments``,
kernel bandwidth method, Theiler window, BDS epsilon multiplier, embedding
resolution, …) were searched at the SAME resolution as genuine economic
parameters, so AlphaProbe/GP-style search could optimise for "which estimator
bias fits this sample" instead of "which real economic regularity exists".

This round adds a machine-readable ``ParamRole`` per parameter (never inferred
from names) plus a role-aware search grade, and verifies the default
(undeclared -> ECONOMIC for searchable / POLICY for non-searchable) keeps the
existing search space unchanged.
"""
from __future__ import annotations

import pytest

from cleaned_operators.base import (
    ParamRole,
    ParamSpec,
    effective_param_role,
    param_search_grade,
    searchable_param_names,
)


def test_paramrole_members():
    assert {r.value for r in ParamRole} == {
        "economic", "horizon", "state_threshold", "estimator_resolution",
        "numerical", "policy",
        # R14 concurrent-session additions (round-3 session coordination):
        "model_order", "missing_policy", "market_policy",
        "support_policy", "source_policy", "session_policy",
    }


def test_default_undeclared_role_preserves_search_space():
    # A searchable param with no declared role stays full-resolution ECONOMIC
    # (backwards compatible: the default grammar is unchanged).
    spec = ParamSpec(dtype=int, min=1, max=20, default=5)
    assert effective_param_role(spec) is ParamRole.ECONOMIC
    assert param_search_grade(spec) == "full"


def test_nonsearchable_defaults_to_policy():
    spec = ParamSpec(dtype=int, default=0, searchable=False)
    assert effective_param_role(spec) is ParamRole.POLICY
    assert param_search_grade(spec) == "excluded"


@pytest.mark.parametrize(
    "role,grade",
    [
        (ParamRole.ECONOMIC, "full"),
        (ParamRole.HORIZON, "full"),
        (ParamRole.STATE_THRESHOLD, "full"),
        (ParamRole.ESTIMATOR_RESOLUTION, "coarse"),
        (ParamRole.NUMERICAL, "excluded"),
        (ParamRole.POLICY, "excluded"),
    ],
)
def test_search_grade_by_declared_role(role, grade):
    spec = ParamSpec(dtype=int, min=1, max=64, default=10, param_role=role)
    assert effective_param_role(spec) is role
    assert param_search_grade(spec) == grade


def test_searchable_false_overrides_role():
    # ``searchable=False`` always wins over any declared role.
    spec = ParamSpec(
        dtype=int, default=3, searchable=False,
        param_role=ParamRole.ECONOMIC,
    )
    assert param_search_grade(spec) == "excluded"


def test_none_spec_is_full_resolution_economic():
    assert effective_param_role(None) is ParamRole.ECONOMIC
    assert param_search_grade(None) == "full"


def test_estimator_resolution_param_is_coarse_grade_not_full():
    # The concrete use case: ``n_segments`` on a spectral estimator must NOT
    # be a full-resolution search dimension.
    spec = ParamSpec(
        dtype=int, min=2, max=32, default=8,
        param_role=ParamRole.ESTIMATOR_RESOLUTION,
    )
    assert param_search_grade(spec) == "coarse"


def test_audit_reports_search_role_per_param():
    from cleaned_operators import operator_audits as oa

    class _Op:
        class _Meta:
            name = "ts_fake_spec_op"
            param_specs = {
                "window": ParamSpec(dtype=int, min=5, max=120, default=20),
                "n_segments": ParamSpec(
                    dtype=int, min=2, max=32, default=8,
                    param_role=ParamRole.ESTIMATOR_RESOLUTION,
                ),
                "seed": ParamSpec(dtype=int, default=0, searchable=False),
            }

        metadata = _Meta()

        def calculate(self, x, **kwargs):
            return x

    results = oa.audit_searchable_params(_Op())
    role_lines = [r for r in results if r.metric.endswith(".search_role")]
    assert any(r.metric == "window.search_role" and r.observed == "economic"
               for r in role_lines)
    assert any(r.metric == "n_segments.search_role"
               and r.observed == "estimator_resolution"
               for r in role_lines)
    assert not any(r.metric == "seed.search_role" for r in role_lines)


def test_state_threshold_is_full_resolution():
    spec = ParamSpec(
        dtype=float, min=0.0, max=1.0, default=0.5,
        param_role=ParamRole.STATE_THRESHOLD,
    )
    assert effective_param_role(spec) is ParamRole.STATE_THRESHOLD
    assert param_search_grade(spec) == "full"


def test_searchable_param_names_splits_by_role():
    class _Meta:
        param_specs = {
            "window": ParamSpec(dtype=int, min=2, max=60, default=20),
            "horizon": ParamSpec(dtype=int, min=1, max=10, default=5,
                                 param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5,
                                   param_role=ParamRole.STATE_THRESHOLD),
            "bins": ParamSpec(dtype=int, min=2, max=32, default=8,
                              param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "epsilon": ParamSpec(dtype=float, default=1e-8,
                                 searchable=False,
                                 param_role=ParamRole.NUMERICAL),
            "method": ParamSpec(dtype=str, choices=["a", "b"], default="a",
                                searchable=False, param_role=ParamRole.POLICY),
        }

    split = searchable_param_names(_Meta())
    assert split["full"] == ["window", "horizon", "threshold"]
    assert split["coarse"] == ["bins"]
    assert split["excluded"] == ["epsilon", "method"]


def test_searchable_param_names_none_metadata():
    assert searchable_param_names(None) == {"full": [], "coarse": [], "excluded": []}
