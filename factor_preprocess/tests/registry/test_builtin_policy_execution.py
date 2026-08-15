import pandas as pd
import pytest

from factor_preprocess.registry.policies import create_default_policies
from factor_preprocess.registry.transforms import create_default_registry


@pytest.mark.parametrize(
    ("policy_name", "step_name", "expected_parameters"),
    [
        ("causal_basic", "forward_fill", {"max_lag": 5}),
        ("causal_basic", "ewma", {"halflife": 20}),
        ("production_full", "forward_fill", {"max_lag": 3}),
        (
            "production_full",
            "ewma",
            {"halflife": 20, "min_periods": 10},
        ),
        ("research_full", "forward_fill", {"max_lag": 10}),
    ],
)
def test_affected_builtin_step_executes_against_registered_transform(
    policy_name,
    step_name,
    expected_parameters,
):
    frame = pd.DataFrame(
        {
            "asset_id": ["A"] * 12,
            "date": pd.date_range("2024-01-01", periods=12),
            "value": [
                1.0,
                None,
                None,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        }
    )
    policy = create_default_policies().get(policy_name)
    step = next(candidate for candidate in policy.steps if candidate.name == step_name)
    transform = create_default_registry().get_function(step.name)

    assert step.parameters == expected_parameters
    result = transform(frame, **step.parameters)

    assert isinstance(result, pd.Series)
    assert result.index.equals(frame.index)
