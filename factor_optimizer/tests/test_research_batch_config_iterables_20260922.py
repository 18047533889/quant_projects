"""Regression coverage for one-shot family iterables."""

import pytest

from factor_optimizer.research_batch import BatchOptimizationConfig


def test_family_generator_is_frozen_without_silently_enabling_every_family():
    families = (name for name in ("SIGN_ORIENTATION", "U_SHAPE_REPAIR"))

    config = BatchOptimizationConfig(families=families)

    assert config.families == ("SIGN_ORIENTATION", "U_SHAPE_REPAIR")


@pytest.mark.parametrize(
    "families, message",
    [
        ((name for name in ("SIGN_ORIENTATION", "SIGN_ORIENTATION")), "unique"),
        ((name for name in ("SIGN_ORIENTATION", "")), "nonempty"),
    ],
)
def test_family_generator_still_enforces_member_contract(families, message):
    with pytest.raises(ValueError, match=message):
        BatchOptimizationConfig(families=families)
