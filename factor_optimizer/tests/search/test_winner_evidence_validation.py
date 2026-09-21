"""Winner selection must reject poisoned policies and ambiguous candidate IDs."""
import pytest
from factor_optimizer.search.winner_selector import WinnerPolicy
from factor_optimizer.search.uncertainty_winner import (
    UncertaintyConfig, UncertaintyEvidence, UncertaintyAwareWinnerSelector,
)

@pytest.mark.parametrize("field", ["alpha", "beta", "gamma", "lambda_"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_policy_weights_cannot_create_plausible_winners(field, bad):
    kwargs = dict(alpha=1., beta=0., gamma=0., lambda_=0., policy_id="audit", policy_version="1")
    kwargs[field] = bad
    with pytest.raises((ValueError, TypeError)):
        WinnerPolicy(**kwargs)

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_meaningful_improvement_is_rejected(bad):
    with pytest.raises(ValueError):
        UncertaintyConfig(minimum_meaningful_improvement=bad)

def test_duplicate_candidate_ids_cannot_overwrite_utility_evidence():
    policy = WinnerPolicy(1, 0, 0, 0, "audit", "1")
    candidates = [
        UncertaintyEvidence("same", {"ic": (.1, .2, .1)}),
        UncertaintyEvidence("same", {"ic": (.8, .9, .8)}),
    ]
    with pytest.raises(ValueError, match="unique"):
        UncertaintyAwareWinnerSelector(policy).select(candidates, {"same": 0})
