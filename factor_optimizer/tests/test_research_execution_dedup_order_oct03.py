"""Lock proposal ordering and representative choice for execution dedup."""
from factor_optimizer.adapters.preprocessing import compile_smoothing_repair
from factor_optimizer.search.execution_dedup import deduplicate_proposals


def _compile(family, parameters):
    return compile_smoothing_repair(
        family, parameters, natural_time_scale=10,
        training_context_ref="dedup-order-train",
    )


def _proposal(plan):
    return plan.family, dict(plan.parameters), plan, 1


def test_groups_keep_first_ordinal_while_aliases_and_representative_use_their_rules():
    # A's two plans resolve to the same EWMA execution, but have distinct
    # family identities. B is a different, intervening EWMA execution.
    causal_a = _compile(
        "CAUSAL_SMOOTHING",
        {"method": "EWMA", "natural_time_scale_relative": .5},
    )
    decay_a = _compile(
        "DECAY_REFINEMENT",
        {"decay": .5, "half_life_relative": True},
    )
    group_b = _compile(
        "CAUSAL_SMOOTHING",
        {"method": "EWMA", "natural_time_scale_relative": .8},
    )
    assert dict(causal_a.parameters) == dict(decay_a.parameters)
    assert causal_a.transform == decay_a.transform == "ewma"
    assert dict(group_b.parameters) != dict(causal_a.parameters)

    # Put A's higher identity first and its lower identity last: raw ordinals
    # are A-high=0, B=1, A-low=2. Sorting groups by representative ordinal
    # would incorrectly return B before A.
    a_high, a_low = sorted(
        (causal_a, decay_a), key=lambda plan: plan.identity, reverse=True,
    )
    assert a_low.identity < a_high.identity
    proposals = [_proposal(a_high), _proposal(group_b), _proposal(a_low)]

    result = deduplicate_proposals(proposals, compile_plan=_compile)

    assert len(result) == 2
    first, second = result
    # Output group order follows each signature's earliest input occurrence,
    # not the chosen representative's ordinal.
    assert first.aliases[0]["base_plan_identity"] == a_high.identity
    assert first.aliases[1]["base_plan_identity"] == a_low.identity
    assert first.proposal[2].identity == a_low.identity
    assert second.proposal[2].identity == group_b.identity
    assert [alias["family"] for alias in first.aliases] == [
        a_high.family, a_low.family,
    ]
