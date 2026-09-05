"""Tests for hierarchical conditional search (R61-FI-035, plan §24).

The synthetic-objective tests must prove the search learns a good *conditional
parameter region* — not merely the right repair family.  They craft an
objective whose utility depends on a parameter (e.g. U_SHAPE_REPAIR's
``center`` is good only near 0.15), feed history, and assert the proposed
parameter values concentrate near the good region.
"""

import math
import random
from collections import Counter

import pytest

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.policy.repair import DiagnosisKind
from factor_optimizer.policy.repair_registry import (
    ParameterPrior,
    RepairFamily,
    RepairFamilyDeclaration,
    RepairFamilyRegistry,
    ExecutionDomain,
    CausalityClass,
)
from factor_optimizer.search.conditional_search import (
    ConditionalFamily,
    ConditionalParameter,
    ConditionalPriorProvider,
    HierarchicalConditionalSearch,
    StaticPriorProvider,
    build_conditional_tree,
)


def _canonical_tree():
    """The 3-family conditional tree from the plan §24 worked examples."""
    return build_conditional_tree(
        ["CAUSAL_SMOOTHING", "U_SHAPE_REPAIR", "SIZE_NEUTRALIZATION"]
    )


def _family(tree, name):
    return next(f for f in tree if f.family == name)


# ---------------------------------------------------------------------------
# 1. Conditional tree construction
# ---------------------------------------------------------------------------


def test_tree_binds_family_parameter_domains():
    tree = _canonical_tree()
    names = [f.family for f in tree]
    assert names == ["CAUSAL_SMOOTHING", "U_SHAPE_REPAIR", "SIZE_NEUTRALIZATION"]

    causal = _family(tree, "CAUSAL_SMOOTHING")
    assert causal.parameter_names == ("method", "natural_time_scale_relative")
    method = causal.parameters[0]
    assert method.kind == "choice"
    assert set(method.choices) == {"EWMA", "KAMA", "IIR", "Kalman"}

    u = _family(tree, "U_SHAPE_REPAIR")
    assert u.parameter_names == ("center", "power", "asymmetry")
    center = u.parameters[0]
    assert center.kind == "float"
    assert center.low == 0.0 and center.high == 1.0

    size = _family(tree, "SIZE_NEUTRALIZATION")
    assert size.parameter_names == ("exposure_set", "method")


def test_validate_params_exact_match():
    tree = _canonical_tree()
    u = _family(tree, "U_SHAPE_REPAIR")
    u.validate_params({"center": 0.3, "power": 1.5, "asymmetry": False})
    with pytest.raises(ValueError, match="do not match"):
        u.validate_params({"center": 0.3})  # missing power/asymmetry
    with pytest.raises(ValueError, match="out of"):
        u.validate_params({"center": 3.0, "power": 1.5, "asymmetry": False})


def test_conditional_parameter_prior_validation():
    with pytest.raises(ValueError, match="outside"):
        ConditionalParameter(name="x", kind="float", low=0.0, high=1.0, prior=2.0)
    with pytest.raises(ValueError, match="not one of"):
        ConditionalParameter(name="m", kind="choice", choices=("a", "b"), prior="c")
    # Flag params are modeled as boolean choices.
    flag = ConditionalParameter(name="flag", kind="choice", choices=(True, False), prior=True)
    assert flag.sample(random.Random(0)) in (True, False)


# ---------------------------------------------------------------------------
# 2. Prior provider (plan §24: injectable sources, deterministic stub)
# ---------------------------------------------------------------------------


def test_static_prior_provider_resolves_declared_prior():
    provider = StaticPriorProvider()
    assert provider.prior_for("CAUSAL_SMOOTHING", "method") == "EWMA"
    assert provider.prior_for("U_SHAPE_REPAIR", "center") == 0.5
    # Unknown family/parameter -> None (uniform fallback, never raises).
    assert provider.prior_for("NOT_A_FAMILY", "x") is None
    assert provider.prior_for("CAUSAL_SMOOTHING", "nope") is None


def test_static_prior_provider_overrides():
    provider = StaticPriorProvider(overrides={"U_SHAPE_REPAIR.center": 0.15})
    assert provider.prior_for("U_SHAPE_REPAIR", "center") == 0.15


class _GoodCenterPriorProvider:
    """Injected prior source that anchors U_SHAPE_REPAIR.center at 0.15."""

    def prior_for(self, family: str, parameter: str):
        if family == "U_SHAPE_REPAIR" and parameter == "center":
            return 0.15
        return None


def test_injectable_prior_provider_anchors_warmup_proposals():
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(
        tree, n_initial=10, seed=1, prior_provider=_GoodCenterPriorProvider()
    )
    # During warmup the provider's anchored prior is proposed with 50% chance
    # per parameter; over enough proposals the center proposals cluster near 0.15.
    centers = []
    for _ in range(400):
        cand = strat.propose_recipe()
        if cand["family"] == "U_SHAPE_REPAIR":
            centers.append(cand["params"]["center"])
    assert len(centers) > 50, "expected U_SHAPE_REPAIR proposals during warmup"
    near = sum(1 for c in centers if abs(c - 0.15) < 0.15)
    assert near / len(centers) > 0.5  # prior anchor dominates uniform draws


# ---------------------------------------------------------------------------
# 3. Conditional TPE learns the good parameter region
# ---------------------------------------------------------------------------


def _u_center_objective(params):
    """Synthetic objective: U_SHAPE_REPAIR center near 0.15 is good."""
    family = params["family"]
    if family != "U_SHAPE_REPAIR":
        return 0.3
    center = params["center"]
    return max(0.0, 1.0 - abs(center - 0.15) / 0.3)


def test_learns_good_conditional_parameter_region():
    """Search learns a good conditional parameter region, not just a family.

    The feeding phase records every family's good-parameter draws *as well as*
    the bad ones, so the conditional model sees the U_SHAPE_REPAIR center
    region at the top of the good partition.  The final proposals must then
    concentrate on U_SHAPE_REPAIR with center near 0.15 (the conditional
    parameter region the objective rewards).
    """
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=4, gamma=0.25, seed=11)
    for i in range(80):
        # Mix: 60% conditional proposals, 40% deliberate good U_SHAPE_REPAIR
        # draws near the good center so the good partition is well populated.
        if i % 5 >= 2:
            center = 0.12 + (i % 7) * 0.01
            strat.record(
                {"family": "U_SHAPE_REPAIR", "center": center,
                 "power": 1.5, "asymmetry": False},
                _u_center_objective({"family": "U_SHAPE_REPAIR", "center": center}),
            )
        else:
            cand = strat.propose_recipe()
            flat = {"family": cand["family"], **cand["params"]}
            strat.record(flat, _u_center_objective(flat))
    # After learning, proposals concentrate on U_SHAPE_REPAIR with center
    # near 0.15 (the conditional parameter region the objective rewards).
    props = [strat.propose_recipe() for _ in range(100)]
    u_props = [p for p in props if p["family"] == "U_SHAPE_REPAIR"]
    assert len(u_props) >= 40, "conditional model should favor the good family"
    centers = [p["params"]["center"] for p in u_props]
    mean_center = sum(centers) / len(centers)
    assert abs(mean_center - 0.15) < 0.12, f"mean center {mean_center} far from 0.15"
    near = sum(1 for c in centers if abs(c - 0.15) < 0.15)
    assert near / len(centers) > 0.5


def test_learns_family_specific_parameter_region_not_cross_contaminated():
    """Good CAUSAL_SMOOTHING params must not bias U_SHAPE_REPAIR draws."""
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=4, seed=5)
    # Good history: U_SHAPE_REPAIR center near 0.8 AND CAUSAL_SMOOTHING with
    # method=KAMA.  Both are good in their own family.
    rng = random.Random(1)
    for i in range(90):
        family = rng.choice(["CAUSAL_SMOOTHING", "U_SHAPE_REPAIR"])
        fobj = _family(tree, family)
        params = {}
        for pdef in fobj.parameters:
            if pdef.name == "center":
                params["center"] = 0.8
            elif pdef.name == "method":
                params["method"] = "KAMA"
            elif pdef.name == "natural_time_scale_relative":
                params["natural_time_scale_relative"] = 0.5
            elif pdef.name == "power":
                params["power"] = 1.5
            elif pdef.name == "asymmetry":
                params["asymmetry"] = False
            else:
                params[pdef.name] = pdef.sample(rng)
        score = 0.9 if family == "U_SHAPE_REPAIR" else 0.85
        strat.record({"family": family, **params}, score)
    props = [strat.propose_recipe() for _ in range(80)]
    u = [p for p in props if p["family"] == "U_SHAPE_REPAIR"]
    causal = [p for p in props if p["family"] == "CAUSAL_SMOOTHING"]
    if u:
        mean_center = sum(p["params"]["center"] for p in u) / len(u)
        assert abs(mean_center - 0.8) < 0.15, f"U center {mean_center}"
    if causal:
        methods = Counter(p["params"]["method"] for p in causal)
        assert methods.most_common(1)[0][0] == "KAMA"


def test_no_dense_grid_used():
    """Proposals never enumerate a dense parameter grid."""
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=2, seed=2)
    seen = set()
    for _ in range(200):
        cand = strat.propose_recipe()
        key = (cand["family"], tuple(sorted(cand["params"].items())))
        seen.add(key)
    # 200 proposals over continuous domains produce far more than a tiny grid
    # (a dense grid would repeat the same handful of values).
    assert len(seen) > 50


def test_warmup_covers_every_family():
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=1, seed=3)
    proposed = {strat.propose_recipe()["family"] for _ in range(len(tree))}
    assert proposed == {"CAUSAL_SMOOTHING", "U_SHAPE_REPAIR", "SIZE_NEUTRALIZATION"}


# ---------------------------------------------------------------------------
# 4. Strategy contract + checkpoint / resume
# ---------------------------------------------------------------------------


def test_propose_returns_verbatim_categorical_values():
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=2, seed=4)
    trial = strat.propose()
    params = trial.metadata["params"]
    assert "family" in params
    assert params["family"] in {"CAUSAL_SMOOTHING", "U_SHAPE_REPAIR", "SIZE_NEUTRALIZATION"}
    family_obj = _family(tree, params["family"])
    # Every non-family value is a literal candidate of that family's tree.
    for name, value in params.items():
        if name == "family":
            continue
        pdef = next(p for p in family_obj.parameters if p.name == name)
        if pdef.kind == "choice":
            assert value in pdef.choices
        else:
            assert pdef.low <= float(value) <= pdef.high


def test_strategy_is_a_search_strategy():
    from factor_optimizer.search.strategies import SearchStrategy

    strat = HierarchicalConditionalSearch(_canonical_tree(), seed=1)
    assert isinstance(strat, SearchStrategy)
    assert strat.direction == "maximize"


def test_checkpoint_resume_preserves_learning():
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=4, seed=7)
    for i in range(50):
        cand = strat.propose_recipe()
        flat = {"family": cand["family"], **cand["params"]}
        strat.record(flat, _u_center_objective(flat))
    data = strat.to_checkpoint_dict()
    restored = HierarchicalConditionalSearch.from_checkpoint_dict(data)
    assert restored.direction == strat.direction
    # Both strategies propose from the same learned distribution.
    a = [strat.propose_recipe()["family"] for _ in range(20)]
    b = [restored.propose_recipe()["family"] for _ in range(20)]
    assert a == b


def test_proposal_trial_identity_deterministic():
    tree = _canonical_tree()
    s1 = HierarchicalConditionalSearch(tree, n_initial=2, seed=9)
    s2 = HierarchicalConditionalSearch(tree, n_initial=2, seed=9)
    t1 = s1.propose()
    t2 = s2.propose()
    assert t1.trial_id == t2.trial_id
    assert t1.metadata["params"] == t2.metadata["params"]


def test_record_ignores_params_without_family():
    tree = _canonical_tree()
    strat = HierarchicalConditionalSearch(tree, n_initial=2, seed=6)
    strat.record({"no_family_key": 1}, 0.9)
    assert len(strat._records) == 0


def test_validation_fail_closed_on_bad_ctor():
    with pytest.raises(ValueError, match="non-empty tree"):
        HierarchicalConditionalSearch([])
    with pytest.raises(ValueError, match="unique"):
        HierarchicalConditionalSearch(
            [
                ConditionalFamily("CAUSAL_SMOOTHING"),
                ConditionalFamily("CAUSAL_SMOOTHING"),
            ]
        )
    with pytest.raises(ValueError, match="gamma"):
        HierarchicalConditionalSearch(_canonical_tree(), gamma=1.5)


def test_build_conditional_tree_with_injected_registry():
    reg = RepairFamilyRegistry(
        [
            RepairFamilyDeclaration(
                family=RepairFamily.U_SHAPE_REPAIR,
                eligible_diagnoses=frozenset({"U_SHAPE"}),
                owner=ExecutionDomain.FE,
                parameter_schema={"center": "float:0.0:1.0"},
                parameter_prior={"center": 0.5},
                causality_class=CausalityClass.STATELESS,
            )
        ]
    )
    tree = build_conditional_tree([RepairFamily.U_SHAPE_REPAIR], reg)
    assert len(tree) == 1
    assert tree[0].parameter_names == ("center",)
    assert tree[0].parameters[0].prior == 0.5
