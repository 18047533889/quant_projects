"""Tests for the categorical (TPE-style) search strategy."""

import pytest

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.categorical_strategy import (
    CategoricalRecipe,
    CategoricalSearchStrategy,
)
from factor_optimizer.search.runner import SearchConfig, SearchRunner
import numpy as np


def _recipes():
    return [
        CategoricalRecipe(
            "winsorize",
            {"clip": ["1.0", "1.5", "2.0"], "k": [3, 5, 7]},
        ),
        CategoricalRecipe(
            "neutralize",
            {"sector": ["gics", "csrc", "none"], "lambda_": [0.1, 1.0, 5.0]},
        ),
        CategoricalRecipe(
            "zscore",
            {"cap": [3.0, 4.0], "group": ["industry", "market"]},
        ),
    ]


def _protocol(fn):
    return EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), fn)


def test_categorical_not_forced_to_float():
    """propose() returns discrete categorical values, not coerced floats."""
    strat = CategoricalSearchStrategy(_recipes(), seed=1)
    trial = strat.propose()
    params = trial.metadata["params"]
    assert "recipe" in params
    assert isinstance(params["recipe"], str)  # categorical identity preserved
    assert params["recipe"] in {r.name for r in _recipes()}
    # Per-recipe values come verbatim from the recipe's choice lists.
    recipe = next(r for r in _recipes() if r.name == params["recipe"])
    for pname, value in params.items():
        if pname == "recipe":
            continue
        # The proposed value is literally one of the recipe's declared
        # candidates (verbatim identity — never a coerced float index).
        assert value in recipe.param_ranges[pname], (
            f"param {pname!r} value {value!r} coerced out of its choice list"
        )


def test_warmup_then_tpe():
    """Strategy does a warmup phase then improves proposal utility over trials."""
    strat = CategoricalSearchStrategy(_recipes(), n_initial=6, gamma=0.25, seed=1)
    # Warmup covers every recipe at least once.
    proposed = [strat.propose().metadata["params"]["recipe"] for _ in range(6)]
    assert len(set(proposed)) == len(_recipes())  # every recipe seen in warmup
    # Feed history where one recipe clearly dominates utility.
    good_recipe = "winsorize"
    for i in range(30):
        if i % 3 == 0:
            strat.record({"recipe": "neutralize", "clip": "1.5"}, 0.1)
        elif i % 3 == 1:
            strat.record({"recipe": "zscore", "clip": "3.0"}, 0.2)
        else:
            strat.record({"recipe": good_recipe, "clip": "1.5"}, 0.9)
    # After the warmup, the TPE-style model favors the good recipe.
    from collections import Counter
    counts = Counter(
        strat.propose_recipe()["recipe"] for _ in range(40)
    )
    assert counts[good_recipe] > counts["neutralize"]
    assert counts[good_recipe] > counts["zscore"]


def test_runner_accepts_categorical_strategy():
    """SearchRunner(config with candidate_strategy) proposes through it."""
    strat = CategoricalSearchStrategy(_recipes(), n_initial=3, seed=7)
    budget = SearchBudget(max_trials=6, max_evaluations=6, max_cost_units=100.0)
    config = SearchConfig(
        budget=budget,
        candidate_strategy=strat,
        enable_multifidelity=False,
    )

    def evaluation_fn(trial, fidelity):
        recipe = trial.metadata["params"].get("recipe", "unknown")
        score = 0.9 if recipe == "winsorize" else 0.3
        return {"evaluation_id": f"e-{trial.trial_id}", "score": score, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}

    runner = SearchRunner(
        config,
        proposal_fn=lambda: (_ for _ in ()).throw(StopIteration("not used")),
        evaluation_fn=_protocol(evaluation_fn),
    )
    session = runner.run("categorical-run")
    assert session.is_finished()
    assert len(session.trials) == 6
    # The categorical strategy proposed the trials; params carry recipe names.
    params = [t.metadata["params"] for t in session.trials]
    assert all("recipe" in p for p in params)


def test_strategy_checkpoint_resume():
    """CategoricalSearchStrategy round-trips its stochastic state."""
    strat = CategoricalSearchStrategy(_recipes(), seed=3)
    strat.record({"recipe": "winsorize", "clip": "1.5"}, 0.8)
    strat.propose()
    data = strat.to_checkpoint_dict()
    restored = CategoricalSearchStrategy.from_checkpoint_dict(data)
    assert restored.direction == strat.direction
    assert restored.n_initial == strat.n_initial
    assert restored.propose_recipe() is not None


def _integrity_evidence(trial_id="t1", kind=None):
    """Passing TreatmentIntegrityEvidence measured from arrays (R55 P0-9)."""
    from factor_optimizer.contracts.treatment_integrity import (
        build_integrity_evidence,
    )

    rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
    before = rng.normal(size=32)
    treated_kind = kind if kind else f"treatment::{trial_id}"
    if treated_kind == "raw":
        after = before
    else:
        after = before * 0.5 + 0.01
    return build_integrity_evidence(
        trial_id,
        treated_kind,
        {} if treated_kind == "raw" else {"window": 3},
        before,
        after,
    )

