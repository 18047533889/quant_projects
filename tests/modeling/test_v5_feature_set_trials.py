import numpy as np
import pytest

from factor_optimizer.contracts.campaign_store import DurableBudgetTracker, SQLiteCampaignStore
from factor_optimizer.contracts.search_budget import SearchBudget
from modeling.feature_set_trials import FeatureSetAction, run_feature_set_trials
from modeling.learners.base import LearnerSpec
from modeling.learners.elastic_net import ElasticNetLearner
from modeling.trainer_governance import FeatureExperimentSpec


def test_n11_real_equal_budget_add_replace_drop_cluster_oof(tmp_path):
    rng = np.random.default_rng(7)
    n = 120
    signal, extra, noise = rng.normal(size=(3, n))
    y = 2.0 * signal + extra + rng.normal(scale=.05, size=n)
    features = {"signal": signal, "extra": extra, "noise": noise}
    folds = ((np.arange(0, 60), np.arange(60, 90)), (np.arange(0, 90), np.arange(90, 120)))
    actions = (
        FeatureSetAction("add-extra", "ADD", ("signal", "noise", "extra"), "features@v2#add"),
        FeatureSetAction("replace-signal", "REPLACE", ("noise", "extra"), "features@v2#replace"),
        FeatureSetAction("drop-noise-cluster", "DROP_CLUSTER", ("signal",), "features@v2#drop", "cluster:noise"),
    )
    spec = FeatureExperimentSpec("set-trials", "features@v1", ("fold-1", "fold-2"), 7,
                                 "cpu-small-v1", "oof:baseline", "elastic-net-fixed", 1, 1.0)
    tracker = DurableBudgetTracker(SQLiteCampaignStore(tmp_path / "campaign.sqlite3"), "set-trials",
                                   SearchBudget(max_trials=4, max_evaluations=4, max_cost_units=4.0))
    factory = lambda _: ElasticNetLearner(LearnerSpec(
        "predictive_elastic_net", "elastic_net",
        {"alpha": 1e-4, "l1_ratio": 0.0, "max_iter": 5000, "tol": 1e-10}, random_seed=7,
    ))
    evidence = run_feature_set_trials(
        X_by_feature=features, y=y, row_ids=tuple(f"row:{i}" for i in range(n)),
        baseline_features=("signal", "noise"), actions=actions, folds=folds,
        experiment_spec=spec, learner_factory=factory, budget_tracker=tracker, min_delta=0.0,
        artifact_store_dir=tmp_path / "models",
    )
    assert len(evidence) == 3
    assert all(item.fold_refs == spec.fold_refs and len(item.model_refs) == 2 for item in evidence)
    assert evidence[0].accepted and evidence[0].paired_delta > 0
    assert evidence[1].accepted is False
    assert evidence[2].action.cluster_ref == "cluster:noise"
    assert tracker.store.budget_state("set-trials")["evaluations_used"] == 4


def test_n11_budget_prevents_unrecorded_extra_set_action(tmp_path):
    n = 20
    tracker = DurableBudgetTracker(SQLiteCampaignStore(tmp_path / "one.sqlite3"), "bounded-set",
                                   SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0))
    spec = FeatureExperimentSpec("bounded-set", "v1", ("f",), 0, "cpu", "base", "recipe", 1, 1.0)
    actions = (FeatureSetAction("a", "ADD", ("x", "z"), "v2a"), FeatureSetAction("b", "ADD", ("x", "z"), "v2b"))
    factory = lambda _: ElasticNetLearner(LearnerSpec("predictive_elastic_net", "elastic_net", {"alpha": .01, "l1_ratio": 0.0}))
    with pytest.raises(ValueError, match="budget exhausted"):
        run_feature_set_trials(X_by_feature={"x": np.arange(n), "z": np.arange(n)**2}, y=np.arange(n), row_ids=tuple(map(str, range(n))),
                               baseline_features=("x",), actions=actions,
                               folds=((np.arange(10), np.arange(10, 20)),), experiment_spec=spec,
                               learner_factory=factory, budget_tracker=tracker,
                               artifact_store_dir=tmp_path / "models")


@pytest.mark.parametrize("folds,match", [
    (((np.arange(10), np.arange(9, 15)),), "disjoint"),
    (((np.arange(10, 20), np.arange(0, 5)),), "forward ordered"),
    (((np.arange(0, 10), np.arange(10, 15)), (np.arange(0, 14), np.arange(14, 20))), "repeat"),
])
def test_n11_fold_leakage_fails_closed(tmp_path, folds, match):
    n = 20
    spec = FeatureExperimentSpec("leak", "v1", tuple(f"f{i}" for i in range(len(folds))), 0,
                                 "cpu", "base", "recipe", 1, 1.0)
    tracker = DurableBudgetTracker(SQLiteCampaignStore(tmp_path / "leak.sqlite3"), "leak",
                                   SearchBudget(max_trials=3, max_evaluations=3, max_cost_units=3.0))
    factory = lambda _: ElasticNetLearner(LearnerSpec("predictive_elastic_net", "elastic_net",
                                                       {"alpha": .01, "l1_ratio": 0.0}))
    with pytest.raises(ValueError, match=match):
        run_feature_set_trials(X_by_feature={"x": np.arange(n), "z": np.arange(n)**2}, y=np.arange(n),
                               row_ids=tuple(map(str, range(n))), baseline_features=("x",),
                               actions=(FeatureSetAction("a", "ADD", ("x", "z"), "v2"),), folds=folds,
                               experiment_spec=spec, learner_factory=factory, budget_tracker=tracker,
                               artifact_store_dir=tmp_path / "models")
