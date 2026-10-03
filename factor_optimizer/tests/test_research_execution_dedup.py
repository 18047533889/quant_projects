from dataclasses import replace

import numpy as np

from factor_optimizer.adapters.preprocessing import compile_smoothing_repair
from factor_optimizer.adapters.repair_execution import compile_value_repair
from factor_optimizer.search.execution_dedup import deduplicate_proposals


def _aliases(context="train-A"):
    causal = compile_smoothing_repair(
        "CAUSAL_SMOOTHING",
        {"method": "EWMA", "natural_time_scale_relative": .5},
        natural_time_scale=10, training_context_ref=context)
    decay = compile_smoothing_repair(
        "DECAY_REFINEMENT", {"decay": .5, "half_life_relative": True},
        natural_time_scale=10, training_context_ref=context)
    return causal, decay


def _compile(family, params):
    return compile_value_repair(family, params, natural_time_scale=10,
                                training_context_ref="train-A")


def test_h5_ewma_aliases_collapse_but_keep_family_provenance_and_final_tie():
    causal, decay = _aliases()
    assert causal.transform == decay.transform == "ewma"
    assert dict(causal.parameters) == dict(decay.parameters) == {
        "halflife": 5., "min_periods": 1}
    proposals = [(causal.family, dict(causal.parameters), causal, 1),
                 (decay.family, dict(decay.parameters), decay, 1)]
    result = deduplicate_proposals(
        proposals, compile_plan=_compile,
        final_identity=lambda plan, sign: plan.identity)
    assert len(result) == 1
    assert len(result[0].aliases) == 2
    assert {alias["family"] for alias in result[0].aliases} == {
        "CAUSAL_SMOOTHING", "DECAY_REFINEMENT"}
    assert result[0].proposal[2].identity == min(causal.identity, decay.identity)
    assert result[0].proposal[2] is not None


def test_orientation_and_training_context_are_signature_dimensions():
    causal, _ = _aliases()
    opposite = compile_smoothing_repair(
        "CAUSAL_SMOOTHING",
        {"method": "EWMA", "natural_time_scale_relative": .5},
        natural_time_scale=10, training_context_ref="train-B")
    proposals = [(causal.family, dict(causal.parameters), causal, 1),
                 (causal.family, dict(causal.parameters), causal, -1),
                 (opposite.family, dict(opposite.parameters), opposite, 1)]
    result = deduplicate_proposals(proposals, compile_plan=_compile)
    assert len(result) == 3
    assert {item.proposal[3] for item in result} == {1, -1}


def test_same_transform_parameters_do_not_merge_across_executor_routes():
    registry_path = compile_smoothing_repair(
        "CAUSAL_SMOOTHING", {"method": "SMA", "natural_time_scale_relative": .5},
        natural_time_scale=10, training_context_ref="train-A")
    direct_path = compile_value_repair(
        "CAUSAL_SMOOTHING", {"method": "SMA", "natural_time_scale_relative": .5},
        natural_time_scale=10, training_context_ref="train-A")
    assert registry_path.transform == direct_path.transform == "trailing_sma"
    assert dict(registry_path.parameters) == dict(direct_path.parameters)
    proposals = [(registry_path.family, {}, registry_path, 1),
                 (direct_path.family, {}, direct_path, 1)]
    result = deduplicate_proposals(proposals, compile_plan=_compile)
    assert len(result) == 2


def test_baseline_binding_is_a_signature_dimension():
    causal, decay = _aliases()
    proposals = [(causal.family, dict(causal.parameters), causal, 1),
                 (decay.family, dict(decay.parameters), decay, 1)]
    same_context = deduplicate_proposals(proposals, compile_plan=_compile,
                                         baseline_context="baseline-A")
    other_context = deduplicate_proposals(proposals, compile_plan=_compile,
                                          baseline_context="baseline-B")
    assert len(same_context) == len(other_context) == 1



def test_optimizer_applies_budget_after_dedup_and_reports_all_aliases(monkeypatch):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    import factor_optimizer.research_batch as research_batch
    import factor_optimizer.research_diagnostics as diagnostics
    import factor_optimizer.adapters.preprocessing as preprocessing
    import factor_optimizer.search.execution_dedup as execution_dedup
    from factor_optimizer.adapters.preprocessing import SmoothingRepairPlan
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan
    from factor_optimizer.search.execution_dedup import DeduplicatedProposal

    rng = np.random.default_rng(82)
    t, n = 240, 40
    values = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)])
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("factor",), ta, aa, values[:, :, None])
    labels = LabelBundle(
        "synthetic", values, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=aa)

    monkeypatch.setattr(research_batch, "_specs", lambda config: [
        ("CAUSAL_SMOOTHING", {"method": "EWMA", "natural_time_scale_relative": .5}),
        ("DECAY_REFINEMENT", {"decay": .5, "half_life_relative": True}),
    ])
    monkeypatch.setattr(preprocessing, "compile_admissible_smoothing_grid",
                        lambda **kwargs: ())
    original = diagnostics.diagnose_training_batch

    def controlled_diagnosis(candidate_batch, candidate_labels, **kwargs):
        result = original(candidate_batch, candidate_labels, **kwargs)
        for record in result.values():
            record["proposed_shape_family"] = None
            record["layer_decay"]["proposed_half_lives"] = [5.]
            record["layer_decay"]["status"] = "unavailable"
            record["layer_decay"]["layers"] = []
        return result

    monkeypatch.setattr(diagnostics, "diagnose_training_batch", controlled_diagnosis)
    config = research_batch.BatchOptimizationConfig(
        selection_objective="rank_ic",
        families=("CAUSAL_SMOOTHING", "DECAY_REFINEMENT", "SIGN_ORIENTATION"),
        maximum_candidates=6, bootstrap_draws=99)

    executions = []
    original_smoothing_execute = SmoothingRepairPlan.execute
    original_value_execute = ValueRepairPlan.execute

    def count_smoothing_execute(self, values, **kwargs):
        if self.transform == "ewma":
            executions.append(self.identity)
        return original_smoothing_execute(self, values, **kwargs)

    def count_value_execute(self, values, **kwargs):
        if self.transform == "ewma":
            executions.append(self.identity)
        return original_value_execute(self, values, **kwargs)

    monkeypatch.setattr(SmoothingRepairPlan, "execute", count_smoothing_execute)
    monkeypatch.setattr(ValueRepairPlan, "execute", count_value_execute)
    result = research_batch.optimize_factor_batch(
        batch, labels, config=config, allow_research=True)
    factor = result.factors["factor"]
    budget = factor.training_diagnostics["candidate_budget"]
    assert budget["raw_required"] == 6
    assert budget["unique_required"] == budget["required"] == 2
    assert budget["deduplicated"] == 4
    assert budget["status"] == "admitted"
    assert len(factor.candidates) == 2
    deduplicated_execute_count = len(executions)

    original_deduplicator = execution_dedup.deduplicate_proposals
    monkeypatch.setattr(execution_dedup, "deduplicate_proposals",
        lambda proposals, **kwargs: tuple(DeduplicatedProposal(proposal, ())
                                         for proposal in proposals))
    executions.clear()
    uncollapsed = research_batch.optimize_factor_batch(
        batch, labels, config=config, allow_research=True).factors["factor"]
    uncollapsed_execute_count = len(executions)
    assert deduplicated_execute_count < uncollapsed_execute_count
    assert (factor.selected_family, factor.plan_identity) == (
        uncollapsed.selected_family, uncollapsed.plan_identity)
    monkeypatch.setattr(execution_dedup, "deduplicate_proposals", original_deduplicator)
    assert all(len(record["execution_aliases"]) == 3 for record in factor.candidates)
    aliases = [alias for record in factor.candidates for alias in record["execution_aliases"]]
    assert any(alias.get("proposal_source") == "TRAIN_layer_decay" for alias in aliases)

    altered = values.copy()
    altered[144:] *= -1
    changed_labels = LabelBundle(
        "synthetic", altered, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)), asset_axis=aa)
    changed = research_batch.optimize_factor_batch(
        batch, changed_labels, config=config, allow_research=True).factors["factor"]
    assert [(row.get("plan_identity"), row["status"], row.get("train_gain"))
            for row in factor.candidates] == [
        (row.get("plan_identity"), row["status"], row.get("train_gain"))
        for row in changed.candidates]

    too_small = replace(config, maximum_candidates=1)
    blocked = research_batch.optimize_factor_batch(
        batch, labels, config=too_small, allow_research=True).factors["factor"]
    assert blocked.status == "budget_exceeded_raw_retained"
    assert blocked.training_diagnostics["candidate_budget"]["required"] == 2
    assert blocked.training_diagnostics["candidate_budget"]["raw_required"] == 6
    assert blocked.candidates == ()


def test_different_registry_binding_hashes_do_not_merge(monkeypatch):
    from types import SimpleNamespace
    import factor_preprocess.registry.transforms as registry_transforms
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    current = {"hash": None}

    class Registry:
        def get(self, name):
            assert name == "ewma"
            return SimpleNamespace(
                name="ewma", version="1", semantic_id="EWMA",
                signature_hash="same-signature",
                implementation_hash=current["hash"], numeric_policy_hash="same-numeric",
                implementation_origin="FP_NATIVE", fe_operator_id=None,
                fit_kind="stateless", numeric_policy="nan", fe_equivalent_semantics=None,
                parameter_domain={"halflife": (1., 60.)})

        def get_execution(self, name):
            return SimpleNamespace(executor=lambda values: values,
                                   execution_identity=None)

        def resolve_origin(self, name):
            return "FP_NATIVE"

    monkeypatch.setattr(registry_transforms, "get_default_registry", lambda: Registry())

    def compile_with_binding(family, params):
        current["hash"] = family
        return ValueRepairPlan(family, "ewma", (("halflife", 5.), ("min_periods", 1)),
                               "train-A", 10.)

    proposals = [("alias-A", {}, None, 1), ("alias-B", {}, None, 1)]
    result = deduplicate_proposals(proposals, compile_plan=compile_with_binding)
    assert len(result) == 2


def test_baseline_identity_changes_execution_signature():
    from factor_optimizer.search.execution_dedup import _execution_signature
    plan = _compile("CAUSAL_SMOOTHING", {
        "method": "EWMA", "natural_time_scale_relative": .5})
    assert _execution_signature(plan, 1, "baseline-A") != _execution_signature(
        plan, 1, "baseline-B")


def test_unknown_plan_subclasses_are_not_deduplicated():
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    class CustomPlan(ValueRepairPlan):
        def execute(self, values, *, allow_research=False):
            return super().execute(values, allow_research=allow_research)

    known = ValueRepairPlan("CAUSAL_SMOOTHING", "ewma",
                            (("halflife", 5.), ("min_periods", 1)), "train-A", 10.)
    custom = CustomPlan("DECAY_REFINEMENT", "ewma",
                        (("halflife", 5.), ("min_periods", 1)), "train-A", 10.)

    result = deduplicate_proposals(
        [(known.family, {}, known, 1), (custom.family, {}, custom, 1)],
        compile_plan=_compile)
    assert len(result) == 2


def test_fe_rank_binding_uses_canonical_pandas_backend_and_tracks_implementation(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_optimizer.search.execution_dedup import _execution_binding, _execution_signature

    plan = compile_value_repair(
        "REPRESENTATION_RANK",
        {"rank_axis": "cross_sectional", "tie_method": "average"},
        natural_time_scale=10, training_context_ref="train-A")
    assert plan.transform == "cs_rank"
    binding = _execution_binding(plan)
    assert binding["route"] == "factor_engine.operator_registry"
    assert binding["binding"]["canonical"] == "rank"
    assert binding["binding"]["backend"] == "pandas_numpy"
    assert binding["binding"]["mode"] == "any"
    assert binding["binding"]["operator_implementation_hash"]
    assert binding["binding"]["operator_contract_hash"]
    assert binding["binding"]["adapter_implementation_hash"]
    assert binding["binding"]["input_contract"] == (
        "pandas.DataFrame[asset_id,date,value]->float64")
    original_signature = _execution_signature(plan, 1, None)

    original_get = OperatorRegistry.get
    actual = original_get("rank", backend="pandas_numpy", mode="any")

    class ReplacementRank:
        metadata = actual.metadata

        def _calculate_series(self, values, **kwargs):
            return values

    def replacement_get(name, backend="pandas_numpy", *, mode="production"):
        canonical = OperatorRegistry.resolve_canonical(name)
        if canonical == "rank" and backend == "pandas_numpy":
            return ReplacementRank()
        return original_get(name, backend=backend, mode=mode)

    monkeypatch.setattr(OperatorRegistry, "get", replacement_get)
    changed = _execution_binding(plan)
    assert changed["binding"]["canonical"] == "rank"
    assert changed["binding"]["operator_implementation_hash"] != (
        binding["binding"]["operator_implementation_hash"])
    assert _execution_signature(plan, 1, None) != original_signature


def test_layered_decay_plans_with_same_lives_deduplicate():
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    first = LayeredDecayPlan((5.,) * 20, "train-A")
    second = LayeredDecayPlan((5.,) * 20, "train-A")
    proposals = [("DECAY_REFINEMENT", {}, first, 1),
                 ("DECAY_REFINEMENT", {}, second, 1)]
    result = deduplicate_proposals(proposals, compile_plan=_compile)
    assert len(result) == 1
    assert result[0].proposal[2].identity == first.identity
