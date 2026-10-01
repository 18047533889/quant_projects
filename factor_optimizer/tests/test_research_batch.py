"""Data-backed automatic TRAIN/VALIDATION selection; TEST stays unused."""
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

def fixture():
    rng = np.random.default_rng(82)
    t, n = 240, 40
    z = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)])
    y = z ** 2
    values = np.stack((y, -y, z, np.full_like(z, np.nan)), axis=-1)
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("good", "reverse", "u", "invalid"), ta, aa, values)
    labels = LabelBundle("synthetic", y, 1, decision_time=tuple(range(t)),
                         label_start_time=tuple(range(1,t+1)), label_end_time=tuple(range(2,t+2)), asset_axis=aa)
    return batch, labels

def api():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    # These analytical fixtures test RankIC mechanics, not daily return risk.
    from functools import partial
    return optimize_factor_batch, partial(BatchOptimizationConfig, selection_objective='rank_ic')

def test_automatic_batch_repairs_sign_and_u_but_preserves_good_and_invalid_raw():
    optimize, Config = api()
    batch, labels = fixture()
    result = optimize(batch, labels, config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    assert tuple(result.factors) == batch.factor_ids
    assert result.factors["good"].selected_family == "NO_OP_RAW"
    assert result.factors["reverse"].selected_family == "SIGN_ORIENTATION"
    assert result.factors["u"].selected_family == "U_SHAPE_REPAIR"
    assert result.factors["invalid"].status == "invalid_raw"
    assert np.isnan(result.optimized.values[:,:,3]).all()
    assert result.factors["u"].validation_lower_bound > .1
    assert result.split.test_indices[0] == 192
    assert max(labels.label_end_time[i] for i in result.split.train_indices) < labels.decision_time[result.split.validation_indices[0]]
    assert max(labels.label_end_time[i] for i in result.split.validation_indices) < labels.decision_time[result.split.test_indices[0]]

def test_test_labels_never_change_selected_method_or_scores():
    optimize, Config = api()
    batch, labels = fixture()
    conf = Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR"))
    baseline = optimize(batch, labels, config=conf, allow_research=True)
    poisoned = labels.values.copy()
    poisoned[192:] = np.nan
    altered = optimize(batch, replace(labels, values=poisoned), config=conf, allow_research=True)
    for factor in batch.factor_ids:
        a, b = baseline.factors[factor], altered.factors[factor]
        assert a.selected_family == b.selected_family
        assert a.plan_identity == b.plan_identity
        assert a.train_gain == b.train_gain
        assert a.validation_lower_bound == b.validation_lower_bound

def test_missing_axes_and_missing_opt_in_are_rejected():
    optimize, Config = api()
    batch, labels = fixture()
    with pytest.raises(ValueError, match="research"):
        optimize(batch, labels)
    wrong = replace(batch, asset_axis=AxisRef("asset", "str", 40))
    with pytest.raises(ValueError, match="axes"):
        optimize(wrong, labels, allow_research=True)

def test_short_data_does_not_invent_a_valid_split():
    optimize, Config = api()
    batch, labels = fixture()
    from factor_optimizer.research_batch import automatic_time_split
    with pytest.raises(ValueError, match="observations"):
        automatic_time_split(replace(labels, values=labels.values[:30], decision_time=labels.decision_time[:30],
            label_start_time=labels.label_start_time[:30], label_end_time=labels.label_end_time[:30]), Config())

def test_candidate_order_does_not_change_selection():
    optimize, Config = api()
    batch, labels = fixture()
    a = optimize(batch, labels, config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    b = optimize(batch, labels, config=Config(families=("U_SHAPE_REPAIR", "SIGN_ORIENTATION")), allow_research=True)
    assert {k:v.plan_identity for k,v in a.factors.items()} == {k:v.plan_identity for k,v in b.factors.items()}

def test_validation_label_failure_falls_back_instead_of_trying_other_candidates():
    optimize, Config = api()
    batch, labels = fixture()
    poisoned = labels.values.copy()
    poisoned[144:192] = 0.
    result = optimize(batch, replace(labels, values=poisoned),
        config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    assert all(item.selected_family == "NO_OP_RAW" for item in result.factors.values())
    assert result.factors["reverse"].train_gain > 1.
    assert result.factors["reverse"].validation_lower_bound is None

def test_factor_order_and_test_features_do_not_change_training_decisions():
    optimize, Config = api()
    batch, labels = fixture()
    conf = Config(families=("SIGN_ORIENTATION",))
    baseline = optimize(batch, labels, config=conf, allow_research=True)
    values = batch.values.copy()
    values[192:] = 1e10
    order = [3, 2, 1, 0]
    permuted = replace(batch, factor_ids=tuple(batch.factor_ids[i] for i in order),
                       values=values[:,:,order])
    changed = optimize(permuted, labels, config=conf, allow_research=True)
    assert {k:v.plan_identity for k,v in baseline.factors.items()} == {k:v.plan_identity for k,v in changed.factors.items()}

@pytest.mark.parametrize("field,bad", [
    ("minimum_improvement", True), ("natural_time_scale", True),
    ("block_length", 0), ("seed", True), ("minimum_assets", 1),
    ("train_fraction", float("nan")), ("bootstrap_draws", 3),
])
def test_invalid_configuration_cannot_drive_automatic_selection(field, bad):
    _, Config = api()
    with pytest.raises((ValueError, TypeError)):
        Config(**{field: bad})


def test_full_coverage_requirement_is_allowed():
    _, Config = api()
    assert Config(minimum_coverage=1.).minimum_coverage == 1.


def test_default_missingness_search_executes_fill_not_only_a_diagnostic_flag():
    optimize, Config = api()
    batch, labels = fixture()
    values = batch.values[:,:,:1].copy()
    values[::3, :3] = np.nan
    batch = replace(batch, factor_ids=("good",), values=values)
    result = optimize(batch, labels, config=Config(families=("MISSINGNESS_FRESHNESS",)),
                      allow_research=True)
    records = result.factors["good"].candidates
    assert any(r["parameters"].get("mode") == "fill" and r["status"] == "train_evaluated"
               for r in records)
    # Filling gaps alone cannot claim higher RankIC on shared observations.
    assert result.factors["good"].selected_family == "NO_OP_RAW"



def test_shared_u_rank_preserves_train_candidate_scores_and_selection(monkeypatch):
    optimize, Config = api()
    batch, labels = fixture()
    batch = replace(batch, factor_ids=("u",), values=batch.values[:, :, 2:3])
    config = Config(families=("U_SHAPE_REPAIR",), bootstrap_draws=99, seed=73)

    import factor_optimizer.shape_rank_reuse as reuse_module
    # This tiny parity fixture intentionally forces the optimization on; it is
    # not evidence for production admission at this frame size.
    monkeypatch.setattr(reuse_module, "should_admit_u_shape_rank_reuse",
                        lambda *args, **kwargs: True)
    reused = optimize(batch, labels, config=config, allow_research=True)
    monkeypatch.setattr(reuse_module, "apply_u_shape_from_rank",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(reuse_module, "apply_u_shape_from_prepared_rank",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(reuse_module.RankFeatureCache, "prepare_train_rank",
                        lambda *args, **kwargs: None)
    from factor_optimizer.adapters import repair_execution
    original_execute = repair_execution.ValueRepairPlan.execute
    reference_rank_shape_executes = []

    def observe_reference_execute(self, *args, **kwargs):
        if self.transform == "rank_shape":
            reference_rank_shape_executes.append(self.identity)
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(repair_execution.ValueRepairPlan, "execute",
                        observe_reference_execute)
    reference = optimize(batch, labels, config=config, allow_research=True)
    assert reference_rank_shape_executes

    reused_factor = reused.factors["u"]
    reference_factor = reference.factors["u"]
    assert (reused_factor.selected_family, reused_factor.plan_identity,
            reused_factor.train_gain, reused_factor.validation_lower_bound) == (
            reference_factor.selected_family, reference_factor.plan_identity,
            reference_factor.train_gain, reference_factor.validation_lower_bound)

    def candidate_evidence(result):
        return [
            (item["family"], item["parameters"], item["status"],
             item.get("plan_identity"), item.get("train_gain"), item.get("coverage"))
            for item in result.factors["u"].candidates
        ]

    assert candidate_evidence(reused) == candidate_evidence(reference)



def test_normal_small_frame_uses_ordinary_execute_without_cache_fingerprint(monkeypatch):
    optimize, Config = api()
    batch, labels = fixture()
    batch = replace(batch, factor_ids=("u",), values=batch.values[:, :, 2:3])
    import factor_optimizer.shape_rank_reuse as reuse_module
    from factor_optimizer.adapters import repair_execution

    original_cache = reuse_module.RankFeatureCache
    caches = []
    executions = []

    class ObserveCache(original_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            caches.append(self)

    original_execute = repair_execution.ValueRepairPlan.execute
    def observe_execute(self, *args, **kwargs):
        if self.transform == "rank_shape":
            executions.append(self.identity)
        return original_execute(self, *args, **kwargs)

    fingerprints = []
    monkeypatch.setattr(reuse_module, "RankFeatureCache", ObserveCache)
    monkeypatch.setattr(reuse_module, "_frame_key",
                        lambda *args, **kwargs: fingerprints.append(args[0]))
    monkeypatch.setattr(repair_execution.ValueRepairPlan, "execute", observe_execute)
    result = optimize(batch, labels, config=Config(families=("U_SHAPE_REPAIR",)),
                      allow_research=True)

    assert caches and all(cache.max_bytes == 0 and cache.rank_calls == 0 for cache in caches)
    assert fingerprints == []
    assert executions
    assert result.factors["u"].candidates


def test_shared_diagnosis_keeps_exact_zero_and_tiny_sign_boundaries(monkeypatch):
    import quant_evaluator.metrics.ic as ic_metrics
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    _, Config = api()
    batch, labels = fixture()
    names = batch.factor_ids[:3]
    batch = replace(batch, factor_ids=names, values=batch.values[:, :, :3])
    signs = dict(zip(names, (0.0, 1e-16, -1e-16)))

    def boundary_ic(train, target, **kwargs):
        series = np.zeros((train.values.shape[0], len(train.factor_ids)))
        for k, name in enumerate(train.factor_ids):
            series[0, k] = signs[name]
        return series, np.full(series.shape, 40, dtype=np.int64)

    monkeypatch.setattr(ic_metrics, "compute_daily_ic", boundary_ic)
    config = Config(families=("SIGN_ORIENTATION",))
    shared = diagnose_training_batch(batch, labels, config=config)
    for k, name in enumerate(names):
        one = replace(batch, factor_ids=(name,),
                      values=batch.values[:, :, k:k+1].copy())
        prior = diagnose_training_batch(one, labels, config=config)[name]
        assert shared[name] == prior
        mean = shared[name]["rank_ic"]
        if signs[name] == 0:
            assert mean == 0
        else:
            assert np.sign(mean) == np.sign(signs[name])
        negative_issue = any(issue["code"] == "negative_rank_ic"
                             for issue in shared[name]["issues"])
        assert negative_issue == (signs[name] < 0)


@pytest.mark.parametrize("families", [
    ("SIGN_ORIENTATION",),
    ("SIGN_ORIENTATION", "U_SHAPE_REPAIR", "INVERTED_U_REPAIR"),
])
def test_shared_diagnosis_preserves_complete_optimizer_selection(monkeypatch, families):
    from dataclasses import fields
    import factor_optimizer.research_diagnostics as diagnostics
    optimize, Config = api()
    batch, labels = fixture()
    config = Config(families=families)
    original = diagnostics.diagnose_training_batch
    shared = optimize(batch, labels, config=config, allow_research=True)

    def independent(diagnostic_batch, *args, **kwargs):
        records = {}
        for k, factor_id in enumerate(diagnostic_batch.factor_ids):
            values = diagnostic_batch.values[:, :, k:k+1].copy()
            validity = diagnostic_batch.validity
            one = replace(
                diagnostic_batch, factor_ids=(factor_id,), values=values,
                validity=None if validity is None else validity[:, :, k:k+1].copy(),
            )
            records.update(original(one, *args, **kwargs))
        return records

    monkeypatch.setattr(diagnostics, "diagnose_training_batch", independent)
    prior = optimize(batch, labels, config=config, allow_research=True)
    assert shared.split == prior.split
    assert shared.optimized.factor_ids == prior.optimized.factor_ids
    np.testing.assert_array_equal(shared.optimized.values, prior.optimized.values)
    np.testing.assert_array_equal(shared.optimized.validity, prior.optimized.validity)
    for factor_id in batch.factor_ids:
        actual, expected = shared.factors[factor_id], prior.factors[factor_id]
        for field in fields(actual):
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            if field.name == "plan":
                assert left.identity == right.identity
            else:
                assert left == right, (factor_id, field.name)


def test_raw_baseline_diagnostics_share_batch_preparation_and_preserve_per_factor_results(monkeypatch):
    optimize, Config = api()
    batch, labels = fixture()
    config = Config(families=("SIGN_ORIENTATION",))
    import factor_optimizer.research_diagnostics as diagnostics
    original_diagnose = diagnostics.diagnose_training_batch

    # Independent per-factor reference matching the prior raw diagnostic input.
    expected = {}
    for k, factor_id in enumerate(batch.factor_ids):
        raw = np.array(batch.values[:, :, k], dtype=float, copy=True)
        if batch.validity is not None:
            raw[~batch.validity[:, :, k]] = np.nan
        raw[~np.isfinite(raw)] = np.nan
        one = replace(batch, factor_ids=(factor_id,), values=raw[:, :, None],
                      validity=np.isfinite(raw[:, :, None]))
        expected.update(original_diagnose(one, labels, config=config))
        expected[factor_id]["input_stage"] = "raw"

    calls = []
    def observe(diagnostic_batch, *args, **kwargs):
        calls.append(tuple(diagnostic_batch.factor_ids))
        return original_diagnose(diagnostic_batch, *args, **kwargs)

    monkeypatch.setattr(diagnostics, "diagnose_training_batch", observe)
    result = optimize(batch, labels, config=config, allow_research=True)

    assert calls == [batch.factor_ids]
    assert tuple(result.factors) == batch.factor_ids
    for factor_id in batch.factor_ids:
        actual = dict(result.factors[factor_id].training_diagnostics)
        # Search appends its budget after the shared TRAIN diagnosis.
        assert actual.pop("candidate_budget") == {
            "required": 1, "maximum": config.maximum_candidates,
            "evaluated": sum(record["status"] == "train_evaluated"
                             for record in result.factors[factor_id].candidates),
            "status": "admitted",
        }
        assert actual == expected[factor_id]


def test_raw_batch_diagnostic_helper_matches_independent_results_for_chunk_widths():
    from factor_optimizer.research_batch_diagnostics import diagnose_raw_batch_in_chunks
    from factor_optimizer.research_diagnostics import diagnose_training_batch

    batch, labels = fixture()
    config = api()[1]()
    bytes_per_factor = batch.values.shape[0] * batch.values.shape[1] * 8
    expected = {}
    for k, factor_id in enumerate(batch.factor_ids):
        one = replace(batch, factor_ids=(factor_id,),
                      values=batch.values[:, :, k:k+1].copy(), validity=None)
        expected.update(diagnose_training_batch(one, labels, config=config))

    for width in (1, 2, len(batch.factor_ids)):
        chunks = []

        def observe(chunk, target, *, config):
            chunks.append(chunk.factor_ids)
            return diagnose_training_batch(chunk, target, config=config)

        actual = diagnose_raw_batch_in_chunks(
            batch, labels, config=config, diagnose_training_batch=observe,
            max_chunk_bytes=width * bytes_per_factor,
        )
        assert actual == expected
        assert tuple(fid for ids in chunks for fid in ids) == batch.factor_ids
        assert all(len(ids) <= width for ids in chunks)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_raw_batch_diagnostic_helper_rejects_invalid_memory_bound(invalid):
    from factor_optimizer.research_batch_diagnostics import diagnose_raw_batch_in_chunks

    batch, labels = fixture()
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        diagnose_raw_batch_in_chunks(
            batch, labels, config=None, diagnose_training_batch=lambda *_a, **_k: {},
            max_chunk_bytes=invalid,
        )
