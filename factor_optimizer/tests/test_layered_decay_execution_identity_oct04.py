from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
from factor_optimizer.search.execution_dedup import _execution_binding, _execution_signature

def test_identity_names_selected_layers():
    identity = _execution_binding(LayeredDecayPlan((5.0,) * 20, "train:test"))["binding"]
    assert {"runner", "state_class", "state_method:step", "state_method:_step_sparse_trusted",
            "state_method:step_sparse", "state_method:_apply_inputs", "state_method:__init__",
            "assign_quantiles_batch", "selected_frame_validator", "plan_execute",
            "qe_validate_tie_policy", "qe_validate_quantile_count",
            "qe_searchsorted_bins"} == set(identity["selected"])
    assert {"plan_adapter", "sparse_runner", "state", "qe_quantile", "frame_validation"} == set(identity["modules"])
    assert "selected_frame_validator" in identity["selected"]

def test_unchanged_layered_decay_identity_is_stable():
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    assert _execution_signature(plan, 1, {"train": "same"}) == _execution_signature(
        plan, 1, {"train": "same"})

def test_signature_changes_for_runtime_selected_implementations(monkeypatch):
    from factor_optimizer.adapters import layered_decay_long
    from factor_preprocess.transforms.layered_decay_state import LayeredDecayState
    from quant_evaluator.metrics import quantile
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    original_signature = _execution_signature(plan, 1, {"train": "same"})
    for obj, name in ((layered_decay_long, "_execute_sparse_layered_decay_validated"),
                       (LayeredDecayState, "__init__"),
                       (LayeredDecayState, "step"),
                       (LayeredDecayState, "step_sparse"),
                       (LayeredDecayState, "_step_sparse_trusted"),
                       (LayeredDecayState, "_apply_inputs"),
                       (quantile, "assign_quantiles_batch"),
                       (quantile, "validate_tie_policy"),
                       (quantile, "_validate_quantile_count"),
                       (quantile, "_searchsorted_bins")):
        original = getattr(obj, name)
        def replacement(*args, __original=original, **kwargs):
            return __original(*args, **kwargs)
        monkeypatch.setattr(obj, name, replacement)
        assert _execution_signature(plan, 1, {"train": "same"}) != original_signature
        original_signature = _execution_signature(plan, 1, {"train": "same"})

def test_signature_changes_when_selected_state_class_changes(monkeypatch):
    from factor_preprocess.transforms import layered_decay_state as state
    from factor_preprocess.transforms.layered_decay_state import LayeredDecayState
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    before = _execution_signature(plan, 1, {"train": "same"})
    class ReplacementState(LayeredDecayState):
        pass
    monkeypatch.setattr(state, "LayeredDecayState", ReplacementState)
    assert _execution_signature(plan, 1, {"train": "same"}) != before

def test_missing_source_fails_closed_at_binding_boundary(monkeypatch):
    from factor_optimizer.adapters import layered_decay_identity
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    monkeypatch.setattr(layered_decay_identity.inspect, "getsource",
        lambda obj: (_ for _ in ()).throw(OSError("no source")))
    try:
        _execution_binding(plan)
    except ValueError as exc:
        assert "cannot certify" in str(exc)
    else:
        raise AssertionError("unavailable identity source must not bind")

def test_missing_selected_state_class_fails_closed(monkeypatch):
    from factor_preprocess.transforms import layered_decay_state as state
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    monkeypatch.setattr(state, "LayeredDecayState", None)
    try:
        _execution_binding(plan)
    except ValueError as exc:
        assert "selected layered-decay implementation" in str(exc)
    else:
        raise AssertionError("missing state class must not be bound")

def test_identity_failure_keeps_duplicate_proposals_separate(monkeypatch):
    from factor_optimizer.adapters import layered_decay_identity
    from factor_optimizer.search.execution_dedup import deduplicate_proposals
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    monkeypatch.setattr(layered_decay_identity.inspect, "getsource",
        lambda obj: (_ for _ in ()).throw(OSError("no source")))
    proposal = ("DECAY_REFINEMENT", {"half_lives": (5.0,) * 20}, plan, 1)
    result = deduplicate_proposals([proposal, proposal], compile_plan=lambda *args: plan)
    assert len(result) == 2
    assert all(item.aliases == () for item in result)

def test_signature_changes_when_selected_frame_validator_changes(monkeypatch):
    from factor_optimizer.adapters import layered_decay as adapter
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    before = _execution_signature(plan, 1, {"train": "same"})
    original = adapter._validate_frame
    def replacement(*args, **kwargs):
        return original(*args, **kwargs)
    monkeypatch.setattr(adapter, "_validate_frame", replacement)
    assert _execution_signature(plan, 1, {"train": "same"}) != before
