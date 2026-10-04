"""Small offline checks for prepared/unprepared benchmark mechanics."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_prepared_joint_real_oct04.py"
SPEC = importlib.util.spec_from_file_location("prepared_joint_real_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_control_wrapper_drops_only_prepared_split():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    wrapper = MODULE._without_prepared_split(original)
    assert wrapper("raw", "candidate", cost_rate=.001,
                   empty_leg_policy="signal_cash", prepared_split=object()) == "ok"
    assert calls == [(('raw', 'candidate'), {
        "cost_rate": .001, "empty_leg_policy": "signal_cash"})]


def test_validation_call_without_prepared_split_is_forwarded_unchanged():
    calls = []
    wrapper = MODULE._without_prepared_split(
        lambda *args, **kwargs: calls.append((args, kwargs)))
    wrapper("raw", "candidate", cost_rate=.001)
    assert calls == [(("raw", "candidate"), {"cost_rate": .001})]


def test_randomized_schedule_is_reproducible_and_fully_paired():
    warmups, blocks = MODULE._paired_orders(20261004)
    assert warmups == MODULE._paired_orders(20261004)[0]
    assert len(blocks) == MODULE.PAIRED_BLOCKS == 3
    assert all(sorted(pair) == ["prepared", "unprepared"] for pair in blocks)
    assert sorted(warmups) == ["prepared", "unprepared"]


def test_injected_optimizer_has_exact_paired_result_parity(monkeypatch):
    from types import SimpleNamespace
    pair_module = SimpleNamespace()
    observed_calls = []
    def paired_series(*args, **kwargs):
        observed_calls.append(dict(kwargs))
        return {"optimized_values_sha256": "abc", "ledger_sha256": "ledger",
                "diagnostics_sha256": "diag"}
    pair_module.paired_series = paired_series
    def runner(batch, labels, *, config, allow_research, lineages):
        assert allow_research is True and config == "config"
        return pair_module.paired_series(
            "raw", "candidate", cost_rate=.001, prepared_split=object())
    common = dict(batch="batch", labels="labels", lineages={}, config="config",
                  split="split", resource_check=lambda: None,
                  source_check=lambda: None, pair_module=pair_module,
                  snapshotter=lambda value: value)
    prepared = MODULE._invoke_pair(mode="prepared", runner=runner,
                                   **common)[0]
    unprepared = MODULE._invoke_pair(mode="unprepared", runner=runner,
                                     **common)[0]
    MODULE._assert_same_result(prepared, unprepared, "tiny injected test")
    assert prepared == {"optimized_values_sha256": "abc",
                           "ledger_sha256": "ledger", "diagnostics_sha256": "diag"}
    assert "prepared_split" in observed_calls[0]
    assert "prepared_split" not in observed_calls[1]
    assert pair_module.paired_series is paired_series


def test_injected_output_or_ledger_mismatch_is_rejected():
    for field in ("optimized_values_sha256", "ledger_sha256"):
        left = {"optimized_values_sha256": "same", "ledger_sha256": "same"}
        right = dict(left, **{field: "different"})
        try:
            MODULE._assert_same_result(left, right, "intentional mismatch")
        except RuntimeError as exc:
            assert "result mismatch" in str(exc)
        else:
            raise AssertionError(f"mismatch in {field} was accepted")


def test_source_drift_rejects_before_optimizer_call_and_wrapper_install(monkeypatch):
    called = []
    from types import SimpleNamespace
    original = lambda *args, **kwargs: None
    pair_module = SimpleNamespace(paired_series=original)
    try:
        MODULE._invoke_pair(
            mode="unprepared", runner=lambda *args, **kwargs: called.append(True),
            batch=None, labels=None, lineages={}, config=None, split=None,
            resource_check=lambda: None,
            source_check=lambda: (_ for _ in ()).throw(RuntimeError("source drift")),
            pair_module=pair_module)
    except RuntimeError as exc:
        assert "source drift" in str(exc)
    else:
        raise AssertionError("source drift was accepted")
    assert called == []
    assert pair_module.paired_series is original


def test_optimizer_exception_restores_original_callable(monkeypatch):
    from types import SimpleNamespace
    original = lambda *args, **kwargs: None
    pair_module = SimpleNamespace(paired_series=original)
    monkeypatch.setattr(MODULE, "_rss_sampler",
                        lambda: ({"peak": 0, "stop": False}, type(
                            "T", (), {"join": lambda self, timeout=None: None})()))
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")
    try:
        MODULE._invoke_pair(
            mode="unprepared", runner=fail, batch=None, labels=None,
            lineages={}, config=None, split=None,
            resource_check=lambda: None, source_check=lambda: None,
            pair_module=pair_module, snapshotter=lambda value: value)
    except RuntimeError as exc:
        assert "synthetic optimizer failure" in str(exc)
    else:
        raise AssertionError("synthetic failure was swallowed")
    assert pair_module.paired_series is original


def test_rss_sampler_start_failure_restores_original_callable(monkeypatch):
    from types import SimpleNamespace
    original = lambda *args, **kwargs: None
    pair_module = SimpleNamespace(paired_series=original)
    def fail_sampler():
        raise RuntimeError("synthetic sampler startup failure")
    monkeypatch.setattr(MODULE, "_rss_sampler", fail_sampler)
    try:
        MODULE._invoke_pair(
            mode="unprepared", runner=lambda *args, **kwargs: None,
            batch=None, labels=None, lineages={}, config=None, split=None,
            resource_check=lambda: None, source_check=lambda: None,
            pair_module=pair_module, snapshotter=lambda value: value)
    except RuntimeError as exc:
        assert "sampler startup failure" in str(exc)
    else:
        raise AssertionError("sampler startup failure was swallowed")
    assert pair_module.paired_series is original


def test_cli_defaults_to_protocol_dry_run_without_loading(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not load or execute the cohort")
    monkeypatch.setattr(MODULE, "run_audit", forbidden)
    assert MODULE.main([]) == 0
    output = capsys.readouterr().out
    assert '"mode":"dry_run"' in output
    assert '"minimum_memory_admission_bytes":34359738368' in output


def test_cli_requires_explicit_execution_for_report_path():
    try:
        MODULE.main(["--report", "unused.json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--report without --execute should be rejected")


def test_object_axis_digest_depends_on_values_not_allocations():
    import numpy as np
    left = np.array(["asset-one-long-name", "asset-two-long-name"], dtype=object)
    right = np.array([value.encode().decode() for value in left], dtype=object)
    assert left[0] is not right[0]
    assert MODULE._array_digest(left) == MODULE._array_digest(right)
    right[1] = "different-asset"
    assert MODULE._array_digest(left) != MODULE._array_digest(right)
    assert MODULE._array_digest(np.array([True], dtype=object)) != MODULE._array_digest(
        np.array([1], dtype=object))
    import pytest
    with pytest.raises(TypeError, match="unsupported object coordinate"):
        MODULE._array_digest(np.array([object()], dtype=object))


def test_real_snapshot_observes_values_masks_axes_and_complete_ledger(monkeypatch):
    import copy
    import sys
    import types
    import numpy as np
    from types import SimpleNamespace as N

    # Only validation of the optimizer fixture is substituted; real digest and
    # snapshot wiring execute unchanged. This is not a real optimizer run.
    validation = types.ModuleType("audit_real_automatic_oct03")
    calls = []
    validation.validate_result = lambda *args: calls.append(True)
    monkeypatch.setitem(sys.modules, "audit_real_automatic_oct03", validation)
    split = N(identity="split", train_indices=(0,), validation_indices=(1,), test_indices=(2,))
    config = N(seed=17)
    item = N(plan=N(identity="plan"), plan_identity="plan", status="raw_retained",
        selected_family="NO_OP_RAW", reason="test", train_gain=None,
        validation_lower_bound=None, validation_candidate_identity=None,
        validation_coverage=None, materialization_error=None,
        candidates=({"family": "rank", "status": "train_evaluated", "train_gain": .1,
                     "execution_aliases": ({"orientation": 1},)},),
        training_diagnostics={"candidate_budget": {"evaluated": 1}},
        baseline_diagnostics={}, joint_diagnostics={})
    result = N(split=split, config=config, factors={"f": item},
        execution_mode="research_only", test_evaluated=False,
        optimized=N(factor_ids=("f",), values=np.arange(6., dtype=float).reshape(3, 2, 1),
            validity=np.ones((3, 2, 1), dtype=bool),
            time_axis=N(values=np.arange(3)), asset_axis=N(values=np.array(["a", "b"])),
            context_refs={}))
    def snapshot(value):
        return MODULE._result_snapshot(value, None, split, config)
    baseline = snapshot(result)
    assert snapshot(copy.deepcopy(result)) == baseline
    changed = copy.deepcopy(result)
    changed.optimized.values[0, 0, 0] = 99.
    assert snapshot(changed)["values"] != baseline["values"]
    changed = copy.deepcopy(result)
    changed.optimized.validity[0, 0, 0] = False
    assert snapshot(changed)["validity"] != baseline["validity"]
    changed = copy.deepcopy(result)
    changed.optimized.values = changed.optimized.values.astype(np.float32)
    assert snapshot(changed)["values"] != baseline["values"]
    changed.optimized.values = changed.optimized.values.reshape(3, 1, 2)
    assert snapshot(changed)["values"]["shape"] == [3, 1, 2]
    changed = copy.deepcopy(result)
    changed.optimized.asset_axis.values[0] = "c"
    assert snapshot(changed)["asset_axis"] != baseline["asset_axis"]
    changed = copy.deepcopy(result)
    changed.factors["f"].candidates[0]["execution_aliases"][0]["orientation"] = -1
    assert (snapshot(changed)["factors"]["f"]["candidate_ledger_sha256"]
            != baseline["factors"]["f"]["candidate_ledger_sha256"])
    assert calls


def test_run_audit_rejects_leaf_lineage_drift_between_pair_calls(monkeypatch):
    from dataclasses import replace
    import importlib.util
    import sys
    from types import ModuleType
    import numpy as np

    from factor_preprocess.contracts.treatment_lineage import ExistingTreatmentSignature

    fixture_path = Path(__file__).with_name("test_real_scaling_cohort_oct03.py")
    fixture_spec = importlib.util.spec_from_file_location("lineage_cohort_fixture", fixture_path)
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    batch, labels, provenance, _ = fixture_module.sample_cohort()
    lineages = {
        factor_id: ExistingTreatmentSignature()
        for factor_id in batch.factor_ids
    }
    before_values = batch.values.copy()
    before_label_values = labels.values.copy()
    before_provenance = dict(provenance)

    helpers = fixture_module.SOURCE_HELPERS
    monkeypatch.setattr(helpers, "check_environment", lambda: None)
    fake_cos = ModuleType("cos_batch_audit")
    fake_cos.load_cos_sample = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("injected source loader should be used")
    )
    monkeypatch.setitem(sys.modules, "cos_batch_audit", fake_cos)
    monkeypatch.setattr(MODULE, "_runtime_fingerprint", lambda: "fixed-runtime")

    attempted = []
    proceeded = []
    changed_factor = batch.factor_ids[0]

    def invoke_pair_stub(**kwargs):
        attempted.append(kwargs["mode"])
        kwargs["source_check"]()
        proceeded.append(kwargs["mode"])
        if len(proceeded) == 1:
            signature = kwargs["lineages"][changed_factor]
            assert isinstance(signature, ExistingTreatmentSignature)
            kwargs["lineages"][changed_factor] = replace(
                signature, cs_rank=not signature.cs_rank
            )
        return {"factors": {}}, 0.0, 0

    monkeypatch.setattr(MODULE, "_invoke_pair", invoke_pair_stub)
    source_loader = lambda **kwargs: (batch, labels, provenance, lineages)
    try:
        MODULE.run_audit(
            source_loader=source_loader,
            auto_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("optimizer must not run in this protocol test")
            ),
            resource_check=lambda: None,
        )
    except RuntimeError as exc:
        assert "leaf lineage drift" in str(exc)
    else:
        raise AssertionError("leaf lineage drift was accepted between pair calls")

    assert len(attempted) == 2
    assert len(proceeded) == 1
    assert np.array_equal(batch.values, before_values, equal_nan=True)
    assert np.array_equal(labels.values, before_label_values, equal_nan=True)
    assert provenance == before_provenance

def test_lineage_canonical_encoding_is_type_separated_and_order_stable():
    import numpy as np
    import pytest
    encode = MODULE._canonical_lineage_value
    assert encode([1]) != encode((1,))
    assert encode({1}) != encode(frozenset({1}))
    assert encode({"x": 1}) != encode([("x", 1)])
    assert encode("1") != encode(1)
    assert encode(True) != encode(1)
    assert encode(1) != encode(1.0)
    assert encode({"a": 1, "b": 2}) == encode({"b": 2, "a": 1})
    with pytest.raises(TypeError, match="non-finite"):
        encode(float("nan"))
    with pytest.raises(TypeError, match="NumPy"):
        encode(np.longdouble("1.25"))


def test_leaf_lineage_fingerprint_tracks_nested_params_and_status():
    from dataclasses import replace
    from factor_preprocess.contracts.treatment_lineage import ExistingTreatmentSignature

    baseline = ExistingTreatmentSignature(winsor_params={"limits": [0.01, 0.99]})
    equivalent = ExistingTreatmentSignature(winsor_params={"limits": [0.01, 0.99]})
    changed = replace(baseline, winsor_params={"limits": [0.02, 0.99]})
    fingerprint = MODULE._leaf_lineage_fingerprint
    assert fingerprint({"factor": baseline}) == fingerprint({"factor": equivalent})
    assert fingerprint({"factor": baseline}) != fingerprint({"factor": changed})
    assert fingerprint({"factor": baseline}) != fingerprint({
        "factor": replace(baseline, status="incomplete")})
    assert fingerprint({"a": baseline, "b": equivalent}) == fingerprint({
        "b": equivalent, "a": baseline})
