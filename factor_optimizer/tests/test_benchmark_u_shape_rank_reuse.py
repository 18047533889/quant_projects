from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters import repair_execution
import factor_optimizer.shape_rank_reuse as shape_rank_reuse

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_u_shape_rank_reuse.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_u_shape_rank_reuse", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _fake_result(batch, labels, *, config, allow_research):
    selected = SimpleNamespace(
        candidates=[{
            "family": "U_SHAPE_REPAIR",
            "parameters": {"strength": 0.5},
            "status": "selected",
            "plan_identity": "fake-plan",
            "train_gain": 0.1,
            "coverage": 0.9,
        }],
        selected_family="U_SHAPE_REPAIR",
        plan_identity="fake-plan",
        train_gain=0.1,
        validation_lower_bound=0.05,
        training_diagnostics={
            "candidate_budget": {
                "required": 1,
                "maximum": config.maximum_candidates,
                "status": "admitted",
                "evaluated": 1,
            }
        },
    )
    return SimpleNamespace(
        factors={"synthetic_u": selected},
        split=SimpleNamespace(validation_start=60),
    )


def _instrumented_globals():
    return (
        shape_rank_reuse.apply_u_shape_from_rank,
        shape_rank_reuse.apply_u_shape_from_prepared_rank,
        shape_rank_reuse._frame_key,
        repair_execution._execute_fe_cs_rank,
        repair_execution.ValueRepairPlan.execute,
        shape_rank_reuse.RankFeatureCache,
        shape_rank_reuse.should_admit_u_shape_rank_reuse,
    )


def test_run_restores_hooks_and_observes_actual_train_frame_rows(tmp_path):
    originals = _instrumented_globals()
    original_cache_class = shape_rank_reuse.RankFeatureCache
    mode = {"current": None}

    def fake_execute(self, values, *, allow_research=False, **kwargs):
        return values["value"]

    def fake_optimizer(batch, labels, *, config, allow_research):
        train_start = int(len(batch.time_axis.values) * config.train_fraction)
        train_values = batch.values[:train_start, :, 0]
        train_times = batch.time_axis.values[:train_start]
        assets = batch.asset_axis.values
        frame = pd.DataFrame({
            "date": np.repeat(train_times, len(assets)),
            "asset_id": np.tile(assets, train_start),
            "value": train_values.reshape(-1),
        })
        cache = shape_rank_reuse.RankFeatureCache()
        assert type(cache) is original_cache_class
        if mode["current"] == "cached":
            prepared = cache.prepare_train_rank(frame, training_context_ref="stub-context")
            assert prepared is not None
            plan = repair_execution.compile_value_repair(
                "U_SHAPE_REPAIR",
                {"center": .5, "power": 1.0, "asymmetry": False},
                natural_time_scale=10.0,
                training_context_ref="stub-context",
            )
            shaped = shape_rank_reuse.apply_u_shape_from_prepared_rank(
                plan, prepared, expected_index=frame.index, allow_research=True,
            )
            assert shaped is not None
        else:
            dummy_plan = SimpleNamespace(transform="rank_shape", identity="fake-plan")
            execute = repair_execution.ValueRepairPlan.execute
            for _ in range(14):
                execute(dummy_plan, frame, allow_research=True)
        return _fake_result(
            batch, labels, config=config, allow_research=allow_research
        )

    fake_rank = lambda frame: frame["value"].rank(pct=True, method="average")
    with (
        patch.object(repair_execution, "_execute_fe_cs_rank", new=fake_rank),
        patch.object(repair_execution.ValueRepairPlan, "execute", new=fake_execute),
    ):
        test_baseline = _instrumented_globals()
        with patch.object(bench, "optimize_factor_batch", side_effect=fake_optimizer):
            mode["current"] = "cached"
            cached = bench.run(time_points=100, assets=20, mode="cached", seed=1234)
            assert _instrumented_globals() == test_baseline

            mode["current"] = "uncached"
            uncached = bench.run(time_points=100, assets=20, mode="uncached", seed=1234)
            assert _instrumented_globals() == test_baseline

            mode["current"] = "cached"
            cached_again = bench.run(time_points=100, assets=20, mode="cached", seed=1234)
            assert _instrumented_globals() == test_baseline
    assert _instrumented_globals() == originals

    assert cached["candidate_evidence_sha256"] == uncached["candidate_evidence_sha256"]
    assert cached["candidate_evidence_sha256"] == cached_again["candidate_evidence_sha256"]
    assert cached["admission_semantics"] == {
        "mode": "forced_for_ab", "reuse_admitted": True,
        "production_gate_bypassed": True,
    }
    assert uncached["admission_semantics"] == {
        "mode": "forced_for_ab", "reuse_admitted": False,
        "production_gate_bypassed": True,
    }
    assert cached["candidate_budget"] == {
        "required": 1, "maximum": 128, "status": "admitted", "evaluated": 1
    }
    assert cached["training_frame_rows"] == 1_200
    assert cached["training_frame_observations"]["consistent"] is True
    assert cached["training_frame_observations"]["cached_fingerprint_rows"] == [1_200]
    assert cached["training_frame_observations"]["cached_fe_rank_rows"] == [1_200]
    assert cached["training_frame_observations"]["prepared_rank_transform_rows"] == [1_200]
    assert cached["prepared_rank_transform_calls"] == 1
    assert uncached["training_frame_rows"] == 1_200
    assert uncached["training_frame_observations"]["uncached_rank_shape_execute_rows"] == [1_200] * 14
    assert uncached["training_frame_observations"]["uncached_initial_train_execute_rows"] == [1_200] * 14
    assert cached["rank_cache_metrics"]["instance_count"] == 1
    assert cached["rank_cache_metrics"]["class_identity_preserved"] is True
    # Prepared rank reuse no longer calls the mutable-frame cache a second time;
    # the invocation-local prepared transform is reported separately above.
    assert cached["rank_cache_metrics"]["totals"]["hits"] == 0
    assert cached["rank_cache_metrics"]["totals"]["rank_calls"] == 1
    assert uncached["rank_cache_metrics"]["instance_count"] == 1
    assert uncached["rank_cache_metrics"]["class_identity_preserved"] is True
    assert uncached["rank_cache_metrics"]["totals"]["hits"] == 0
    assert uncached["rank_cache_metrics"]["totals"]["rank_calls"] == 0
    assert uncached["prepared_rank_transform_calls"] == 0
    assert uncached["training_frame_observations"]["prepared_rank_transform_rows"] == []
    assert cached["fixture"]["panel_seed"] == 1234
    assert cached["fixture"]["optimizer_seed"] == 73
    assert cached["fixture"]["panel_values"] == "standard_normal"
    assert cached["execution_limits"]["max_input_cells"] == 3_000_000

    output = tmp_path / "run.json"
    encoded = bench._write_compact_record(output, cached)
    assert output.read_text() == encoded + "\n"
    with pytest.raises(FileExistsError):
        bench._write_compact_record(output, cached)


def test_run_rejects_oversized_panel_before_optimization():
    with patch.object(bench, "optimize_factor_batch") as optimize:
        with pytest.raises(ValueError, match="bounded 3,000,000-cell limit"):
            bench.run(
                time_points=1000, assets=3001, mode="cached", seed=20261001
            )
    optimize.assert_not_called()


def test_run_passes_families_and_candidate_budget():
    def fake_optimizer(batch, labels, *, config, allow_research):
        return _fake_result(batch, labels, config=config, allow_research=allow_research)

    with patch.object(bench, "optimize_factor_batch", side_effect=fake_optimizer) as optimize:
        try:
            bench.run(time_points=100, assets=20, mode="uncached", seed=9,
                      families=("U_SHAPE_REPAIR",), maximum_candidates=6)
        except RuntimeError:
            # The stub optimizer does not create all expected rank observations.
            pass
        config = optimize.call_args.kwargs["config"]
    assert config.families == ("U_SHAPE_REPAIR",)
    assert config.maximum_candidates == 6
