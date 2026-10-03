"""Fast safety tests for the explicit real-COS linear-shape CLI."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_linear_shape_profile_abba as cli
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle


@pytest.fixture
def fake_live(monkeypatch, tmp_path):
    calls = {"closed": []}
    dates, assets = tuple(range(cli.SHAPE[0])), tuple(range(cli.SHAPE[1]))
    records = tuple(f"factor-{i}" for i in range(cli.SHAPE[2]))
    rows, labels = tuple(range(cli.SHAPE[2])), object()
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True, "test": "bounded"})
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda _: object())
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *args: records)
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *args: (dates, assets, rows))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda d, a, *_: (d, a, labels))

    class Source:
        def close(self):
            calls["closed"].append(True)

    source, oracle_bundle = Source(), object()
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **kw: source)
    module = ModuleType("quant_evaluator.scripts.source_linear_shape_oracle")
    module.reference_source_linear_shape = lambda s, lb, **kw: (
        oracle_bundle if s is source and lb is labels else pytest.fail("wrong oracle inputs"))
    monkeypatch.setitem(sys.modules, "quant_evaluator.scripts.source_linear_shape_oracle", module)
    axis = tmp_path / "axis.json"
    axis.write_text("{}", encoding="utf-8")
    return SimpleNamespace(calls=calls, dates=dates, assets=assets, rows=rows,
        labels=labels, source=source, oracle_bundle=oracle_bundle, axis=axis,
        output=tmp_path / "report.json")


def test_default_is_preflight_only_and_never_touches_runtime_or_source(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True, "ram": "ok"})
    for name in ("_auto_batch_cuda_rejection", "_runtime_ready", "_warm"):
        monkeypatch.setattr(cli, name, lambda *a, _name=name, **kw: pytest.fail(f"dry-run called {_name}"))
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("dry-run read manifest"))
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **kw: pytest.fail("dry-run made source"))
    assert cli.main([]) == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert (report["kind"], report["status"], report["run_started"]) == (cli.REPORT_KIND, "preflight_only", False)
    assert report["shape"] == list(cli.SHAPE) and tuple(report["metric_ids"]) == cli.METRICS


def test_run_requires_axis_and_output_before_preflight(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("preflight should not run"))
    with pytest.raises(SystemExit) as err:
        cli.main(["--run"])
    assert err.value.code == 2


@pytest.mark.parametrize("which", ["output", "progress"])
def test_existing_output_or_progress_is_refused_before_reads(fake_live, monkeypatch, which):
    state = fake_live
    progress = state.output.with_suffix(state.output.suffix + ".progress.json")
    target = state.output if which == "output" else progress
    target.write_text("owned", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("collision should precede preflight"))
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("read before reject"))
    with pytest.raises(SystemExit) as err:
        cli.main(["--run", "--axis-index", str(state.axis), "--output", str(state.output)])
    assert err.value.code == 2 and target.read_text(encoding="utf-8") == "owned"


def test_vram_source_gate_fails_before_cos_or_oracle(fake_live, monkeypatch):
    state = fake_live
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: "insufficient VRAM")
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("manifest read before source gate"))
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: pytest.fail("axis read before source gate"))
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **kw: pytest.fail("COS before source gate"))
    with pytest.raises(SystemExit, match="VRAM gate"):
        cli.main(["--run", "--axis-index", str(state.axis), "--output", str(state.output)])
    assert not state.output.exists() and state.calls["closed"] == []


def test_runtime_unavailable_fails_closed_and_closes_oracle_source(fake_live, monkeypatch):
    state = fake_live
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: None)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: False)
    monkeypatch.setattr(cli, "_warm", lambda *a: pytest.fail("warm despite unavailable runtime"))
    monkeypatch.setattr(cli, "_checked_run_backend", lambda *a, **kw: pytest.fail("routed without runtime"))
    with pytest.raises(SystemExit, match="CuPy unavailable"):
        cli.main(["--run", "--axis-index", str(state.axis), "--output", str(state.output)])
    assert state.calls["closed"] == [True] and not state.output.exists()


def _bundle(*, status="qualified_current_source", applied=True, winner="cuda", cache="cache_hit"):
    return SimpleNamespace(metadata={"source_qualification_applied": applied,
        "source_qualification_status": status, "source_qualification_winner": winner,
        "source_qualification_cache_status": cache},
        scalar_metrics={m: np.array([i + .25], dtype=np.float64) for i, m in enumerate(cli.METRICS)},
        observation_counts={m: np.array([1], dtype=np.int64) for m in cli.METRICS})


def _qualification(winner="cuda", width=8):
    return SimpleNamespace(winning_backend=winner, actual_tile_sizes=((winner, width),))


def _records_bound_to_outputs(records, values, counts):
    """Bind real typed qualification records to the fixture's output bytes."""
    outputs_by_metric = {}
    for backend in ("cpu", "cuda"):
        outputs_by_metric[backend] = {}
        for metric, output in zip(cli.METRICS,
                (records[0].cpu if backend == "cpu" else records[0].cuda).outputs):
            value = np.ascontiguousarray(values[metric])
            count = np.ascontiguousarray(counts[metric], dtype="<i8")
            outputs_by_metric[backend][metric] = replace(output,
                values_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                finite_mask_sha256=hashlib.sha256(
                    np.isfinite(value).astype(np.uint8).tobytes()).hexdigest(),
                observation_counts_sha256=hashlib.sha256(count.tobytes()).hexdigest())
    bound = []
    for record in records:
        measurements = {}
        for backend in ("cpu", "cuda"):
            original = getattr(record, backend)
            measurements[backend] = replace(original, outputs=tuple(
                outputs_by_metric[backend][metric] for metric in cli.METRICS))
        comparisons = tuple(replace(item,
            cpu_values_sha256=outputs_by_metric["cpu"][item.metric_id].values_sha256,
            cuda_values_sha256=outputs_by_metric["cuda"][item.metric_id].values_sha256,
            cpu_finite_mask_sha256=outputs_by_metric["cpu"][item.metric_id].finite_mask_sha256,
            cuda_finite_mask_sha256=outputs_by_metric["cuda"][item.metric_id].finite_mask_sha256,
            cpu_observation_counts_sha256=outputs_by_metric["cpu"][item.metric_id].observation_counts_sha256,
            cuda_observation_counts_sha256=outputs_by_metric["cuda"][item.metric_id].observation_counts_sha256)
            for item in record.comparison_evidence)
        bound.append(replace(record, cpu=measurements["cpu"], cuda=measurements["cuda"],
            comparison_evidence=comparisons))
    return tuple(bound)


def test_auto_verification_checks_context_width_status_cache_and_oracle(monkeypatch):
    context = "exact-context"
    monkeypatch.setattr(cli, "_oracle_report", lambda b, r, c, **kw: {"pass": c == context, "metrics": {}})
    receipt = {"context_before": context, "context_after": context,
        "backend_used": "cuda", "effective_max_tile_size": 8}
    got = cli._verify_auto(_bundle(), receipt, _qualification(), context, object(),
        run_index=5, require_cache_hit=True)
    assert got["qualification_applied"] and got["cache_status"] == "cache_hit"
    assert got["output_matches_independent_oracle"] and set(got["values_sha256"]) == set(cli.METRICS)
    invalid = [
        (_bundle(), {**receipt, "context_after": "changed"}, _qualification(), "context"),
        (_bundle(), {**receipt, "effective_max_tile_size": 16}, _qualification(), "qualified"),
        (_bundle(status="stale"), receipt, _qualification(), "qualified"),
        (_bundle(applied=False), receipt, _qualification(), "qualified"),
        (_bundle(winner="cpu"), receipt, _qualification(), "qualified"),
        (_bundle(cache="miss"), receipt, _qualification(), "default auto"),
    ]
    for bundle, bad_receipt, qual, message in invalid:
        with pytest.raises(ValueError, match=message):
            cli._verify_auto(bundle, bad_receipt, qual, context, object(), run_index=5, require_cache_hit=True)
    monkeypatch.setattr(cli, "_oracle_report", lambda *a, **kw: pytest.fail("oracle mismatch"))
    with pytest.raises(pytest.fail.Exception):
        cli._verify_auto(_bundle(), receipt, _qualification(), context, object(), run_index=5, require_cache_hit=True)


@pytest.mark.parametrize("fail_terminal_progress_update", [False, True])
def test_main_orchestrates_four_abba_and_two_auto_verifications(
        fake_live, monkeypatch, fail_terminal_progress_update):
    from test_source_linear_shape_report_reader_oct04 import _payload, _records
    from quant_evaluator.runtime.source_route_profiles import (
        validate_source_route_profile_qualification,
    )
    from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report

    state = fake_live
    payload_template = _payload()
    records = _records()
    context = records[0].context
    qualification = validate_source_route_profile_qualification(
        records, expected_context=context)
    winner = qualification.winning_backend
    width = dict(qualification.actual_tile_sizes)[winner]
    factor_ids = tuple(f"f-{i}" for i in range(cli.SHAPE[-1]))
    values = {}
    counts = {}
    for index, metric in enumerate(cli.METRICS):
        row = np.arange(cli.SHAPE[-1], dtype=np.float64) / (index + 1)
        row[index::17] = np.nan
        values[metric] = row
        counts[metric] = np.isfinite(row).astype(np.int64)
    records = _records_bound_to_outputs(records, values, counts)
    qualification = validate_source_route_profile_qualification(
        records, expected_context=context)
    winner = qualification.winning_backend
    width = dict(qualification.actual_tile_sizes)[winner]
    expected = BatchEvaluationBundle(
        factor_ids=factor_ids, label_id="shape-fixture",
        scalar_metrics={name: value.copy() for name, value in values.items()},
        metadata={"source_request_fingerprint": context.request_content_sha256},
        observation_counts={name: value.copy() for name, value in counts.items()},
    )
    preflight = payload_template["preflight"]
    monkeypatch.setattr(cli, "preflight", lambda: dict(preflight))
    oracle_module = sys.modules["quant_evaluator.scripts.source_linear_shape_oracle"]
    monkeypatch.setattr(oracle_module, "reference_source_linear_shape",
        lambda source, labels, **kwargs: expected)
    calls, warm = [], []
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: None)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "_warm", lambda policy, backend: warm.append(backend))
    if fail_terminal_progress_update:
        class FailingTerminalProgressWriter(cli.ExclusiveProgressWriter):
            def write(self, base, events, *, status="running", error_type=None):
                if status == "complete":
                    raise OSError("diagnostic progress finalization failed")
                return super().write(base, events, status=status, error_type=error_type)
        monkeypatch.setattr(cli, "ExclusiveProgressWriter", FailingTerminalProgressWriter)

    def run_backend(backend, **kwargs):
        calls.append((backend, kwargs))
        if backend == "auto":
            assert kwargs["expected_auto_cuda"] is (winner == "cuda")
            explicit = "source_qualification" in kwargs
            if explicit:
                assert kwargs["source_qualification"] == records
            metadata = {"source_qualification_applied": True,
                "source_qualification_status": "qualified_current_source",
                "source_qualification_winner": winner}
            if not explicit:
                metadata["source_qualification_cache_status"] = "cache_hit"
            bundle = BatchEvaluationBundle(
                factor_ids=factor_ids, label_id="shape-fixture",
                scalar_metrics={name: value.copy() for name, value in values.items()},
                metadata={**metadata,
                    "source_request_fingerprint": context.request_content_sha256},
                observation_counts={name: value.copy() for name, value in counts.items()},
            )
            return bundle, {"context_before": context, "context_after": context,
                "backend_used": winner, "effective_max_tile_size": width}
        used = "cuda" if backend == "cuda_strict" else "cpu"
        bundle = BatchEvaluationBundle(
            factor_ids=factor_ids, label_id="shape-fixture",
            scalar_metrics={name: value.copy() for name, value in values.items()},
            metadata={"source_request_fingerprint": context.request_content_sha256},
            observation_counts={name: value.copy() for name, value in counts.items()},
        )
        return bundle, {"context_before": context, "context_after": context,
            "backend_used": used, "effective_max_tile_size": width,
            "seconds": 1., "total_wall_seconds": 1.}
    monkeypatch.setattr(cli, "_checked_run_backend", run_backend)

    def produce(*, oracle, run_kwargs, run_backend_fn, context_observer, progress_observer):
        assert callable(oracle) and run_kwargs["selected_metrics"] == cli.METRICS
        order = ("cpu", "cuda_strict", "cuda_strict", "cpu")
        oracle_reports = []
        for index, backend in enumerate(order):
            bundle, receipt = run_backend_fn(backend,
                context_observer=context_observer, **dict(run_kwargs))
            progress_observer({"run_index": index, "backend_used": receipt["backend_used"]})
            oracle_reports.append(cli._oracle_report(
                bundle, receipt, context, backend=receipt["backend_used"],
                run_index=index, oracle=oracle))
        return SimpleNamespace(records=records, run_order=order,
            oracle_reports=tuple(oracle_reports))
    monkeypatch.setattr(cli, "produce_source_route_profile_abba", produce)

    args = ["--run", "--axis-index", str(state.axis), "--output", str(state.output)]
    if fail_terminal_progress_update:
        with pytest.raises(OSError, match="diagnostic progress finalization failed"):
            cli.main(args)
    else:
        assert cli.main(args) == 0
    assert [name for name, _ in calls] == ["cpu", "cuda_strict", "cuda_strict", "cpu", "auto", "auto"]
    progress_path = state.output.with_suffix(state.output.suffix + ".progress.json")
    progress_report = __import__("json").loads(progress_path.read_text(encoding="utf-8"))
    assert progress_report["status"] == (
        "running" if fail_terminal_progress_update else "complete")
    assert "error_type" not in progress_report
    assert [row["run_index"] for row in progress_report["validated_runs"]] == list(range(4))
    assert [row["backend_used"] for row in progress_report["validated_runs"]] == ["cpu", "cuda", "cuda", "cpu"]
    assert progress_report["qualification_available"] is False
    assert "profile_records" not in progress_report
    assert warm == ["cpu", "cuda_strict"]
    assert state.calls["closed"] == [True]
    report = __import__("json").loads(state.output.read_text(encoding="utf-8"))
    assert report["kind"] == cli.REPORT_KIND and report["status"] == "complete"
    assert report["qualification_winner"] == winner
    assert report["default_auto_verification"]["cache_status"] == "cache_hit"
    assert "cache_status" not in report["auto_verification"]
    typed = load_source_profile_report(state.output)
    assert typed.kind == cli.REPORT_KIND
    assert typed.qualification == qualification
    assert typed.records == records


def test_auto_verification_real_oracle_rejects_bad_values_counts_and_shape():
    from test_source_linear_shape_report_reader_oct04 import _records
    from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
    from quant_evaluator.scripts.source_profile_abba import IndependentOracleMismatchError

    records = _records()
    context = records[0].context
    qualification = __import__(
        "quant_evaluator.runtime.source_route_profiles", fromlist=[
            "validate_source_route_profile_qualification"]).validate_source_route_profile_qualification(
                records, expected_context=context)
    winner = qualification.winning_backend
    width = dict(qualification.actual_tile_sizes)[winner]
    factor_ids = tuple(f"f-{i}" for i in range(cli.SHAPE[-1]))
    expected_values = {metric: np.linspace(-0.5, 0.5, len(factor_ids), dtype=np.float64)
                       for metric in cli.METRICS}
    expected_counts = {metric: np.ones(len(factor_ids), dtype=np.int64)
                       for metric in cli.METRICS}
    expected = BatchEvaluationBundle(factor_ids, "shape-fixture",
        scalar_metrics={k: v.copy() for k, v in expected_values.items()},
        metadata={"source_request_fingerprint": context.request_content_sha256},
        observation_counts={k: v.copy() for k, v in expected_counts.items()})
    actual = BatchEvaluationBundle(factor_ids, "shape-fixture",
        scalar_metrics={k: v.copy() for k, v in expected_values.items()},
        metadata={"source_request_fingerprint": context.request_content_sha256,
            "source_qualification_applied": True,
            "source_qualification_status": "qualified_current_source",
            "source_qualification_winner": winner},
        observation_counts={k: v.copy() for k, v in expected_counts.items()})
    receipt = {"context_before": context, "context_after": context,
        "backend_used": winner, "effective_max_tile_size": width}

    bad_oracles = []
    bad_counts = {k: v.copy() for k, v in expected_counts.items()}
    bad_counts[cli.METRICS[0]][0] = 0
    bad_oracles.append(BatchEvaluationBundle(factor_ids, "shape-fixture",
        scalar_metrics={k: v.copy() for k, v in expected_values.items()},
        metadata=expected.metadata,
        observation_counts=bad_counts))
    bad_masks = {k: v.copy() for k, v in expected_values.items()}
    bad_masks[cli.METRICS[0]][0] = np.nan
    bad_oracles.append(BatchEvaluationBundle(factor_ids, "shape-fixture",
        scalar_metrics=bad_masks, metadata=expected.metadata,
        observation_counts={k: v.copy() for k, v in expected_counts.items()}))
    bad_shape = {k: v.copy() for k, v in expected_values.items()}
    bad_shape[cli.METRICS[0]] = bad_shape[cli.METRICS[0]][:-1]
    bad_oracles.append(BatchEvaluationBundle(factor_ids, "shape-fixture",
        scalar_metrics=bad_shape, metadata=expected.metadata,
        observation_counts={k: v.copy() for k, v in expected_counts.items()}))

    for bad_oracle in bad_oracles:
        with pytest.raises(IndependentOracleMismatchError):
            cli._verify_auto(actual, receipt, qualification, context, bad_oracle,
                run_index=4, require_cache_hit=False)


def test_shape_report_reader_rejects_extra_auto_fields_and_other_report_family():
    from test_source_linear_shape_report_reader_oct04 import _payload
    from test_source_profile_report_reader_oct04 import _payload_v2 as pearson_payload
    from quant_evaluator.scripts.source_profile_report_reader import (
        parse_source_profile_report,
    )

    payload = _payload()
    payload["auto_verification"]["cache_status"] = "cache_hit"
    with pytest.raises(ValueError, match="fields mismatch"):
        parse_source_profile_report(__import__("json").dumps(payload))

    shape_payload = _payload()
    pearson = pearson_payload()
    assert parse_source_profile_report(__import__("json").dumps(pearson)).kind != cli.REPORT_KIND
    shape_payload["kind"] = pearson["kind"]
    with pytest.raises(ValueError, match="metric order"):
        parse_source_profile_report(__import__("json").dumps(shape_payload))


def test_report_writer_never_overwrites(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli._write_report({"kind": cli.REPORT_KIND, "status": "complete"}, path)
    assert path.read_text(encoding="utf-8") == "owned"


def test_late_progress_collision_preserves_other_workers_evidence(fake_live, monkeypatch):
    state = fake_live
    progress = state.output.with_suffix(state.output.suffix + ".progress.json")
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: None)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)

    def warm(policy, backend):
        if backend == "cuda_strict":
            progress.write_text("other-worker-evidence", encoding="utf-8")

    monkeypatch.setattr(cli, "_warm", warm)
    monkeypatch.setattr(cli, "produce_source_route_profile_abba",
        lambda **kwargs: pytest.fail("ABBA ran after exclusive progress collision"))
    with pytest.raises(FileExistsError):
        cli.main(["--run", "--axis-index", str(state.axis), "--output", str(state.output)])
    assert progress.read_text(encoding="utf-8") == "other-worker-evidence"
    assert not state.output.exists()
    assert state.calls["closed"] == [True]


def test_keyboard_interrupt_marks_progress_failed_and_is_reraised(fake_live, monkeypatch):
    state = fake_live
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: None)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "_warm", lambda *a: None)

    def interrupted(**kwargs):
        kwargs["progress_observer"]({"run_index": 0, "backend_used": "cpu"})
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "produce_source_route_profile_abba", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--run", "--axis-index", str(state.axis), "--output", str(state.output)])
    path = state.output.with_suffix(state.output.suffix + ".progress.json")
    report = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_type"] == "KeyboardInterrupt"
    assert report["qualification_available"] is False
    assert not state.output.exists()


def test_missing_configured_cli_fails_before_resources_or_network(monkeypatch, tmp_path, capsys):
    from data_access.cos import mirror
    missing = tmp_path / "private-configured-cli"
    monkeypatch.setattr(mirror, "_cli_binary", lambda: str(missing))
    monkeypatch.setattr(cli.source_batch, "preflight", lambda *a: pytest.fail("resource checks before missing CLI rejection"))
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("COS access despite missing CLI"))
    with pytest.raises(SystemExit, match="research COS CLI is unavailable") as error:
        cli.main([])
    assert str(missing) not in str(error.value)
    assert "private-configured-cli" not in capsys.readouterr().out


def test_resolved_cli_preserves_closed_resource_preflight_fields(monkeypatch):
    from data_access.cos import mirror
    from test_source_linear_shape_report_reader_oct04 import _payload
    resource_gate = _payload()["preflight"]
    monkeypatch.setattr(mirror, "_cli_binary", lambda: sys.executable)
    monkeypatch.setattr(cli.source_batch, "preflight", lambda *a: dict(resource_gate))
    result = cli.preflight()
    assert result == resource_gate
    assert set(result) == set(resource_gate)
    assert not any(name.startswith("configured_cli") for name in result)
