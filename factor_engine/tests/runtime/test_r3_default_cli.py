"""CLI adapter unit tests; not approved-data acceptance."""
import json
from types import SimpleNamespace
import pytest
from factor_engine import run_pipeline


@pytest.fixture
def facade(monkeypatch):
    calls = []
    class Engine:
        policy = SimpleNamespace(max_manifest_factors=10)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run_many(self, factors):
            factors = list(factors)
            calls.append(factors)
            return {"status": "SUCCEEDED", "counts": {"SUCCEEDED": len(factors)}}
    monkeypatch.setattr("factor_engine.runtime.default_engine.get_engine", Engine)
    return calls


def test_default_cli_uses_same_facade_without_tuning(tmp_path, facade):
    path = tmp_path / "formulas.json"
    path.write_text(json.dumps({"factors": [{"name": "x", "formula": "close + open"}]}))
    receipt = run_pipeline._run_default_compute(path)
    assert receipt["counts"] == {"SUCCEEDED": 1}
    assert facade[0][0].name == "x" and facade[0][0].surface == "all"


@pytest.mark.parametrize("payload", [
    {"factors": [], "backend": "pandas"}, {"factors": []},
    {"factors": [{"name": "x", "formula": "close", "source": "other"}]},
    {"factors": [{"name": "x", "formula": "close"}] * 2},
])
def test_default_cli_rejects_bad_request_before_compute(tmp_path, facade, payload):
    path = tmp_path / "formulas.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run_pipeline._run_default_compute(path)
    assert not facade


def test_default_cli_accepts_no_performance_switches(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_pipeline", "default-compute", "factors.json"])
    args = run_pipeline.parse_args()
    assert args.command == "default-compute" and not hasattr(args, "backend")
    monkeypatch.setattr("sys.argv", ["run_pipeline", "default-compute", "factors.json", "--n-jobs", "8"])
    with pytest.raises(SystemExit):
        run_pipeline.parse_args()


@pytest.mark.parametrize("status,exit_code", [("SUCCEEDED", 0), ("COMPLETED_WITH_ERRORS", 1)])
def test_cli_receipt_and_exit_preserve_partial_failure(monkeypatch, capsys, status, exit_code):
    monkeypatch.setattr("sys.argv", ["run_pipeline", "default-compute", "factors.json"])
    monkeypatch.setattr(run_pipeline, "_run_default_compute", lambda _: {"status": status})
    if exit_code:
        with pytest.raises(SystemExit) as caught:
            run_pipeline.main()
        assert caught.value.code == exit_code
    else:
        run_pipeline.main()
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_cli_keeps_valid_peer_and_typed_parse_rejections(tmp_path, facade):
    from factor_engine.runtime.finite_manifest import RejectedFactorDefinition

    path = tmp_path / "formulas.json"
    entries = [
        {"name": "valid", "formula": "close + open"},
        {"name": "syntax", "formula": "close + ("},
        {"name": "unknown", "formula": "r3_nonexistent_operator(close)"},
    ]
    path.write_text(json.dumps({"factors": entries}))
    run_pipeline._run_default_compute(path)
    items = facade[0]
    assert [item.name for item in items] == [entry["name"] for entry in entries]
    assert not isinstance(items[0], RejectedFactorDefinition)
    assert all(isinstance(item, RejectedFactorDefinition) for item in items[1:])
    assert items[2].error_code == "OPERATOR_UNKNOWN"
    assert items[1].definition_digest != items[2].definition_digest
