"""Tests for the independent additional model-remediation hard gates."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_model_additional_remediation.py"


def _audit():
    spec = importlib.util.spec_from_file_location("audit_model_additional_remediation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_requested_gates_have_honest_three_state_results():
    payload = _audit().run_audit()
    expected = {
        "MODEL_ZERO_INSAMPLE_HYPERPARAM_SELECTION",
        "MODEL_ZERO_SILENT_EVALUATION_FALLBACK",
        "MODEL_SELECTION_USES_DATEWISE_CROSS_SECTIONAL_OBJECTIVE",
        "MODEL_TURNOVER_IDENTITY_INVARIANT",
        "MODEL_NEGATIVE_CONTROLS_REQUIRE_EXERCISED_MUTATION",
        "MODEL_NEGATIVE_CONTROLS_KNOWN_BAD_VARIANTS_FAIL",
        "MODEL_OOS_HISTORY_HAS_UNIQUE_DATE_ARTIFACT_MAPPING",
        "MODEL_LABEL_MATURITY_USES_SESSION_CALENDAR",
        "MODEL_SAMPLE_TELEMETRY_MATCHES_ACTUAL_FIT_COHORT",
        "MODEL_ZERO_IGNORED_DECLARED_SAMPLE_WEIGHTS",
        "MODEL_PRODUCTION_SCORING_REQUIRES_ASOF_CONTEXT",
        "MODEL_RESOLVER_USES_ACTIVE_DEPLOYMENT_NOT_LATEST_FIT",
        "MODEL_PRODUCTION_ARTIFACT_REQUIRES_CERTIFICATION",
        "MODEL_HISTORICAL_ARTIFACT_PREDICTOR_ABI_REPRODUCIBLE",
        "MODEL_TRAINER_ENFORCES_PARAMETER_SEARCH_POLICY",
        "MODEL_ZERO_UNKNOWN_FEATURE_AVAILABILITY_FAIL_OPEN",
        "MODEL_LABEL_CONTRACT_IDENTITY_COMPLETE",
        "MODEL_DECISION_CLOCK_IDENTITY_COMPLETE",
        "MODEL_PREDICTION_ROW_ALIGNMENT_VERIFIED",
        "MODEL_TRAINING_UNKNOWN_EXCEPTIONS_FAIL_LOUD",
    }
    assert set(payload["gates"]) == expected
    assert payload["summary"]["total"] == len(expected)
    for name, gate in payload["gates"].items():
        assert gate["status"] in {"PASS", "FAIL", "NOT_RUN"}, name
        assert gate["check"]
        if gate["status"] == "NOT_RUN":
            assert gate["executed_cases"] == gate["failed_cases"] == 0
        elif gate["status"] == "PASS":
            assert gate["executed_cases"] > 0 and gate["failed_cases"] == 0
        else:
            assert gate["executed_cases"] > 0 and gate["failed_cases"] > 0


def test_no_literal_pass_gate_construction():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Gate":
            assert not (node.args and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "PASS"), node.lineno


def test_behavior_and_ast_probes_are_not_presence_only():
    gates = _audit().run_audit()["gates"]
    # Current-tree outcomes may be PASS/FAIL/NOT_RUN as implementation evolves;
    # the probes themselves must have executed where runtime surfaces exist.
    for name in (
        "MODEL_ZERO_INSAMPLE_HYPERPARAM_SELECTION",
        "MODEL_ZERO_SILENT_EVALUATION_FALLBACK",
        "MODEL_NEGATIVE_CONTROLS_REQUIRE_EXERCISED_MUTATION",
        "MODEL_TRAINING_UNKNOWN_EXCEPTIONS_FAIL_LOUD",
    ):
        assert gates[name]["executed_cases"] > 0
    negative = gates["MODEL_NEGATIVE_CONTROLS_KNOWN_BAD_VARIANTS_FAIL"]
    assert negative["details"]["known_bad_mutation_killed"] is True
    assert negative["executed_cases"] >= 2


def test_cli_writes_requested_path_and_refuses_to_overwrite(tmp_path):
    out = tmp_path / "additional.json"
    first = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], cwd=ROOT,
                           text=True, capture_output=True)
    assert first.returncode in {0, 1}
    assert json.loads(out.read_text(encoding="utf-8"))["generated_by"].endswith(
        "audit_model_additional_remediation.py")
    assert list(tmp_path.iterdir()) == [out]

    second = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], cwd=ROOT,
                            text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
