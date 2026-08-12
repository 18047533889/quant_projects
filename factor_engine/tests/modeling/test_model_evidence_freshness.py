# -*- coding: utf-8 -*-
"""P0 evidence-freshness and no-silent-fallback tests.

Covers:

* ``MODEL_CURRENT_HEAD_EVIDENCE_FRESH`` is TRUE **only** when the evidence SHA
  equals the repository HEAD resolved at runtime (equal / mismatched / None /
  unavailable-sentinel cases are exercised — stale or unverifiable evidence
  never passes, and ``UNKNOWN==UNKNOWN`` can no longer fake-green).
* The walk-forward runner FAILS (propagates) when the authoritative trainer
  errors — it never falls back to a self-contained path that keeps the gates
  green.
* The gate report carries an honest machine-readable ``status`` (PASS / FAIL /
  NOT_RUN) and gates that cannot be genuinely exercised are NOT_RUN, never a
  hard-coded PASS.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from modeling.contracts import LabelContract, ashare_decision_clock
from modeling.evidence import _repository_head, report_hard_gate_set, run_walk_forward_evidence
from modeling.leakage_guard import synthetic_panel
from modeling.learners import PCRLearner


def _wf_spec():
    return {
        "train_lookback_bars": 4,
        "validation_bars": 2,
        "test_bars": 2,
        "step_bars": 2,
        "embargo_bars": 0,
    }


def _small_dataset():
    return synthetic_panel(n_dates=12, n_stocks=15, n_features=3, seed=7)


# --------------------------------------------------------------------------- #
# 1. freshness — the evidence SHA must equal the repository HEAD
# --------------------------------------------------------------------------- #
def test_freshness_true_when_sha_equals_repository_head():
    head = _repository_head()
    if head is None:
        pytest.skip("not inside a git repository")
    gates = report_hard_gate_set(git_sha=head)
    entry = gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]
    assert entry["value"] is True
    assert entry["status"] == "PASS"


def test_freshness_false_when_sha_mismatches_head():
    head = _repository_head()
    if head is None:
        pytest.skip("not inside a git repository")
    wrong = ("0" * 40) if head != ("0" * 40) else ("1" * 40)
    gates = report_hard_gate_set(git_sha=wrong)
    entry = gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]
    assert entry["value"] is False
    assert entry["status"] == "FAIL"
    # The message must name both SHAs (evidence bound to <sha> but HEAD is <sha>).
    assert f"evidence bound to {wrong}" in entry["check"]
    assert f"repository HEAD is {head}" in entry["check"]


def test_freshness_false_when_sha_none():
    gates = report_hard_gate_set(git_sha=None)
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is False


def test_freshness_false_for_unavailable_head_sentinel():
    # A generator whose git probe failed must record a real SHA or None — never
    # a sentinel ("UNKNOWN") that could compare equal to another failure value.
    head = _repository_head()
    if head is None:
        pytest.skip("not inside a git repository")
    gates = report_hard_gate_set(git_sha="UNKNOWN")  # current_head defaults to the real HEAD
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is False


def test_freshness_respects_explicit_current_head_override():
    gates = report_hard_gate_set(git_sha="abc123", current_head="abc123")
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is True
    gates2 = report_hard_gate_set(git_sha="abc123", current_head="def456")
    assert gates2["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is False


# --------------------------------------------------------------------------- #
# 2. gate report — honest status, no hard-coded PASS
# --------------------------------------------------------------------------- #
def test_all_gates_carry_an_honest_status_field():
    gates = report_hard_gate_set()
    assert len(gates) == 19
    for name, entry in gates.items():
        assert "status" in entry, name
        assert entry["status"] in {"PASS", "FAIL", "NOT_RUN"}, name
        assert isinstance(entry["value"], bool), name
        assert entry["check"], f"{name} has an empty check"


def test_unrunnable_gate_is_not_run_not_fake_pass():
    gates = report_hard_gate_set()
    entry = gates["MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION"]
    assert entry["value"] is False
    assert entry["status"] == "NOT_RUN"
    assert "NOT_RUN" in entry["check"]


def test_genuine_support_probes_run_real_fits():
    from modeling.evidence import _probe_asof_resolution, _probe_moe_support, _probe_regime_support

    assert _probe_regime_support() is True
    assert _probe_moe_support() is True
    assert _probe_asof_resolution() is True


def _load_model_gate_generator():
    path = Path(__file__).resolve().parents[2] / "scripts" / "generate_model_hard_gates.py"
    spec = importlib.util.spec_from_file_location("generate_model_hard_gates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_ledger(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_canonical_summary_treats_feature_inputs_as_declaration(tmp_path):
    _write_ledger(tmp_path / "MODEL_CANONICAL_LEDGER.csv", [{
        "canonical": "m", "lane": "MODEL_FEATURE_SCORE", "feature_inputs": "x,y",
        "stateful": "False", "evidence_sha": "head",
    }])
    summary = _load_model_gate_generator()._canonical_gate_summary(tmp_path, "head")
    typed = summary["MODEL_ALL_DIRECT_USE_HAVE_TYPED_INPUTS"]
    assert typed["status"] == "PASS"
    assert typed["canonical_executed"] == 1


def test_stateful_applicability_is_not_state_contract_proof(tmp_path):
    _write_ledger(tmp_path / "MODEL_CANONICAL_LEDGER.csv", [{
        "canonical": "m", "lane": "STATE_CONDITION_EVENT", "feature_inputs": "x",
        "stateful": "True", "evidence_sha": "head",
    }])
    summary = _load_model_gate_generator()._canonical_gate_summary(tmp_path, "head")
    gate = summary["MODEL_ALL_STATEFUL_HAVE_STATE_CONTRACT"]
    assert gate["status"] == "NOT_RUN"
    assert gate["canonical_total"] == 1
    assert gate["canonical_executed"] == 0


def test_stale_rows_remain_visible_in_totals(tmp_path):
    _write_ledger(tmp_path / "MODEL_CANONICAL_LEDGER.csv", [{
        "canonical": "m", "lane": "MODEL_FEATURE_SCORE", "feature_inputs": "x",
        "stateful": "False", "evidence_sha": "old",
    }])
    summary = _load_model_gate_generator()._canonical_gate_summary(tmp_path, "head")
    gate = summary["MODEL_ALL_DIRECT_USE_HAVE_TYPED_INPUTS"]
    assert gate["status"] == "NOT_RUN"
    assert gate["ledger_canonical_total"] == 1
    assert gate["fresh_canonical_total"] == 0
    assert "ledger_total=1 fresh_total=0" in gate["detail"]


# --------------------------------------------------------------------------- #
# 3. no silent fallback — authoritative trainer errors FAIL and propagate
# --------------------------------------------------------------------------- #
def test_production_trainer_error_fails_not_fallback(monkeypatch):
    ds = _small_dataset()
    lc = LabelContract(label_name="y", horizon_bars=1)
    clock = ashare_decision_clock()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("authoritative trainer exploded")

    monkeypatch.setattr("modeling.trainer.train_model", _boom)

    with pytest.raises(RuntimeError, match="authoritative trainer exploded"):
        run_walk_forward_evidence(
            PCRLearner, ds, _wf_spec(),
            label_contract=lc, decision_clock=clock,
            hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
            model_version="1.0", random_seed=0, min_folds=2,
        )


def test_predictor_error_fails_not_fallback(monkeypatch):
    # Trainer succeeds (stubbed with a real frozen artifact) so the flow reaches
    # the predictor; a predictor error must FAIL, never be papered over.
    from types import SimpleNamespace

    from modeling.leakage_guard import default_trainer_fn, synthetic_panel

    ds = synthetic_panel(n_dates=12, n_stocks=20, n_features=3, seed=7)
    lc = LabelContract(label_name="y", horizon_bars=1)
    clock = ashare_decision_clock()
    art = default_trainer_fn(ds)
    fake_result = SimpleNamespace(
        artifact=art, selected_hyperparams={"n_components": 3}, validation_best_rank_ic=0.1
    )
    monkeypatch.setattr("modeling.trainer.train_model", lambda **_kw: fake_result)

    def _boom_predict(*_args, **_kwargs):
        raise RuntimeError("predictor exploded")

    monkeypatch.setattr("modeling.predictor.Predictor.predict", _boom_predict)

    with pytest.raises(RuntimeError, match="predictor exploded"):
        run_walk_forward_evidence(
            PCRLearner, ds, _wf_spec(),
            label_contract=lc, decision_clock=clock,
            hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
            model_version="1.0", random_seed=0, min_folds=2,
        )
