# -*- coding: utf-8 -*-
"""R61-P0 #64: fail-closed evidence gate wiring in the mining operator allowlist.

Covers ``factor_engine.api.mining_integration.get_mining_operators_from_manifest``
(gate wired for production/cold_start BEFORE returning operators) plus the
cold-start sibling ``factor_engine.mining.direct_use.get_direct_use_mining_operators_from_manifest``.

Fabricated-stale pattern mirrors the repo-root ``tests/test_evidence_gate.py``:
monkeypatch ``evidence.current.DEFAULT_OUT`` to a tmp file containing a stale
CURRENT.json so the gate cannot prove freshness.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings

import pytest

# The FE test conftest (factor_engine/tests/conftest.py) imports
# ``factor_engine.mining`` at module scope, which triggers the P0-03 legacy-name
# alias finder in ``factor_engine/deprecated_shims.py``.  That finder maps the
# unqualified name ``evidence`` to the DATA-ONLY namespace dir
# ``factor_engine/evidence/`` (an artifacts mount, not the top-level evidence
# package), so once it is installed a plain ``import evidence.current`` can never
# reach /home/sunhaiwei/quant_projects/evidence.  We therefore load the REAL
# top-level ``evidence`` package into ``sys.modules`` by explicit spec BEFORE any
# code imports factor_engine, and never re-import it under the legacy name.
if "evidence" not in sys.modules or "factor_engine" in str(getattr(sys.modules.get("evidence"), "__file__", "")):
    _ev_spec = importlib.util.spec_from_file_location(
        "evidence",
        "/home/sunhaiwei/quant_projects/evidence/__init__.py",
        submodule_search_locations=["/home/sunhaiwei/quant_projects/evidence"],
    )
    assert _ev_spec is not None and _ev_spec.submodule_search_locations is not None
    _ev = importlib.util.module_from_spec(_ev_spec)
    sys.modules["evidence"] = _ev
    assert _ev_spec.loader is not None
    _ev_spec.loader.exec_module(_ev)

import evidence.current as current_mod  # noqa: E402  (top-level package pinned by spec)
import evidence.gate as gate_mod  # noqa: E402
from evidence.gate import StaleAgentOperatorEvidence, require_current_evidence

from factor_engine.api.mining_integration import get_mining_operators_from_manifest
from factor_engine.mining.direct_use import (  # noqa: F401
    get_direct_use_mining_operators_from_manifest,
)


def _stub_stale_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route current_evidence_report to a deterministic STALE summary.

    Using this (instead of the DEFAULT_OUT tmp-file pattern) keeps the test
    hermetic — no heavy live-tree identity walk.  The repo-root gate test covers
    the real evaluate() path; here we only need require_current_evidence's
    fail-closed decision + the wiring order in the allowlist functions.
    """
    monkeypatch.setattr(gate_mod, "current_evidence_report", lambda: {
        "all_current": False,
        "status": "STALE",
        "summary": {"current": 0, "total": 3, "stale": 3, "failed": 0, "not_run": 0},
        "changed_artifacts": ["operator-inventory"],
    })


def _stub_current_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate_mod, "current_evidence_report", lambda: {
        "all_current": True,
        "status": "CURRENT",
        "summary": {"current": 3, "total": 3, "stale": 0, "failed": 0, "not_run": 0},
        "changed_artifacts": [],
    })


def _stub_manifest(monkeypatch: pytest.MonkeyPatch, result: list[str]) -> None:
    """Prevent the mining-layer manifest machinery from running.

    ``get_mining_operators_from_manifest`` imports the mining-layer loader
    lazily, so stubbing ``factor_engine.mining.direct_use`` functions is enough
    — a test that must reach past the evidence gate needs no real manifest file.
    """
    monkeypatch.setattr(
        "factor_engine.mining.direct_use.validate_direct_use_manifest",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "factor_engine.mining.direct_use.load_validated_manifest",
        lambda *a, **k: {"operators": result},
    )
    # R61-P0 #64: the api entry now runs the evidence gate BEFORE the manifest
    # stub, so the stub below lets tests that reach past the gate return the
    # stub operator list without touching a real manifest file.
    monkeypatch.setattr(
        "factor_engine.mining.direct_use.get_direct_use_mining_operators_from_manifest",
        lambda *a, **k: result,
    )


def _stub_sibling_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the mining-layer manifest internals (NOT the entry under test)."""
    monkeypatch.setattr(
        "factor_engine.mining.direct_use.validate_direct_use_manifest",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "factor_engine.mining.direct_use.load_validated_manifest",
        lambda *a, **k: {"operators": ["ts_mean"]},
    )


# ---------------------------------------------------------------------------
# production mode + stale evidence -> StaleAgentOperatorEvidence
# ---------------------------------------------------------------------------


def test_production_mode_stale_evidence_raises(tmp_path, monkeypatch) -> None:
    """The manifest function must fail closed BEFORE returning operators."""
    _stub_stale_report(monkeypatch)
    with pytest.raises(StaleAgentOperatorEvidence, match="CURRENT"):
        get_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="production",
            market="ashare",
        )


def test_cold_start_mode_stale_evidence_raises(tmp_path, monkeypatch) -> None:
    _stub_stale_report(monkeypatch)
    with pytest.raises(StaleAgentOperatorEvidence, match="CURRENT"):
        get_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="cold_start",
            market="ashare",
        )


def test_manifest_allow_stale_does_not_bypass_gate_in_production(
    tmp_path, monkeypatch
) -> None:
    """allow_stale=True relaxes the manifest fingerprint, NOT the evidence gate."""
    _stub_stale_report(monkeypatch)
    _stub_manifest(monkeypatch, ["ts_mean", "ts_std"])
    with pytest.raises(StaleAgentOperatorEvidence, match="CURRENT"):
        get_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="production",
            allow_stale=True,
            market="ashare",
        )


# ---------------------------------------------------------------------------
# research mode + stale evidence -> no raise (gate evaluated only)
# ---------------------------------------------------------------------------


def test_research_mode_stale_evidence_no_raise(tmp_path, monkeypatch) -> None:
    _stub_stale_report(monkeypatch)
    _stub_manifest(monkeypatch, ["ts_mean", "ts_std"])
    ops = get_mining_operators_from_manifest(
        str(tmp_path / "manifest.json"),
        run_mode="research",
        market="ashare",
    )
    assert ops == ["ts_mean", "ts_std"]


def test_research_mode_stale_evidence_no_raise_no_context(
    tmp_path, monkeypatch
) -> None:
    """Gate bypass at the mining layer without registry filtering."""
    _stub_stale_report(monkeypatch)
    _stub_manifest(monkeypatch, ["ts_mean", "ts_std"])
    ops = get_direct_use_mining_operators_from_manifest(
        str(tmp_path / "manifest.json"),
        run_mode="research",
        context=None,
    )
    assert ops == ["ts_mean", "ts_std"]


def test_research_mode_current_evidence_evaluates(tmp_path, monkeypatch) -> None:
    """research calls the gate with allow_stale=True (evaluate, never raise)."""
    _stub_stale_report(monkeypatch)
    _stub_manifest(monkeypatch, ["ts_rank"])
    report = require_current_evidence(allow_stale=True)
    assert report["all_current"] is False
    ops = get_direct_use_mining_operators_from_manifest(
        str(tmp_path / "manifest.json"),
        run_mode="research",
        context=None,
    )
    assert ops == ["ts_rank"]


# ---------------------------------------------------------------------------
# cold-start sibling in mining/direct_use.py
# ---------------------------------------------------------------------------


def test_cold_start_sibling_stale_evidence_raises(tmp_path, monkeypatch) -> None:
    """The mining-layer cold-start sibling is a separate production surface.

    The evidence gate must fire BEFORE the manifest machinery; stub the manifest
    so the only failure available is the evidence one.
    """
    _stub_stale_report(monkeypatch)
    _stub_sibling_manifest(monkeypatch)
    with pytest.raises(StaleAgentOperatorEvidence, match="CURRENT"):
        get_direct_use_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="cold_start",
        )


# ---------------------------------------------------------------------------
# FACTOR_ENGINE_EVIDENCE_GATE=off -> no raise in production mode
# ---------------------------------------------------------------------------


def test_off_switch_disables_raise_in_production(tmp_path, monkeypatch) -> None:
    _stub_stale_report(monkeypatch)
    _stub_manifest(monkeypatch, ["ts_mean"])
    monkeypatch.setenv("FACTOR_ENGINE_EVIDENCE_GATE", "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops = get_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="production",
            market="ashare",
        )
    assert ops == ["ts_mean"]
    assert any(
        "FACTOR_ENGINE_EVIDENCE_GATE=off" in str(w.message) for w in caught
    ), "off-switch must emit a warning"


def test_off_switch_requires_exact_value(tmp_path, monkeypatch) -> None:
    """Any value other than the exact 'off' stays fail-closed."""
    _stub_stale_report(monkeypatch)
    monkeypatch.setenv("FACTOR_ENGINE_EVIDENCE_GATE", "1")
    with pytest.raises(StaleAgentOperatorEvidence, match="CURRENT"):
        get_mining_operators_from_manifest(
            str(tmp_path / "manifest.json"),
            run_mode="production",
            market="ashare",
        )
