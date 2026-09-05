# -*- coding: utf-8 -*-
"""Tests for the R61-P0 fail-closed evidence gate (evidence/gate.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    from evidence import current as current_mod

    monkeypatch.setattr(current_mod, "DEFAULT_OUT", tmp_path / "CURRENT.json")
    (tmp_path / "CURRENT.json").write_text(json.dumps(payload), encoding="utf-8")


def test_gate_raises_stale_when_file_missing(tmp_path, monkeypatch) -> None:
    from evidence import current as current_mod
    from evidence.gate import StaleAgentOperatorEvidence, require_current_evidence

    monkeypatch.setattr(current_mod, "DEFAULT_OUT", tmp_path / "CURRENT.json")
    with pytest.raises(StaleAgentOperatorEvidence):
        require_current_evidence()


def test_gate_raises_when_artifact_stale(tmp_path, monkeypatch) -> None:
    from evidence import current as current_mod
    from evidence.gate import StaleAgentOperatorEvidence, require_current_evidence

    payload = {
        "schema_version": 3,
        "source_snapshot_id": "src:v1:deadbeefdeadbeef",
        "source_tree_identity": "stale-root",
        "sha_bindings": {},
        "artifacts": {
            art_id: {
                "artifact_id": art_id,
                "status": "CURRENT",
                "bound_input_identity": {
                    "source_snapshot_id": "src:v1:older0older0000"
                },
                "executed_at": "2026-09-01T00:00:00+00:00",
                "stale_inputs": [],
            }
            for art_id in current_mod.ARTIFACT_IDS
        },
    }
    _write_current(tmp_path, monkeypatch, payload)
    with pytest.raises(StaleAgentOperatorEvidence):
        require_current_evidence()


def test_gate_passes_when_bound_matches_live(tmp_path, monkeypatch) -> None:
    """When the bound snapshot equals the LIVE one and executed_at present."""
    from evidence import current as current_mod
    from evidence.gate import require_current_evidence

    # Build a payload whose artifact bound-input identity is seeded from the
    # LIVE identity computed by build_current() — the honest CURRENT shape.
    # build_current() itself must NOT see the tmp DEFAULT_OUT (it would fold
    # the fake prior), so write the file first, then compute live bindings,
    # then seed the bound identities from the real live state.
    live = current_mod.build_current()
    for art_id in live["artifacts"]:
        live["artifacts"][art_id]["bound_input_identity"] = {
            "source_snapshot_id": live["source_snapshot_id"],
            "ImplementationClosureHashSet": live["sha_bindings"]["ImplementationClosureHashSet"],
            "SemanticCatalogIdentity": live["sha_bindings"]["SemanticCatalogIdentity"],
            "PlatformSourceTreeIdentity": live["source_tree_identity"],
        }
        live["artifacts"][art_id]["executed_at"] = "2026-09-05T00:00:00+00:00"
    _write_current(tmp_path, monkeypatch, live)
    report = require_current_evidence()
    assert report["all_current"] is True
