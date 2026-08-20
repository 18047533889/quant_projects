# -*- coding: utf-8 -*-
"""Tests for evidence/registry.py — input dependency tracking and auto-staleness."""
from __future__ import annotations

from pathlib import Path

import pytest

from evidence.registry import (
    ARCHIVE_ROOT,
    CURRENT_DIR,
    EVIDENCE_ROOT,
    REPO_ROOT,
    EvidenceArtifact,
    FreshnessReport,
    archive_current,
    check_all_fresh,
    check_freshness,
    clear_registry,
    evidence_staleness_summary,
    get_artifact,
    init_registry,
    list_artifacts,
    publish_current,
    register_artifact,
    sha256_files,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Start each test with a clean live registry."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_artifact(
    artifact_id: str = "test-art",
    *,
    artifact_type: str = "inventory",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> EvidenceArtifact:
    inputs = inputs or ["operator_inventory.csv"]
    outputs = outputs or inputs
    art = EvidenceArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        description="test artifact",
        generator_script="scripts/r30_phase1_inventory.py",
        runner_command="python3 scripts/r30_phase1_inventory.py",
        output_files=outputs,
        input_dependencies=inputs,
    )
    art.evidence_input_identity = art.recompute_identity()
    art.generated_at = "2026-08-21T00:00:00+00:00"
    return art


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------


def test_register_and_list():
    art = _make_artifact("demo-inv", artifact_type="inventory")
    register_artifact(art)

    assert get_artifact("demo-inv") is art
    assert get_artifact("DEMO-INV") is art
    assert len(list_artifacts()) == 1
    assert list_artifacts("inventory") == [art]
    assert list_artifacts("benchmark") == []


def test_bad_type_rejected():
    art = _make_artifact("bad", artifact_type="banana")
    with pytest.raises(ValueError, match="Unsupported artifact_type"):
        register_artifact(art)


# ---------------------------------------------------------------------------
# Identity / staleness
# ---------------------------------------------------------------------------


def test_identity_changes_on_content_change(tmp_path):
    f = _touch(tmp_path / "src.txt", "v1")
    art = _make_artifact("id-test", inputs=[str(f)])
    register_artifact(art)
    original_identity = art.evidence_input_identity

    # Mutate source file
    f.write_text("v2", encoding="utf-8")
    new_identity = art.recompute_identity()
    assert new_identity != original_identity

    report = check_freshness(art)
    assert report.stale_count == 1
    assert report.stale_ids == ["id-test"]
    assert art.stale is True
    assert str(f) in art.stale_inputs


def test_identity_is_stable_when_input_unchanged(tmp_path):
    f = _touch(tmp_path / "input.txt", "steady")
    art = _make_artifact("steady", inputs=[str(f)])
    register_artifact(art)

    report = check_freshness(art)
    assert report.all_fresh
    assert art.stale is False


def test_missing_input_produces_deterministic_identity(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    art = _make_artifact("missing-input", inputs=[str(missing)])
    id1 = art.recompute_identity()
    id2 = art.recompute_identity()
    assert id1 == id2
    assert len(id1) == 64  # hex SHA-256


def test_sha256_files_order_independent(tmp_path):
    a = _touch(tmp_path / "a.txt", "alpha")
    b = _touch(tmp_path / "b.txt", "beta")
    assert sha256_files([a, b]) == sha256_files([b, a])


def test_check_all_fresh_summary(tmp_path):
    f1 = _touch(tmp_path / "a.txt", "ok")
    f2 = _touch(tmp_path / "b.txt", "ok")
    a1 = _make_artifact("f1", inputs=[str(f1)])
    a2 = _make_artifact("f2", inputs=[str(f2)])
    register_artifact(a1)
    register_artifact(a2)

    report = check_all_fresh()
    assert isinstance(report, FreshnessReport)
    assert report.total == 2
    assert report.all_fresh

    f2.write_text("changed", encoding="utf-8")
    report = check_all_fresh()
    assert report.stale_count == 1
    assert report.stale_ids == ["f2"]


# ---------------------------------------------------------------------------
# publish / archive
# ---------------------------------------------------------------------------


def test_publish_current_copies_outputs(tmp_path):
    out = _touch(tmp_path / "built.json", '{"ok": true}')
    art = _make_artifact(
        "pub-test",
        outputs=[str(out)],
        inputs=[str(out)],
    )
    register_artifact(art)

    dest = publish_current(art)
    assert dest == CURRENT_DIR
    copied = CURRENT_DIR / out.name
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == '{"ok": true}'


def test_archive_creates_snapshot(tmp_path):
    out = _touch(tmp_path / "snapshot.json", '{"snap": 1}')
    art = _make_artifact(
        "archive-test",
        outputs=[str(out)],
        inputs=[str(out)],
    )
    register_artifact(art)

    snapshot_dir = archive_current(art, snapshot="SNAP01")
    assert snapshot_dir == ARCHIVE_ROOT / "SNAP01"
    copied = snapshot_dir / out.name
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == '{"snap": 1}'


# ---------------------------------------------------------------------------
# Default registry bootstrap
# ---------------------------------------------------------------------------


def test_init_registry_populates_expected_entries():
    init_registry()
    artifacts = list_artifacts()
    ids = [a.artifact_id for a in artifacts]
    assert "operator-inventory" in ids
    assert "operator-certification" in ids
    assert "operator-benchmark-manifest" in ids

    for art in artifacts:
        assert art.evidence_input_identity, art.artifact_id
        assert art.artifact_type in {"inventory", "certification", "benchmark"}


def test_evidence_staleness_summary_shape():
    summary = evidence_staleness_summary()
    assert {"total", "stale_count", "stale_ids", "all_fresh", "details"} <= set(summary.keys())
    assert isinstance(summary["details"], dict)
