from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import export_direct_mining_catalog as exporter


HEAD = "a" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_valid_artifacts(out_dir: Path, docs_dir: Path) -> None:
    fingerprint = {"head": HEAD, "dirty": False}
    catalog_rows = [
        {
            "canonical": "eligible_op",
            "direct_use_status": "direct_terminal",
            "directly_usable": True,
            "production_certified": True,
        },
        {
            "canonical": "retained_op",
            "direct_use_status": "direct_intermediate",
            "directly_usable": True,
            "production_certified": False,
        },
    ]
    _write_json(
        out_dir / "direct_mining_catalog.json",
        {
            "schema_version": "factor_engine.r18.direct_mining_catalog.v1",
            "fingerprint": fingerprint,
            "count": 2,
            "operators": catalog_rows,
        },
    )
    _write_json(
        out_dir / "direct_mining_manifest.json",
        {
            "schema_version": "factor_engine.r18.direct_mining_manifest.v1",
            "fingerprint": fingerprint,
            "count": 1,
            "operators": ["eligible_op"],
        },
    )
    _write_json(
        docs_dir / "R18_DIRECT_USE_MATRIX.json",
        {
            "schema_version": "factor_engine.r18.direct_use_matrix.v1",
            "fingerprint": fingerprint,
            "count": 3,
            "rows": catalog_rows
            + [{"canonical": "research_op", "direct_use_status": "research_tool"}],
        },
    )
    _write_json(
        docs_dir / "R18_OPERATOR_SMOKE_RECIPES.json",
        {
            "schema_version": "factor_engine.r18.operator_smoke_recipes.v1",
            "fingerprint": fingerprint,
            "count": 2,
            "recipes": [
                {"canonical": "eligible_op"},
                {"canonical": "retained_op"},
            ],
        },
    )


@pytest.fixture
def artifact_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    out_dir = tmp_path / "build" / "mining"
    docs_dir = tmp_path / "docs"
    _write_valid_artifacts(out_dir, docs_dir)
    monkeypatch.setattr(exporter, "_current_head", lambda: HEAD)
    return out_dir, docs_dir


def _run_check(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    docs_dir: Path,
) -> int:
    def registry_load_forbidden(**_: object) -> dict[str, int]:
        raise AssertionError("--check must not build or load the registry")

    monkeypatch.setattr(exporter, "build_all", registry_load_forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_direct_mining_catalog.py",
            "--check",
            "--out",
            str(out_dir),
            "--docs",
            str(docs_dir),
        ],
    )
    return exporter.main()


def test_check_accepts_consistent_artifacts_without_registry_load(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs

    assert _run_check(monkeypatch, out_dir, docs_dir) == 0
    assert "current and consistent" in capsys.readouterr().out


def test_check_fails_when_required_artifact_is_missing(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    (docs_dir / "R18_OPERATOR_SMOKE_RECIPES.json").unlink()

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    assert "missing required artifact" in capsys.readouterr().err


def test_check_fails_on_count_and_canonical_set_mismatch(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    manifest_path = out_dir / "direct_mining_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["count"] = 2
    payload["operators"] = ["retained_op"]
    _write_json(manifest_path, payload)

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    stderr = capsys.readouterr().err
    assert "count=2 does not match operators length=1" in stderr
    assert "manifest canonical set does not match eligible catalog rows" in stderr


def test_check_fails_on_stale_head_fingerprint(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    catalog_path = out_dir / "direct_mining_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["fingerprint"]["head"] = "b" * 40
    _write_json(catalog_path, payload)

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    stderr = capsys.readouterr().err
    assert "fingerprint.head=" in stderr
    assert f"does not match HEAD={HEAD!r}" in stderr


def test_check_fails_when_fingerprint_is_dirty(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    path = docs_dir / "R18_OPERATOR_SMOKE_RECIPES.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fingerprint"]["dirty"] = True
    _write_json(path, payload)

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    assert "fingerprint.dirty must be exactly false" in capsys.readouterr().err


def test_check_fails_on_wrong_schema_version(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    path = out_dir / "direct_mining_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "factor_engine.r18.direct_mining_manifest.v0"
    _write_json(path, payload)

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    stderr = capsys.readouterr().err
    assert "schema_version=" in stderr
    assert "does not match expected" in stderr


def test_check_fails_on_catalog_matrix_metadata_drift(
    artifact_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir, docs_dir = artifact_dirs
    path = docs_dir / "R18_DIRECT_USE_MATRIX.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["production_certified"] = False
    _write_json(path, payload)

    assert _run_check(monkeypatch, out_dir, docs_dir) == 1
    stderr = capsys.readouterr().err
    assert "catalog/matrix metadata mismatch" in stderr
    assert "eligible_op.production_certified" in stderr
