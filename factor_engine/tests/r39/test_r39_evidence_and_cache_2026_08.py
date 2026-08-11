# -*- coding: utf-8 -*-
"""R39 #75 / #72 —— 证据模块无 .git 可用 + autopilot cache 消费者接线。

#75
    ``backend.factor_operator_evidence`` 的 inherited 校验运行期直接调 git；打进
    wheel/container 无 .git → fail-closed → 部署不可用。构建期生成 SCMManifest，
    运行期无 .git 时读 manifest 取代 ``git merge-base`` / ``git diff``：
      - 有 manifest + build==certified（或 certified 是 build 祖先）→ 不报
        「certified commit unavailable / cannot enumerate changes」；
      - 无 manifest 且无 .git → fail-closed（明确报 SCMManifest missing）；
      - manifest 记录 certified 不是祖先 → fail-closed。

#72
    ResourceAutopilotService._apply_cache_consumers 把 decision.cache_budget_bytes
    应用到 DA QueryResultCache（动态 grow/shrink 真实接线）。
"""
from __future__ import annotations

import json

import pytest

import backend.factor_operator_evidence as foe
import backend.evidence_provenance as ep
import evidence.scm_manifest as sm


def _patch_heavy_deps(monkeypatch):
    """跳过证据校验里重量级的 operator registry / 全量 hash 计算（聚焦 git path）。"""
    monkeypatch.setattr(foe, "_production_sets", lambda: (set(), set()))
    monkeypatch.setattr(ep, "evidence_artifact_valid", lambda *a, **k: True)
    monkeypatch.setattr(ep, "implementation_hashes_for", lambda *a, **k: {})


def _write_manifest(
    path,
    *,
    build: str,
    certified: str,
    ancestor: bool,
    changed: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "build_commit_sha": build,
                "certified_commit_sha": certified,
                "certified_is_ancestor": ancestor,
                "changed_since_certified": list(changed or []),
                "generated_at": "2026-08-11T00:00:00+00:00",
                "source": "git",
            }
        ),
        encoding="utf-8",
    )


def _minimal_payload(certified: str = "aaaa1111") -> dict:
    return {
        "certification_mode": "inherited_runtime_audit",
        "certified_commit_sha": certified,
        "allowed_post_certification_blob_shas": {},
        "allowed_unhashed_artifact_paths": [],
        "audited_factor_canonicals": [],
        "audited_factor_implementation_hashes": {},
    }


# ---------------------------------------------------------------------------
# #75 evidence without .git
# ---------------------------------------------------------------------------


def test_evidence_without_git_uses_scm_manifest(tmp_path, monkeypatch):
    """有 SCMManifest（build==certified）→ 不报 git-only 错误。"""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="aaaa1111", certified="aaaa1111", ancestor=True)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    errors = foe._inherited_validation_errors(_minimal_payload())
    assert (
        "certified commit is unavailable or is not an ancestor of the checkout"
        not in errors
    )
    assert "cannot enumerate post-certification FactorEngine changes" not in errors
    assert not any("SCMManifest missing" in e for e in errors)


def test_evidence_without_git_and_manifest_fails_closed(tmp_path, monkeypatch):
    """无 .git 且无 manifest → 明确 fail-closed，绝不假装通过。"""
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", tmp_path / "missing.json")

    errors = foe._inherited_validation_errors(_minimal_payload())
    assert any("SCMManifest missing" in e for e in errors)


def test_evidence_without_git_manifest_not_ancestor_fails_closed(tmp_path, monkeypatch):
    """manifest 记录 certified 不是 build 祖先 → fail-closed。"""
    manifest_path = tmp_path / "scm_manifest.json"
    _write_manifest(manifest_path, build="bbbb2222", certified="aaaa1111", ancestor=False)
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(foe, "_git_available", lambda: False)
    monkeypatch.setattr(sm, "SCM_MANIFEST_PATH", manifest_path)

    errors = foe._inherited_validation_errors(_minimal_payload())
    assert any("not an ancestor of the build commit" in e for e in errors)


def test_generate_scm_manifest_writes(tmp_path):
    """build-time 生成：有 live .git 时能写出含 build_commit_sha 的 manifest。"""
    out = tmp_path / "scm_manifest.json"
    data = sm.generate_scm_manifest(out_path=out)
    assert data["build_commit_sha"]
    assert out.is_file()
    loaded = sm.SCMManifest.load(out)
    assert loaded is not None
    assert loaded.build_commit_sha == data["build_commit_sha"]


# ---------------------------------------------------------------------------
# #72 autopilot cache 消费者 → DA QueryResultCache
# ---------------------------------------------------------------------------


def test_autopilot_cache_consumer_applies_budget(tmp_path, monkeypatch):
    from data_access.read.query_cache import (
        get_query_cache,
        reset_query_cache,
    )
    from runtime.resource_autopilot_service import ResourceAutopilotService

    reset_query_cache()
    try:
        cache = get_query_cache()
        cache.set_with_size("k", __import__("pyarrow").table({"a": [1]}), nbytes=100)
        assert cache.max_bytes != 12345
        broker = type("B", (), {"_resource_controller": lambda self: None})()
        svc = ResourceAutopilotService(broker=broker)
        decision = type(
            "D", (), {"cache_budget_bytes": 12345}
        )()
        svc._apply_cache_consumers(decision)
        assert cache.max_bytes == 12345
    finally:
        reset_query_cache()
