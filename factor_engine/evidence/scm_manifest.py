# -*- coding: utf-8 -*-
"""R39 #75 —— Build-time SCMManifest：把部署的源码树绑定到提交，运行期无 .git。

问题
    证据模块（``backend.factor_operator_evidence``）的 ``inherited_runtime_audit``
    校验在运行期直接调用 ``git merge-base`` / ``git diff``。打进 wheel/容器后没有
    ``.git`` → 这些调用全部失败 → 证据 fail-closed → 部署不可用。

方案
    构建期（有 live .git 时）运行 :func:`generate_scm_manifest`，把以下事实固化成
    ``evidence/scm_manifest.json`` 随包分发：
        - ``build_commit_sha``：构建时 HEAD；
        - ``certified_commit_sha``：证据工件绑定的被审提交；
        - ``certified_is_ancestor``：被审提交是否 HEAD 的祖先（构建期算好）；
        - ``changed_since_certified``：``git diff --name-only certified..HEAD``
          的变更文件清单（构建期算好）。

运行期（无 .git）：证据模块读 manifest 取代这两次 git 调用。manifest 缺失（打包
漏了）且 .git 也不可用 → **fail-closed**（明确报 SCMManifest missing），绝不假装
通过。这符合「无 manifest 时 documented fail-closed fallback」。

注：blob sha 校验（``_git_blob_sha``）运行期无需 .git —— ``git hash-object`` 失败
时已回退到按文件字节手工计算 git blob sha，manifest 不承担这部分。
"""
from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
SCM_MANIFEST_PATH = FE_ROOT / "evidence" / "scm_manifest.json"


@dataclass(frozen=True)
class SCMManifest:
    """构建期固化的 SCM 事实（不可变）。"""

    build_commit_sha: str = ""
    certified_commit_sha: str = ""
    certified_is_ancestor: bool = False
    changed_since_certified: tuple[str, ...] = ()
    generated_at: str = ""
    source: str = "git"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SCMManifest | None":
        """从 JSON 加载；文件缺失 / 损坏 → None（调用方决定 fail-closed）。"""
        path = Path(path) if path is not None else SCM_MANIFEST_PATH
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            build_commit_sha=str(data.get("build_commit_sha") or ""),
            certified_commit_sha=str(data.get("certified_commit_sha") or ""),
            certified_is_ancestor=bool(data.get("certified_is_ancestor", False)),
            changed_since_certified=tuple(
                str(x) for x in (data.get("changed_since_certified") or [])
            ),
            generated_at=str(data.get("generated_at") or ""),
            source=str(data.get("source") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_commit_sha": self.build_commit_sha,
            "certified_commit_sha": self.certified_commit_sha,
            "certified_is_ancestor": self.certified_is_ancestor,
            "changed_since_certified": list(self.changed_since_certified),
            "generated_at": self.generated_at,
            "source": self.source,
        }


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def generate_scm_manifest(
    *,
    out_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """从 live .git 生成 SCMManifest（构建期调用）。

    ``certified_commit_sha`` 取证据工件的 ``certified_commit_sha``（inherited 工件）
    或 ``commit_sha``（direct 工件）。.git 不可用 → 抛 RuntimeError（构建期必须有
    确定性来源，绝不生成半成品 manifest）。
    """
    repo_root = Path(repo_root) if repo_root is not None else FE_ROOT.parent
    evidence_path = Path(evidence_path) if evidence_path is not None else (
        FE_ROOT / "evidence" / "factor_operator_verified.json"
    )
    out_path = Path(out_path) if out_path is not None else SCM_MANIFEST_PATH

    build_sha = _git(repo_root, "rev-parse", "HEAD")
    if not build_sha:
        raise RuntimeError("SCMManifest 生成需要 live .git（build-time）；rev-parse HEAD 失败")

    certified = ""
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        certified = str(
            payload.get("certified_commit_sha")
            or payload.get("commit_sha")
            or ""
        )
    except Exception:
        certified = ""

    certified_is_ancestor = False
    changed: list[str] = []
    if certified:
        ancestor = _git(
            repo_root, "merge-base", "--is-ancestor", certified, build_sha
        )
        certified_is_ancestor = ancestor is not None  # returncode 0 → 输出非 None（空串也算）
        # git merge-base --is-ancestor 成功时 stdout 为空串（strip 后 ''），
        # 但 _git 已把非 0 returncode 映射成 None，因此 is None 即非祖先。
        raw_changed = _git(
            repo_root, "diff", "--name-only", f"{certified}..{build_sha}", "--", "factor_engine"
        )
        if raw_changed:
            changed = [line for line in raw_changed.splitlines() if line.strip()]

    manifest = SCMManifest(
        build_commit_sha=build_sha,
        certified_commit_sha=certified,
        certified_is_ancestor=certified_is_ancestor,
        changed_since_certified=tuple(changed),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source="git",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest.to_dict()


__all__ = ["SCMManifest", "SCM_MANIFEST_PATH", "generate_scm_manifest"]
