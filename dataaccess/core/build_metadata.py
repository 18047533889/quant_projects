"""
R32-P0-109: 版本改成SCM/tag自动驱动.
R32-P0-111: build SHA在build-time固化,不依赖运行时.git.
R32-P0-112: 当前HEAD CI evidence闭环.

Build metadata and version management.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class BuildInfo:
    """Build-time metadata固化.

    R32-P0-111: wheel/container中无.git很正常.
    构建生成_build_info.py:
    - version
    - sha
    - build id
    - build time
    - dirty flag

    Production缺build info → startup fail.
    """

    version: str
    build_sha: str
    build_id: str
    build_time: str
    dirty: bool
    tag: str | None = None
    branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "build_sha": self.build_sha,
            "build_id": self.build_id,
            "build_time": self.build_time,
            "dirty": self.dirty,
            "tag": self.tag,
            "branch": self.branch,
        }

    def validate_production_ready(self) -> None:
        """Validate build info is production-ready.

        Raises:
            ValidationError: If build info incomplete or dirty
        """
        if not self.build_sha:
            raise ValidationError(
                "R32-P0-111: Production requires build SHA. "
                "Build info incomplete."
            )

        if not self.version:
            raise ValidationError(
                "R32-P0-109: Production requires version. "
                "Build info incomplete."
            )

        if self.dirty:
            raise ValidationError(
                "R32-P0-111: Production禁止dirty build. "
                f"Build {self.build_id} has uncommitted changes."
            )

        if not self.build_time:
            raise ValidationError(
                "R32-P0-111: Production requires build timestamp."
            )


class ScmVersionResolver:
    """SCM/tag-driven version resolution.

    R32-P0-109: pyproject.toml不再人工改base version.
    使用setuptools-scm或等价:
    - tag: dataaccess-v0.11.0
    - non-tag: dev+sha
    - build-time _build_info.py

    Major/minor/patch由release policy/tag决定,不自动猜.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def get_version_from_git(self) -> str:
        """Get version from git tags.

        Returns:
            Version string: "0.11.0" for tagged, "0.11.0.dev123+gabcdef" for dev

        Raises:
            ValidationError: If git not available or repo not clean
        """
        try:
            # Get latest tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--long", "--dirty"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                # No tags yet
                return "0.0.0.dev0+unknown"

            describe = result.stdout.strip()

            # Parse git describe output: v0.11.0-5-gabcdef-dirty
            parts = describe.split("-")

            if len(parts) >= 3:
                tag = parts[0].lstrip("v")
                commits_since = parts[1]
                sha = parts[2]

                if "dirty" in describe:
                    return f"{tag}.dev{commits_since}+{sha}.dirty"
                elif commits_since == "0":
                    # Exact tag
                    return tag
                else:
                    # Dev version
                    return f"{tag}.dev{commits_since}+{sha}"

            return "0.0.0.dev0+unknown"

        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValidationError(
                f"R32-P0-109: Cannot determine version from git: {exc}"
            ) from exc

    def get_build_sha(self) -> str:
        """Get current commit SHA.

        Returns:
            Full 40-char SHA

        Raises:
            ValidationError: If git not available
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValidationError(
                f"R32-P0-111: Cannot determine build SHA from git: {exc}"
            ) from exc

    def is_dirty(self) -> bool:
        """Check if working tree is dirty."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return True  # Assume dirty if can't determine

    def get_current_tag(self) -> str | None:
        """Get current tag if on exact tag."""
        try:
            result = subprocess.run(
                ["git", "describe", "--exact-match", "--tags"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def get_current_branch(self) -> str | None:
        """Get current branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def generate_build_info(repo_root: Path | None = None) -> BuildInfo:
    """Generate build info at build time.

    R32-P0-111: Build-time generation, not runtime .git dependency.

    Args:
        repo_root: Repository root (defaults to auto-detect)

    Returns:
        BuildInfo with all fields populated
    """
    import time
    import uuid

    if repo_root is None:
        # Try to find repo root
        current = Path.cwd()
        while current != current.parent:
            if (current / ".git").exists():
                repo_root = current
                break
            current = current.parent
        else:
            raise ValidationError("Cannot find git repository root")

    resolver = ScmVersionResolver(repo_root)

    return BuildInfo(
        version=resolver.get_version_from_git(),
        build_sha=resolver.get_build_sha(),
        build_id=uuid.uuid4().hex,
        build_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dirty=resolver.is_dirty(),
        tag=resolver.get_current_tag(),
        branch=resolver.get_current_branch(),
    )


def write_build_info_module(build_info: BuildInfo, output_path: Path) -> None:
    """Write _build_info.py module.

    R32-P0-111: Generate at build time for wheel/container.
    """
    content = f'''"""
Build-time generated metadata.

Generated at build time, not runtime. Do not edit manually.
"""

__version__ = {build_info.version!r}
__build_sha__ = {build_info.build_sha!r}
__build_id__ = {build_info.build_id!r}
__build_time__ = {build_info.build_time!r}
__build_dirty__ = {build_info.dirty!r}
__build_tag__ = {build_info.tag!r}
__build_branch__ = {build_info.branch!r}
'''

    output_path.write_text(content)


def load_build_info() -> BuildInfo:
    """Load build info from _build_info module.

    Returns:
        BuildInfo from installed package

    Raises:
        ValidationError: In production if build info missing
    """
    try:
        from data_access import _build_info

        # Try new format first (R32-P0-111 generated)
        if hasattr(_build_info, "__build_sha__"):
            return BuildInfo(
                version=_build_info.__version__,
                build_sha=_build_info.__build_sha__,
                build_id=_build_info.__build_id__,
                build_time=_build_info.__build_time__,
                dirty=_build_info.__build_dirty__,
                tag=getattr(_build_info, "__build_tag__", None),
                branch=getattr(_build_info, "__build_branch__", None),
            )

        # Fall back to existing vcs-versioning format
        return BuildInfo(
            version=_build_info.__version__,
            build_sha=getattr(_build_info, "__commit_id__", None) or "unknown",
            build_id="legacy",
            build_time="unknown",
            dirty=False,
            tag=None,
            branch=None,
        )
    except (ImportError, AttributeError):
        # Development mode: generate on demand
        prod_mode = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
            "1",
            "true",
            "yes",
        }

        if prod_mode:
            raise ValidationError(
                "R32-P0-111: Production缺build info → startup fail. "
                "Package must be built with _build_info.py generation."
            )

        # Dev mode: try to generate from git
        try:
            return generate_build_info()
        except ValidationError:
            # No git available, return placeholder
            return BuildInfo(
                version="0.0.0.dev0+local",
                build_sha="unknown",
                build_id="dev",
                build_time="unknown",
                dirty=True,
            )


@dataclass(frozen=True)
class CiEvidence:
    """CI evidence bundle for current HEAD.

    R32-P0-112: 当前HEAD CI evidence闭环:
    - commit final code
    - clean clone
    - wheel install
    - tests
    - benchmarks
    - push
    - GitHub Actions
    - evidence绑定最终SHA
    """

    commit_sha: str
    workflow_run_id: str | None = None
    workflow_status: str | None = None
    tests_passed: bool | None = None
    benchmarks_passed: bool | None = None
    evidence_url: str | None = None

    def is_complete(self) -> bool:
        """Check if CI evidence is complete."""
        return all(
            [
                self.commit_sha,
                self.workflow_run_id,
                self.workflow_status == "success",
                self.tests_passed,
                self.benchmarks_passed,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "workflow_run_id": self.workflow_run_id,
            "workflow_status": self.workflow_status,
            "tests_passed": self.tests_passed,
            "benchmarks_passed": self.benchmarks_passed,
            "evidence_url": self.evidence_url,
            "complete": self.is_complete(),
        }


__all__ = [
    "BuildInfo",
    "ScmVersionResolver",
    "CiEvidence",
    "generate_build_info",
    "write_build_info_module",
    "load_build_info",
]
