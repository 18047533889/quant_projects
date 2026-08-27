"""Canonical build-time identity for DataAccess."""
from __future__ import annotations
import hashlib, os, re, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from data_access.core.exceptions import ValidationError
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_UNKNOWN = {"", "unknown", "none", "null", "dev", "local", "untagged"}

#: Fallback marker read from the generated module when a dev fallback is
#: checked in (see data_access/_build_info.py ``BUILT_AT_RUNTIME_FALLBACK``).
_FALLBACK_MARKER = "BUILT_AT_RUNTIME_FALLBACK"

#: Repository root used ONLY by the development/production gate that compares
#: the runtime build_sha with the live git HEAD.  Never consulted in an
#: installed artifact without a checkout (the env variable allows setting it
#: explicitly in deployment).
_DEV_REPO_ROOT = os.environ.get("QUANT_REPO_ROOT", "").strip() or (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".git",
        )
    )
    else ""
)


def _validate_production_revision_matches(info: BuildInfo) -> None:
    """P0-04 production gate: build_sha must match the build's git HEAD.

    Only enforced when a git checkout is discoverable (dev checkout or the
    explicitly-configured deployment repo root).  A checked-in fallback module
    is always rejected in production (an unknown-A build must never run).
    """
    try:
        from data_access import _build_info as _gen
    except ImportError:
        _gen = None
    if _gen is not None and getattr(_gen, _FALLBACK_MARKER, False):
        raise ValidationError(
            "production build rejected: data_access build metadata is a checked-in "
            f"{_FALLBACK_MARKER} (BUILT_AT_RUNTIME_FALLBACK=True); regenerate "
            "via scripts/regenerate_build_info.py before production packaging"
        )
    repo_root = _DEV_REPO_ROOT
    if not repo_root:
        # No checkout discoverable: the generated file is the authority; the
        # build-time generator guarantees build_sha matches build-time HEAD, so
        # equality cannot be evaluated here.
        return
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError:
        return
    if result.returncode != 0:
        return
    head = result.stdout.strip()
    if head and info.build_sha != head:
        raise ValidationError(
            "production build metadata does not match git HEAD: "
            f"build_sha={info.build_sha!r} head={head!r}"
        )

def validate_revision(value: str, *, field: str = "build_sha") -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value.strip()):
        raise ValidationError(f"{field} must be a full 40-character hexadecimal revision")
    return value.strip().lower()

def _required(value: Any, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in _UNKNOWN: raise ValidationError(f"build metadata field {field!r} is unavailable")
    return text

@dataclass(frozen=True)
class BuildInfo:
    version: str | None
    build_sha: str | None
    build_id: str | None
    build_time: str | None
    dirty: bool | None
    dependency_lock_hash: str | None = None
    tag: str | None = None
    branch: str | None = None
    def validate_production_ready(self) -> None:
        version = _required(self.version, "version")
        if not _VERSION_RE.fullmatch(version): raise ValidationError(f"invalid package version {version!r}")
        validate_revision(_required(self.build_sha, "build_sha"))
        if not _required(self.build_id, "build_id"): raise ValidationError("build_id unavailable")
        stamp = _required(self.build_time, "build_time")
        try: datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as exc: raise ValidationError(f"invalid build_time {stamp!r}") from exc
        if self.dirty is not False: raise ValidationError("production build metadata must have dirty=False")
    @property
    def available(self) -> bool:
        try: self.validate_production_ready()
        except ValidationError: return False
        return True
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()

def _generated_build_info() -> BuildInfo:
    try: import importlib; _build_info = importlib.import_module("data_access._build_info")
    except (ImportError, AttributeError): return BuildInfo(None, None, None, None, None)
    return BuildInfo(getattr(_build_info, "__version__", None), getattr(_build_info, "__build_sha__", getattr(_build_info, "__commit_id__", None)), getattr(_build_info, "__build_id__", None), getattr(_build_info, "__build_time__", None), getattr(_build_info, "__build_dirty__", None), getattr(_build_info, "__dependency_lock_hash__", None), getattr(_build_info, "__build_tag__", None), getattr(_build_info, "__build_branch__", None))

def validate_version_match(observed: str | None, expected: str | None) -> str:
    """Validate that two package-version claims identify the same release.

    Version claims are part of the build identity; silently accepting a wheel
    whose generated metadata disagrees with its declared package version can
    reuse incompatible cache/results.  Keep this check typed and independent
    of Git so it is safe at runtime.
    """
    observed_text = _required(observed, "observed version")
    expected_text = _required(expected, "expected version")
    if not _VERSION_RE.fullmatch(observed_text):
        raise ValidationError(f"invalid observed package version {observed_text!r}")
    if not _VERSION_RE.fullmatch(expected_text):
        raise ValidationError(f"invalid expected package version {expected_text!r}")
    if observed_text != expected_text:
        raise ValidationError(
            "package version mismatch: "
            f"generated={observed_text!r}, declared={expected_text!r}"
        )
    return observed_text


def load_build_info(*, production: bool | None = None) -> BuildInfo:
    if production is None: production = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
    info = _generated_build_info()
    # Runtime identity is read only from the build-time generated module.  Do
    # not consult package metadata or SCM here: an installed artifact may not
    # have either a matching distribution database or a checkout beside it.
    if production:
        info.validate_production_ready()
        # P0-04: a production build must carry the exact revision it was built
        # from.  A checked-in fallback (BUILT_AT_RUNTIME_FALLBACK) or a
        # build_sha that does not equal the build-time HEAD is a hard failure
        # (never silently degrade to unknown-A metadata).
        _validate_production_revision_matches(info)
    return info

class ScmVersionResolver:
    """Build-only Git resolver; runtime metadata never calls this class."""
    def __init__(self, repo_root: Path): self.repo_root = repo_root
    def _run(self, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=self.repo_root, capture_output=True, text=True, check=False)
        if result.returncode: raise ValidationError(f"Git build metadata command failed: {' '.join(args)}")
        return result.stdout.strip()
    def get_build_sha(self) -> str: return validate_revision(self._run("rev-parse", "HEAD"), field="commit_sha")
    def get_version_from_git(self) -> str:
        value = self._run("describe", "--tags", "--long", "--dirty").lstrip("v")
        return value.split("-", 1)[0] if value.count("-") < 2 else value
    def is_dirty(self) -> bool: return bool(self._run("status", "--porcelain"))
    def get_current_tag(self) -> str | None: return None
    def get_current_branch(self) -> str | None: return self._run("rev-parse", "--abbrev-ref", "HEAD") or None

def generate_build_info(repo_root: Path | None = None) -> BuildInfo:
    resolver = ScmVersionResolver(repo_root or Path.cwd())
    sha = resolver.get_build_sha()
    version = resolver.get_version_from_git()
    dirty = resolver.is_dirty()
    tag = resolver.get_current_tag()
    branch = resolver.get_current_branch()
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            build_time = datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("SOURCE_DATE_EPOCH must be a valid Unix timestamp") from exc
    else:
        build_time = resolver._run("show", "-s", "--format=%cI", "HEAD")
        try:
            build_time = datetime.fromisoformat(build_time.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError as exc:
            raise ValidationError(f"invalid Git commit timestamp {build_time!r}") from exc
    identity = "|".join(str(value or "") for value in (version, sha, build_time, dirty, tag))
    build_id = f"{sha[:12]}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"
    return BuildInfo(version, sha, build_id, build_time, dirty, tag=tag, branch=branch)

def write_build_info_module(build_info: BuildInfo, output_path: Path) -> None:
    build_info.validate_production_ready()
    values = {"__version__": build_info.version, "__build_sha__": build_info.build_sha, "__build_id__": build_info.build_id, "__build_time__": build_info.build_time, "__build_dirty__": build_info.dirty, "__dependency_lock_hash__": build_info.dependency_lock_hash, "__build_tag__": build_info.tag, "__build_branch__": build_info.branch}
    output_path.write_text('"""Generated at build time. Do not edit manually."""\n' + "\n".join(f"{k} = {v!r}" for k, v in values.items()) + "\n", encoding="utf-8")

@dataclass(frozen=True)
class CiEvidence:
    commit_sha: str
    workflow_run_id: str | None = None
    workflow_status: str | None = None
    tests_passed: bool | None = None
    benchmarks_passed: bool | None = None
    evidence_url: str | None = None
    def is_complete(self) -> bool: return all([self.commit_sha, self.workflow_run_id, self.workflow_status == "success", self.tests_passed, self.benchmarks_passed])
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "complete": self.is_complete()}
__all__ = ["BuildInfo", "ScmVersionResolver", "CiEvidence", "generate_build_info", "write_build_info_module", "load_build_info", "validate_revision", "validate_version_match"]
