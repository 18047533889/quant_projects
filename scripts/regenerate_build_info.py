"""P0-04: regenerate ``data_access/_build_info.py`` from the live Git tree.

Run at PEP 517 build time (via the ``data_access`` private build backend, or
manually from the repo root) so an installed ``data_access`` wheel always
carries build facts matching the exact revision it was built from.

Generation sources (in precedence order):
  * environment overrides ``QUANT_BUILD_VERSION`` only for the *version*
    string (packaging may want a release version independent of SCM state);
  * git HEAD (``rev-parse --short HEAD``) for build_sha;
  * commit timestamps for build_time;
  * dirty flag from ``git status --porcelain``;
  * tag/branch from ``git describe --tags`` / ``rev-parse --abbrev-ref HEAD``;
  * dependency lock hash from the canonical lock file listed in the project
    metadata (``requirements-production.lock`` body, dedup of lock_digest line).

The regenerated module is the SINGLE runtime authority: ``data_access.__init__
`` reads version/build info only from this file (see ``core/build_metadata.py``).
A checked-in fallback is marked ``BUILT_AT_RUNTIME_FALLBACK`` and the
production gate refuses to run with ``dirty=True`` or a build_sha that does not
equal the current git HEAD.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = REPO_ROOT / "data_access" / "_build_info.py"

_PROJECT_VERSION = "0.10.2"
#: Canonical lock file whose body digest is embedded as dependency_lock_hash.
LOCK_FILE = REPO_ROOT / "requirements-production.lock"
#: Fallback lock (data_access-specific constraints) used when the umbrella lock
#: is absent (standalone data_access build).
_LOCK_FALLBACK = REPO_ROOT / "data_access" / "constraints" / "py310.txt"

FALLBACK_MARKER = "BUILT_AT_RUNTIME_FALLBACK"


def _git(*args: str) -> str:
    """Run git with output-capture; raise on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_safe(*args: str) -> str | None:
    """Like ``_git`` but returns None on failure (no tags etc.)."""
    try:
        return _git(*args)
    except RuntimeError:
        return None


def _short_sha() -> str:
    value = _git("rev-parse", "--short=12", "HEAD")
    return value


def _full_sha() -> str:
    return _git("rev-parse", "HEAD")


def _is_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def _commit_time() -> str:
    raw = _git("show", "-s", "--format=%cI", "HEAD")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dependency_lock_hash() -> str | None:
    for path in (LOCK_FILE, _LOCK_FALLBACK):
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            body = [ln for ln in lines if not ln.startswith("# lock_digest")]
            content = "\n".join(body) + "\n"
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
    return None


def _build_id(sha: str, version: str, build_time: str, dirty: bool) -> str:
    identity = "|".join([version, sha, build_time, str(dirty)])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{sha[:12]}-{digest}"


def generate() -> dict[str, object]:
    """Collect the live build facts for the current checkout."""
    sha = _full_sha()
    version = os.environ.get("QUANT_BUILD_VERSION") or _PROJECT_VERSION
    dirty = _is_dirty()
    commit_time = _commit_time()
    tag = _git_safe("describe", "--tags", "--exact-match", "HEAD") or _git_safe(
        "describe", "--tags", "--long"
    )
    branch = _git_safe("rev-parse", "--abbrev-ref", "HEAD")
    lock_hash = _dependency_lock_hash()
    build_id = _build_id(sha, version, commit_time, dirty)
    return {
        "__version__": version,
        "__build_sha__": sha,
        "__build_id__": build_id,
        "__build_time__": commit_time,
        "__build_dirty__": bool(dirty),
        "__dependency_lock_hash__": lock_hash,
        "__build_tag__": tag,
        "__build_branch__": branch,
    }


def _render(values: dict[str, object], *, fallback: bool = False) -> str:
    lines = [
        '"""Generated at build time. Do not edit manually."""',
        "from __future__ import annotations",
        "",
    ]
    if fallback:
        lines.append(f"{FALLBACK_MARKER} = True")
        lines.append("# This is the checked-in dev fallback; the production gate")
        lines.append("# rejects a dirty/unknown-A build. Regenerate via")
        lines.append("# scripts/regenerate_build_info.py before packaging.")
        lines.append("")
    else:
        lines.append(f"{FALLBACK_MARKER} = False")
        lines.append("")
    for key, value in values.items():
        lines.append(f"{key} = {value!r}")
    lines.append("")
    lines.append("version = __version__")
    lines.append("__version_tuple__ = version_tuple = tuple(int(part) for part in version.split('.')[:3])")
    lines.append("commit_id = __commit_id__ = __build_sha__")
    lines.append("")
    return "\n".join(lines)


def write(output: Path | None) -> Path:
    """Regenerate and write ``_build_info.py`` to *output* (default data_access/)."""
    target = output or OUTPUT_DEFAULT
    values = generate()
    target.write_text(_render(values, fallback=False), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"output path (default {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="emit the checked-in runtime fallback instead of live facts",
    )
    args = parser.parse_args(argv)
    target = args.output or OUTPUT_DEFAULT
    if args.fallback:
        values = {
            "__version__": _PROJECT_VERSION,
            "__build_sha__": "0" * 40,
            "__build_id__": "fallback-" + time.strftime("%Y%m%d"),
            "__build_time__": "1970-01-01T00:00:00Z",
            "__build_dirty__": True,
            "__dependency_lock_hash__": None,
            "__build_tag__": None,
            "__build_branch__": "unknown",
        }
        target.write_text(_render(values, fallback=True), encoding="utf-8")
    else:
        write(target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())