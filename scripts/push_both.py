#!/usr/bin/env python3
"""Safely publish committed root-main subtrees to the 13 mirror repositories."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(os.environ.get("PUSH_BOTH_ROOT", "/home/sunhaiwei/quant_projects"))
CACHE = Path(os.environ.get("PUSH_BOTH_CACHE", str(Path.home() / ".cache/qs_sync")))
ORG = "HKUST-QUANT-SOCIETY"
ROOT_OWNER = "18047533889"
REPOS = ["factor_engine", "data_access", "vectorbt_qs", "riskfolio_qs",
         "quant_evaluator", "quant_platform", "modeling", "factor_preprocess",
         "factor_optimizer", "factor_assets", "alphaprobe", "platform_web", "lightgbm_qs"]
ALIASES = {"fe": "factor_engine"}


class SyncError(RuntimeError):
    pass


def sanitize(text: str) -> str:
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", text)
    return re.sub(r"(gh[opsu]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "***", text)


def run(args, *, cwd=None, env=None, check=True) -> str:
    p = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if check and p.returncode:
        raise SyncError(f"command failed ({args[0]}): {sanitize(p.stderr).strip()}")
    return p.stdout.strip()


def git(repo: Path, *args: str, env=None, check=True) -> str:
    return run(["git", "-C", str(repo), *args], env=env, check=check)


def remote_identity(url: str) -> tuple[str, str, str]:
    # Accept HTTPS and git@host:owner/name.git without ever returning credentials.
    if re.match(r"^[^/@]+@[^:]+:", url):
        host, path = url.split("@", 1)[1].split(":", 1)
    else:
        parsed = urlparse(url)
        host, path = (parsed.hostname or ""), parsed.path.lstrip("/")
    parts = path.removesuffix(".git").split("/")
    if len(parts) != 2:
        raise SyncError("remote URL must identify exactly one GitHub owner/repository")
    return host.lower(), parts[0], parts[1]


def verify_remote(repo: Path, remote: str, owner: str, name: str) -> None:
    url = git(repo, "remote", "get-url", remote)
    if remote_identity(url) != ("github.com", owner, name):
        raise SyncError(f"unexpected {remote} target; expected github.com/{owner}/{name}")


def verify_push_remote(repo: Path, remote: str, owner: str, name: str) -> None:
    url = git(repo, "remote", "get-url", "--push", remote)
    if remote_identity(url) != ("github.com", owner, name):
        raise SyncError(f"unexpected {remote} push target; expected github.com/{owner}/{name}")


def object_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    root_objects = git(root, "rev-parse", "--git-path", "objects")
    if not os.path.isabs(root_objects):
        root_objects = str(root / root_objects)
    prior = env.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
    env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = root_objects + ((os.pathsep + prior) if prior else "")
    return env


def source_tree(root: Path, commit: str, name: str) -> str:
    obj = git(root, "rev-parse", f"{commit}:{name}")
    if git(root, "cat-file", "-t", obj) != "tree":
        raise SyncError(f"root main entry {name} is not a directory tree")
    return obj


def preview(repo: Path, old: str, tree: str, env: dict[str, str]) -> tuple[dict[str, int], list[dict[str, str]]]:
    counts = {"add": 0, "modify": 0, "delete": 0}
    changes = []
    out = git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", old, tree, env=env)
    for line in out.splitlines():
        status = line.split("\t", 1)[0][:1]
        key = {"A": "add", "D": "delete"}.get(status, "modify")
        counts[key] += 1
        changes.append({"status": status, "path": line.split("\t")[-1]})
    return counts, changes


def trees_unchanged(source: str, remote: str) -> bool:
    return source == remote


def refresh_unchanged(cache: Path, auth_url: str, source_tree_sha: str,
                      env: dict[str, str]) -> tuple[str, str]:
    git(cache, "fetch", "--quiet", auth_url, "main", env=env)
    remote_commit = git(cache, "rev-parse", "FETCH_HEAD^{commit}", env=env)
    remote_tree = git(cache, "rev-parse", "FETCH_HEAD^{tree}", env=env)
    if remote_tree != source_tree_sha:
        raise SyncError("remote main changed after preflight; refusing unchanged receipt")
    return remote_commit, remote_tree


def authenticated_url(anchor: str, name: str) -> str:
    parsed = urlparse(anchor)
    if parsed.scheme != "https" or not parsed.hostname:
        raise SyncError("hkust-org credential anchor must be an HTTPS GitHub URL")
    if parsed.hostname.lower() != "github.com":
        raise SyncError("hkust-org credential anchor must target github.com")
    auth = ""
    if parsed.username is not None:
        auth = parsed.username
        if parsed.password is not None:
            auth += ":" + parsed.password
        auth += "@"
    return f"{parsed.scheme}://{auth}github.com/{ORG}/{name}.git"


def assert_root_state(expected_sha: str | None, *, allow_dirty: bool) -> tuple[str, bool]:
    branch = git(ROOT, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch != "main":
        raise SyncError("root repository must be checked out on main")
    head = git(ROOT, "rev-parse", "HEAD^{commit}")
    main = git(ROOT, "rev-parse", "main^{commit}")
    if head != main:
        raise SyncError("root HEAD must equal local main")
    if expected_sha is not None and head != expected_sha:
        raise SyncError("root HEAD changed during preflight; refusing to publish")
    dirty = bool(git(ROOT, "status", "--porcelain", "--untracked-files=normal"))
    if dirty and not allow_dirty:
        raise SyncError("root working tree is dirty; commit or remove all changes before syncing")
    return head, dirty


def probe_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=".push_both_write_probe_", dir=log_dir):
            pass
    except OSError as exc:
        raise SyncError(f"result log directory is not writable: {sanitize(str(exc))}") from exc


def write_logs(result: dict, result_path: Path, private_log: Path) -> None:
    try:
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        private_log.write_text(datetime.now(timezone.utc).isoformat() + " " +
                               sanitize(json.dumps(result)) + "\n")
    except OSError as exc:
        raise SyncError(f"cannot write result log: {sanitize(str(exc))}") from exc


def select_only(value: str | None) -> list[str]:
    if not value:
        return list(REPOS)
    exact = ALIASES.get(value, value)
    if exact not in REPOS:
        raise SyncError(f"--only must name one mirror ({', '.join(REPOS)}) or alias fe")
    return [exact]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    result = {"dryRun": ns.dry_run, "rootSHA": None, "mirrors": [], "success": False}
    log_dir = ROOT / "logs"
    private_log = log_dir / "push_both_private.log"
    result_path = log_dir / "push_both_result.json"
    try:
        probe_log_dir(log_dir)
        repos = select_only(ns.only)
        root_sha, dirty = assert_root_state(None, allow_dirty=ns.dry_run)
        if dirty:
            result["warning"] = "root working tree is dirty; dry-run uses committed main only"
            print("WARNING: root is dirty; preview uses committed main only", file=sys.stderr)
        result["rootSHA"] = root_sha
        if not ns.only:
            verify_remote(ROOT, "origin", ROOT_OWNER, "quant_projects")
        anchor = git(ROOT, "remote", "get-url", "hkust-org")
        anchor_host, anchor_owner, _ = remote_identity(anchor)
        if (anchor_host, anchor_owner) != ("github.com", ORG):
            raise SyncError(f"unexpected hkust-org target; expected github.com/{ORG}/...")

        env = object_env(ROOT)
        plans = []
        for name in repos:
            cache = CACHE / name
            if not (cache / ".git").is_dir():
                raise SyncError(f"mirror cache missing or invalid: {cache}")
            verify_remote(cache, "origin", ORG, name)
            auth_url = authenticated_url(anchor, name)
            # Fetch is mandatory and fail-closed. It does not change cache worktree or branches.
            git(cache, "fetch", "--quiet", auth_url, "main", env=env)
            parent = git(cache, "rev-parse", "FETCH_HEAD^{commit}", env=env)
            parent_tree = git(cache, "rev-parse", "FETCH_HEAD^{tree}", env=env)
            tree = source_tree(ROOT, root_sha, name)
            counts, changes = preview(cache, parent, tree, env)
            unchanged = trees_unchanged(tree, parent_tree)
            entry = {"name": name, "sourceTree": tree, "remoteParent": parent,
                     "remoteURL": "github.com/%s/%s" % (ORG, name),
                     "preview": counts, "changes": changes,
                     "mirrorCommit": parent if unchanged else None,
                     "verifiedRemoteTree": parent_tree if unchanged else None,
                     "status": "unchanged" if unchanged else "planned"}
            result["mirrors"].append(entry)
            plans.append((cache, name, tree, parent, entry, auth_url))
            print(f"{name}: +{counts['add']} ~{counts['modify']} -{counts['delete']}")
            for change in changes:
                print(f"  {change['status']}\t{change['path']}")

        if ns.dry_run:
            result["success"] = True
            return 0

        # Fetching every mirror can take time. Refuse if another task changed root meanwhile.
        assert_root_state(root_sha, allow_dirty=False)
        # Root is published first.
        if not ns.only:
            verify_push_remote(ROOT, "origin", ROOT_OWNER, "quant_projects")
            git(ROOT, "push", "origin", f"{root_sha}:refs/heads/main")
        # Only after root succeeds do mirrors publish immutable commit IDs.
        for cache, name, tree, parent, entry, auth_url in plans:
            if entry["status"] == "unchanged":
                remote_commit, remote_tree = refresh_unchanged(cache, auth_url, tree, env)
                entry.update(mirrorCommit=remote_commit, verifiedRemoteTree=remote_tree)
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            commit = run(["git", "-C", str(cache), "-c", "user.name=Sun Haiwei", "-c",
                          "user.email=sunhaiwei@users.noreply.github.com", "commit-tree", tree,
                          "-p", parent, "-m", f"sync {name} from quant_projects {root_sha} at {stamp}"], env=env)
            git(cache, "push", auth_url, f"{commit}:refs/heads/main", env=env)
            git(cache, "fetch", "--quiet", auth_url, "main", env=env)
            remote_commit = git(cache, "rev-parse", "FETCH_HEAD^{commit}", env=env)
            remote_tree = git(cache, "rev-parse", "FETCH_HEAD^{tree}", env=env)
            if remote_commit != commit or remote_tree != tree:
                raise SyncError(f"post-push verification failed for {name}")
            entry.update(mirrorCommit=commit, verifiedRemoteTree=remote_tree, status="pushed")
        result["success"] = True
        return 0
    except (SyncError, OSError) as exc:
        result["error"] = sanitize(str(exc))
        print(f"FATAL: {result['error']}", file=sys.stderr)
        return 1
    finally:
        try:
            write_logs(result, result_path, private_log)
        except SyncError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
