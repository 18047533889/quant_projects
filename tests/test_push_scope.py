"""Real tiny Git fixtures: never copy project repositories or data."""
import importlib.util
import subprocess
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("scope_publisher", Path(__file__).parents[1] / "scripts/push_both.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    for name in ("factor_engine", "factor_optimizer"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "module.py").write_text(name)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "fixture")
    return tmp_path


def candidate(repo, path):
    tree = git(repo, "rev-parse", "HEAD:" + path)
    return git(repo, "commit-tree", tree, "-m", "fixture mirror")


def test_whole_project_cannot_be_pushed_as_factor_engine(repo):
    with pytest.raises(pb.SyncError, match="tree"):
        pb.validate_publication(repo, repo, "HEAD", "https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git", "HEAD")


def test_exact_subtree_is_accepted_but_sibling_is_rejected(repo):
    url = "git@github.com:HKUST-QUANT-SOCIETY/factor_engine.git"
    pb.validate_publication(repo, repo, "HEAD", url, candidate(repo, "factor_engine"))
    with pytest.raises(pb.SyncError, match="tree"):
        pb.validate_publication(repo, repo, "HEAD", url, candidate(repo, "factor_optimizer"))


def test_root_only_goes_to_personal_project(repo):
    pb.validate_publication(repo, repo, "HEAD", "https://github.com/18047533889/quant_projects.git", "HEAD")
    with pytest.raises(pb.SyncError, match="target"):
        pb.validate_publication(repo, repo, "HEAD", "https://github.com/other/quant_projects.git", "HEAD")


def test_multiple_push_destinations_are_rejected(repo):
    git(repo, "remote", "add", "origin", "https://github.com/18047533889/quant_projects.git")
    git(repo, "config", "--add", "remote.origin.pushurl", "https://github.com/18047533889/quant_projects.git")
    git(repo, "config", "--add", "remote.origin.pushurl", "https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git")
    with pytest.raises(pb.SyncError, match="push target"):
        pb.verify_push_remote(repo, "origin", "18047533889", "quant_projects")


def test_nonmain_and_deletion_are_rejected(repo):
    url = "https://github.com/18047533889/quant_projects.git"
    with pytest.raises(pb.SyncError, match="main"):
        pb.validate_publication(repo, repo, "HEAD", url, "HEAD", "refs/heads/mistake")
    with pytest.raises(pb.SyncError, match="delet"):
        pb.validate_publication(repo, repo, "HEAD", url, "0" * 40)


def test_nested_sibling_library_is_rejected(repo):
    (repo / "factor_engine" / "factor_optimizer").mkdir()
    (repo / "factor_engine" / "factor_optimizer" / "oops.py").write_text("copy")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "contaminated")
    with pytest.raises(pb.SyncError, match="nested"):
        pb.validate_publication(repo, repo, "HEAD", "https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git", candidate(repo, "factor_engine"))


def test_hook_blocks_wrong_tree_and_accepts_exact_subtree(repo):
    hook = Path(__file__).parents[1] / "scripts/check_push_scope.py"
    url = "https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git"
    env = dict(os.environ, PUSH_BOTH_ROOT=str(repo))
    for sha, allowed in [(git(repo, "rev-parse", "HEAD"), False),
                         (candidate(repo, "factor_engine"), True)]:
        result = subprocess.run(
            ["python3", str(hook), "arbitrary-remote-name", url], cwd=repo, env=env,
            input=f"HEAD {sha} refs/heads/main {'0' * 40}\n", text=True,
            capture_output=True)
        assert (result.returncode == 0) == allowed, result.stderr
        if not allowed:
            assert "tree mismatch" in result.stderr
