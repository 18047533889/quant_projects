import importlib.util
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts" / "push_both.py"
spec = importlib.util.spec_from_file_location("push_both", MODULE)
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def g(repo, *args, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


class PushBothTests(unittest.TestCase):
    def test_remote_validation_and_exact_only(self):
        self.assertEqual(pb.remote_identity("https://secret@github.com/HKUST-QUANT-SOCIETY/modeling.git"),
                         ("github.com", "HKUST-QUANT-SOCIETY", "modeling"))
        self.assertEqual(pb.select_only("fe"), ["factor_engine"])
        with self.assertRaises(pb.SyncError):
            pb.select_only("factor")
        with self.assertRaisesRegex(pb.SyncError, "HTTPS"):
            pb.authenticated_url("http://token@github.com/HKUST-QUANT-SOCIETY/modeling.git", "modeling")

    def test_root_must_be_on_main(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.check_call(["git", "init", "-q", "-b", "dev", str(repo)])
            g(repo, "config", "user.name", "Test")
            g(repo, "config", "user.email", "test@example.invalid")
            (repo / "x").write_text("x")
            g(repo, "add", ".")
            g(repo, "commit", "-qm", "initial")
            old_root = pb.ROOT
            pb.ROOT = repo
            try:
                with self.assertRaisesRegex(pb.SyncError, "checked out on main"):
                    pb.assert_root_state(None, allow_dirty=False)
            finally:
                pb.ROOT = old_root
        with self.assertRaisesRegex(pb.SyncError, "HTTPS"):
            pb.authenticated_url("http://token@github.com/HKUST-QUANT-SOCIETY/a.git", "modeling")

    def test_root_pushurl_is_validated_separately(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.check_call(["git", "init", "-q", str(repo)])
            g(repo, "remote", "add", "origin", "https://github.com/18047533889/quant_projects.git")
            g(repo, "remote", "set-url", "--push", "origin", "https://github.com/wrong/quant_projects.git")
            pb.verify_remote(repo, "origin", "18047533889", "quant_projects")
            with self.assertRaisesRegex(pb.SyncError, "push target"):
                pb.verify_push_remote(repo, "origin", "18047533889", "quant_projects")

    def test_root_state_requires_main_and_head_equal_main(self):
        global_root = pb.ROOT
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
            g(repo, "config", "user.name", "Test")
            g(repo, "config", "user.email", "test@example.invalid")
            (repo / "a").write_text("a")
            g(repo, "add", ".")
            g(repo, "commit", "-qm", "one")
            pb.ROOT = repo
            sha, dirty = pb.assert_root_state(None, allow_dirty=False)
            self.assertEqual(sha, g(repo, "rev-parse", "HEAD"))
            self.assertFalse(dirty)
            g(repo, "checkout", "-qb", "other")
            with self.assertRaisesRegex(pb.SyncError, "checked out on main"):
                pb.assert_root_state(None, allow_dirty=False)
        pb.ROOT = global_root

    def test_log_directory_probe_rejects_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "not-a-directory"
            path.write_text("x")
            with self.assertRaises((pb.SyncError, FileExistsError)):
                pb.probe_log_dir(path)

    def test_result_log_write_failure_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            blocker = Path(td) / "blocker"
            blocker.write_text("not a directory")
            with self.assertRaisesRegex(pb.SyncError, "cannot write result log"):
                pb.write_logs({}, blocker / "result.json", blocker / "private.log")

    def test_equal_trees_are_unchanged(self):
        self.assertTrue(pb.trees_unchanged("abc", "abc"))
        self.assertFalse(pb.trees_unchanged("abc", "def"))

    def test_unchanged_receipt_refetches_and_detects_remote_move(self):
        with mock.patch.object(pb, "git", side_effect=["", "a" * 40, "tree"]):
            commit, tree = pb.refresh_unchanged(Path("/cache"), "https://example", "tree", {})
        self.assertEqual(commit, "a" * 40)
        self.assertEqual(tree, "tree")
        with mock.patch.object(pb, "git", side_effect=["", "b" * 40, "different"]):
            with self.assertRaisesRegex(pb.SyncError, "changed after preflight"):
                pb.refresh_unchanged(Path("/cache"), "https://example", "tree", {})

    def test_subtree_preview_and_commit_do_not_change_cache_worktree(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, cache = base / "root", base / "cache"
            for repo in (root, cache):
                repo.mkdir()
                subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
                g(repo, "config", "user.name", "Test")
                g(repo, "config", "user.email", "test@example.invalid")
            (root / "lib").mkdir()
            (root / "lib" / "new.txt").write_text("new\n")
            g(root, "add", ".")
            g(root, "commit", "-qm", "root")
            (cache / "old.txt").write_text("old\n")
            g(cache, "add", ".")
            g(cache, "commit", "-qm", "old")
            parent = g(cache, "rev-parse", "HEAD")
            tree = pb.source_tree(root, "main", "lib")
            env = pb.object_env(root)
            counts, changes = pb.preview(cache, parent, tree, env)
            self.assertEqual(counts, {"add": 1, "modify": 0, "delete": 1})
            self.assertEqual({c["path"] for c in changes}, {"new.txt", "old.txt"})
            commit = pb.run(["git", "-C", str(cache), "commit-tree", tree, "-p", parent, "-m", "sync"], env=env)
            self.assertEqual(g(cache, "rev-parse", f"{commit}^{{tree}}", env=env), tree)
            self.assertEqual((cache / "old.txt").read_text(), "old\n")
            self.assertFalse((cache / "new.txt").exists())

    def test_gitlink_is_rejected_as_tree(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
            g(repo, "config", "user.name", "Test")
            g(repo, "config", "user.email", "test@example.invalid")
            # A 160000 entry resolves to a commit, and the sync's explicit type guard rejects it.
            empty_tree = subprocess.check_output(["git", "hash-object", "-t", "tree", "--stdin"], input="", text=True).strip()
            commit = g(repo, "commit-tree", empty_tree, "-m", "nested")
            line = f"160000 commit {commit}\tnested\n"
            tree = subprocess.check_output(["git", "mktree"], input=line, text=True, cwd=repo).strip()
            top = g(repo, "commit-tree", tree, "-m", "top")
            with self.assertRaisesRegex(pb.SyncError, "not a directory tree"):
                pb.source_tree(repo, top, "nested")


if __name__ == "__main__":
    unittest.main()
