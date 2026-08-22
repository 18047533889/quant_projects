"""
data_access/tests/test_check_allowlist —— PR6 静态检查脚本的端到端测试

覆盖：
- 无违规时退出码 0
- 违规时退出码 1 并在 stderr 列出违规
- allowlist 里登记的路径被豁免
- AST 精度：docstring / 注释里的 `pd.read_parquet` 字面量不触发
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = REPO_ROOT / "scripts" / "check_data_access_allowlist.py"


def _run_checker(*files: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """注意：脚本内部用 `REPO_ROOT = __file__ 的上两层`，这是 scripts/check_data_access_allowlist.py
    在真实仓库里的位置；临时目录场景下要把脚本和一份空 allowlist 一起 copy 过去。
    """
    return subprocess.run(
        [sys.executable, str(CHECKER), *(str(f) for f in files)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _setup_fake_repo(tmp_path: Path, allowlist_content: str) -> tuple[Path, Path]:
    """在 tmp_path 下搭一个 mini 仓库：allowlist + scripts/check_data_access_allowlist.py 拷贝过来。

    返回 (fake_repo_root, fake_checker_path)。
    """
    fake_root = tmp_path / "fake_repo"
    fake_root.mkdir()
    (fake_root / ".data_access_allowlist.yaml").write_text(
        allowlist_content, encoding="utf-8"
    )
    scripts_dir = fake_root / "scripts"
    scripts_dir.mkdir()
    fake_checker = scripts_dir / "check_data_access_allowlist.py"
    fake_checker.write_text(CHECKER.read_text(encoding="utf-8"), encoding="utf-8")
    return fake_root, fake_checker


def _run_fake_checker(
    fake_checker: Path,
    cwd: Path,
    *files: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(fake_checker), *(str(f) for f in files)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_clean_file_passes(tmp_path: Path):
    fake_root, fake_checker = _setup_fake_repo(tmp_path, "[]\n")
    good = fake_root / "good.py"
    good.write_text(
        textwrap.dedent(
            """
            from data_access import get_store

            store = get_store()
            df = store.read_frame("us_stocks_sip_day_aggs", columns=["ticker"])
            """
        ).strip(),
        encoding="utf-8",
    )
    result = _run_fake_checker(fake_checker, fake_root)
    assert result.returncode == 0, result.stderr


def test_violation_fails_with_stderr(tmp_path: Path):
    fake_root, fake_checker = _setup_fake_repo(tmp_path, "[]\n")
    bad = fake_root / "bad.py"
    bad.write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def leaked():
                df = pd.read_parquet("forbidden.parquet")
                df.to_parquet("also_forbidden.parquet")
                return df
            """
        ).strip(),
        encoding="utf-8",
    )
    result = _run_fake_checker(fake_checker, fake_root)
    assert result.returncode == 1
    assert "bad.py" in result.stderr
    assert "pd.read_parquet" in result.stderr
    assert ".to_parquet" in result.stderr


def test_allowlist_exempts_file(tmp_path: Path):
    allowlist = "- path: bad.py\n  reason: 'test fixture'\n"
    fake_root, fake_checker = _setup_fake_repo(tmp_path, allowlist)
    bad = fake_root / "bad.py"
    bad.write_text(
        textwrap.dedent(
            """
            import duckdb
            conn = duckdb.connect(":memory:")
            """
        ).strip(),
        encoding="utf-8",
    )
    result = _run_fake_checker(fake_checker, fake_root)
    assert result.returncode == 0, result.stderr


def test_glob_pattern_matches_subdir(tmp_path: Path):
    allowlist = "- path: 'tests/**'\n  reason: 'test fixtures'\n"
    fake_root, fake_checker = _setup_fake_repo(tmp_path, allowlist)
    tests_dir = fake_root / "tests"
    tests_dir.mkdir()
    bad = tests_dir / "test_stuff.py"
    bad.write_text(
        "import pandas as pd\npd.read_parquet('x.parquet')\n",
        encoding="utf-8",
    )
    result = _run_fake_checker(fake_checker, fake_root)
    assert result.returncode == 0, result.stderr


def test_docstring_and_comment_are_not_flagged(tmp_path: Path):
    fake_root, fake_checker = _setup_fake_repo(tmp_path, "[]\n")
    good = fake_root / "good.py"
    good.write_text(
        textwrap.dedent(
            '''
            """模块 docstring：提到 pd.read_parquet 和 duckdb.connect 只是解释规范。"""

            # 下面这行注释也不应触发: pd.read_parquet / pq.read_table / .to_parquet

            value = "pd.read_parquet 字符串里的内容也不应触发"


            def f() -> int:
                """函数 docstring: 也不该因为提到 .to_parquet 就炸。"""
                return 1
            '''
        ).strip(),
        encoding="utf-8",
    )
    result = _run_fake_checker(fake_checker, fake_root)
    assert result.returncode == 0, result.stderr


def test_single_file_mode_scans_only_that_file(tmp_path: Path):
    """pre-commit 会把 staged 文件列表传进来——只扫指定文件。"""
    fake_root, fake_checker = _setup_fake_repo(tmp_path, "[]\n")
    bad = fake_root / "bad.py"
    bad.write_text("import duckdb\n", encoding="utf-8")
    other_bad = fake_root / "other_bad.py"
    other_bad.write_text("import duckdb\n", encoding="utf-8")

    # 只传 bad.py，不应扫到 other_bad.py
    result = _run_fake_checker(fake_checker, fake_root, bad)
    assert result.returncode == 1
    assert "bad.py" in result.stderr
    assert "other_bad.py" not in result.stderr


@pytest.mark.skipif(not CHECKER.exists(), reason="checker script missing")
def test_real_repo_passes():
    """对真实仓库跑一次；PR6 完结状态应该 0 违规。"""
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"checker reported violations in real repo:\n{result.stderr}"
    )
