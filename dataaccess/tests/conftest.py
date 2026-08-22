"""
data_access 测试集配置 + 防护网。

## 为什么有这个文件

2026-04-19 当天，publish_from_staging 的一个 bug（`candidate_dir = Path()` 当 sentinel
但 Path() == PosixPath('.') ，exists()==True）让 finally 里的 shutil.rmtree(candidate_dir)
在 pytest CWD 上执行——而 pytest 默认 CWD 就是仓库根，仓库被整仓清空两次。

bug 本身已修（改成 None），但我们不能指望「再也不会有人写出同类 bug」。
这里加一个 pytest session 级别的守卫：把测试过程中的 CWD 强制迁出仓库树，
任何未来同类的 rmtree('.') 或 unlink('.') 都只会炸沙盒，不会炸仓库。

## 设计

- session-level autouse：session 开始前，如果 CWD 在项目根（或其子目录），
  chdir 到 pytest 管理的 tmp sandbox。session 结束后再 chdir 回去。
- per-test autouse：每个测试开始时，二次校验 CWD 不在仓库里；真有人 chdir 进来了就 fail，
  让问题第一时间暴露出来，而不是等 rmtree 炸了才发现。

## opt-out

如果某个测试真的需要 CWD 是仓库根（不应该有这种需求），自己 `monkeypatch.chdir(...)` 进去；
但请确认你不会在测试里走任何 rmtree/unlink 路径——否则又是一次事故。

维护人：quant 基础平台组    最后更新：2026-04-20
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# .../quantsociety_backend_project/data_access/tests/conftest.py → 三层上去就是仓库根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_under_project(path: Path) -> bool:
    """path 解析后是否在项目根目录树内（含项目根本身）。"""
    try:
        path.resolve().relative_to(_PROJECT_ROOT)
        return True
    except ValueError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _guard_cwd_against_rmtree_accidents(tmp_path_factory: pytest.TempPathFactory):
    """session 级：把 CWD 强制迁出仓库，session 结束再还回去。

    这是防 rmtree('.')/unlink('.') 类 bug 把仓库清空的最后一道保险。
    """
    original_cwd = Path.cwd()
    moved = False
    if _is_under_project(original_cwd):
        sandbox = tmp_path_factory.mktemp("data_access_cwd_sandbox")
        os.chdir(sandbox)
        moved = True

    try:
        yield
    finally:
        if moved:
            try:
                os.chdir(original_cwd)
            except (FileNotFoundError, OSError):
                # 万一 original_cwd 被删了（真炸了就别盖住事故本身的错），静默
                pass


@pytest.fixture(autouse=True)
def _assert_cwd_not_in_repo_during_test():
    """per-test 级：每个测试开始时校验 CWD 不在仓库里。

    session 守卫已经 chdir 出去了，正常情况下这条永远不触发。
    一旦有 fixture 或前一个测试里执行了 `os.chdir(repo_root)`，这里就 fail，
    比等到某个 rmtree 真炸了再排查快得多。
    """
    cwd = Path.cwd()
    if _is_under_project(cwd):
        pytest.fail(
            f"测试开始时 CWD={cwd} 在项目根 {_PROJECT_ROOT} 下。"
            f"历史上这会让 shutil.rmtree('.') 或 os.unlink('.') 误删仓库。"
            f"请改用 monkeypatch.chdir(tmp_path) 或其他 fixture 隔离。"
        )
    yield


@pytest.fixture(autouse=True)
def _reset_production_env_leaks():
    """清理跨模块测试遗留的生产环境变量。"""
    yield
    os.environ.pop("QUANT_PRODUCTION_MODE", None)
    if os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production":
        os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)
