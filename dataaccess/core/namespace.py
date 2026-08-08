"""
data_access.namespace —— 多人共用账号下的运行命名空间与审计身份

背景：
    团队共用同一个 Linux 登录账号（yluel@mafm2），$HOME 共享；因此
    OS uid 不能区分人，~/.<alias> 之类的家目录文件也不可靠（多人互相覆盖）。
    本模块通过环境变量把「运行上下文」显式化：

        QUANT_RUN_NAMESPACE  —— 必填推荐，决定个人实验数据的写入路径
        QUANT_OPERATOR       —— 审计用的人类可读名字，publish 时强烈建议填

职责：
    1. resolve_namespace() 给个人实验产物算出一个隔离用目录名
    2. resolve_operator()  给审计日志算出「这次动作是谁做的」
    3. 两者都有兜底值，保证最差情况下也能跑（不会因为没设环境变量直接 hang）

非职责：
    不负责把 namespace 拼进实际路径（那是 paths.py 的事）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import Any


_NAMESPACE_ENV = "QUANT_RUN_NAMESPACE"
_OPERATOR_ENV = "QUANT_OPERATOR"
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# #37：session-scoped namespace 覆盖（DataAccessSession）。thread-local，比全局
# 环境变量更安全——长期 worker 里不同请求可以各自绑定 namespace。
_session = threading.local()


def _sanitize(value: str, *, fallback: str = "anon") -> str:
    """清洗不安全字符，防止 namespace 被拼到路径里造成目录穿越或奇怪文件名。"""
    cleaned = _SANITIZE_RE.sub("_", value).strip("._-")
    return cleaned or fallback


@lru_cache(maxsize=1)
def _git_branch() -> str | None:
    """尽力取当前分支名；不在 git 仓库或 git 不可用时返回 None。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def resolve_namespace() -> str:
    """返回当前运行的 namespace，用于隔离个人实验产物。

    解析优先级（高→低）：
        1. QUANT_RUN_NAMESPACE 环境变量（显式指定，最高优先）
        2. git 分支 + 进程 PID 兜底（保证多窗口同跑不撞）

    返回：
        已清洗的字符串，只含 [A-Za-z0-9._-]，可安全拼进路径。

    WHY：不用 ~/.quant_user_alias 这种家目录文件做身份识别 —— 团队共用
        同一个 $HOME，文件会互相覆盖，根本区分不出谁是谁。要区分人必须
        让每个成员在自己 shell / tmux 里 export 环境变量。

    # #37：session-scoped 覆盖（thread-local）优先于全局环境变量。
    """
    session_ns = getattr(_session, "namespace", None)
    if session_ns:
        return _sanitize(session_ns)
    explicit = os.environ.get(_NAMESPACE_ENV, "").strip()
    if explicit:
        return _sanitize(explicit)

    branch = _git_branch() or "nobranch"
    pid = os.getpid()
    return _sanitize(f"anon__{branch}__{pid}")


def set_session_namespace(namespace: str | None) -> None:
    """#37 设置当前线程的 session namespace（``None`` 清除，回退环境变量）。

    用于 ``DataAccessSession(namespace=...)``：请求级绑定，路径 resolve 时再
    生效——不再依赖全局环境变量决定多租户路径。
    """
    _session.namespace = _sanitize(namespace) if namespace else None


def session_namespace() -> str | None:
    return getattr(_session, "namespace", None)


@contextmanager
def namespace_scope(namespace: str | None):
    """#37 上下文管理器：进入时绑定 session namespace，退出时恢复。"""
    prev = getattr(_session, "namespace", None)
    set_session_namespace(namespace)
    try:
        yield
    finally:
        _session.namespace = prev


class DataAccessSession:
    """#37 请求/session 级命名空间上下文。

    用法::

        with DataAccessSession(namespace="run_123"):
            store.write_arrow("factor_lake_staging", tbl, factor_id="x")

    内部所有 ``resolve_namespace()``（写路径目录绑定）在该作用域内使用
    ``run_123``，退出后恢复环境变量兜底。避免「loader 加载时就把
    ${RUN_NAMESPACE} 替换死」导致长期 worker 多租户串路径。
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace

    def __enter__(self) -> "DataAccessSession":
        set_session_namespace(self.namespace)
        return self

    def __exit__(self, *exc: Any) -> None:
        set_session_namespace(None)


_OPERATOR_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._@+-]+")


def resolve_operator() -> str | None:
    """返回审计用的 operator 名字；没配就返回 None（由调用方决定要不要告警）。

    WHY：publish / 写 published_root 的动作必须记录「是谁干的」才有意义；
         共用 uid 下 os.getlogin() 永远是 yluel，没参考价值。

    operator 只进审计日志（不拼进路径），所以允许 `@` 和 `+` 这类邮箱/身份标识常用字符；
    namespace 则严格（走 _sanitize），因为会被拼进文件路径。
    """
    explicit = os.environ.get(_OPERATOR_ENV, "").strip()
    if not explicit:
        return None
    cleaned = _OPERATOR_SANITIZE_RE.sub("_", explicit).strip("._-")
    return cleaned or None


def is_namespace_explicit() -> bool:
    """判断 namespace 是否来自显式环境变量（而非兜底生成）。

    用于告警：如果写入了 namespaced 数据集但 namespace 是兜底生成的，
    说明用户可能忘记 export 了，日志里打个 warning。
    """
    return bool(os.environ.get(_NAMESPACE_ENV, "").strip())
