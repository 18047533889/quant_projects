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
import socket
import subprocess
import contextvars
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from data_access.core.exceptions import ValidationError


_NAMESPACE_ENV = "QUANT_RUN_NAMESPACE"
_OPERATOR_ENV = "QUANT_OPERATOR"
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_VALID_NS_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# #37 / #P0-32：session-scoped namespace 覆盖（DataAccessSession）。
# 用 ``contextvars.ContextVar`` 而不是 ``threading.local()`` —— 前者随 asyncio
# task 传播，同一线程内不同协程可各自绑定 namespace，不会串。
_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "data_access_session_namespace", default=None
)


def validate_namespace_chars(value: str, *, context: str = "namespace") -> str:
    """#P1-62/#P1-63 显式 namespace 只允许 ``[A-Za-z0-9._-]``。

    sanitizer 用 ``_`` 替换非法字符不是 injective（``a/b`` 和 ``a_b`` 都变
    ``a_b``），且 ``..`` / ``///`` 会被清洗成 anon 但仍被当「显式设置」——
    这两个都靠显式校验拒绝：坏 namespace 直接报错，绝不静默替换/汇聚。
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context} 必须是非空字符串")
    if not _VALID_NS_RE.match(value):
        raise ValidationError(
            f"{context}={value!r} 含非法字符。只允许 [A-Za-z0-9._-]。"
            "非法 namespace 会被错误地清洗/汇聚，请改用合法名字。"
        )
    # 纯点/纯下划线（"." / ".." / "___"）会被清洗成 anon 但仍被当「显式设置」——
    # 显式拒绝，防路径穿越 / 汇聚到 anon。
    if not value.strip("._-") or value in {".", ".."}:
        raise ValidationError(
            f"{context}={value!r} 非法：不能是纯 . / _ / -（会汇聚成 anon 或被当路径穿越）"
        )
    return value


def _sanitize(value: str, *, fallback: str = "anon") -> str:
    """清洗不安全字符（只用于**生成的**兜底段，如 git 分支/pid/hostname）。

    #P1-62 显式 namespace 不走这里——显式值必须先过 ``validate_namespace_chars``。
    """
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
    session_ns = _session.get()
    if session_ns:
        return _sanitize(session_ns)
    explicit = os.environ.get(_NAMESPACE_ENV, "").strip()
    if explicit:
        # #P1-63 显式坏 namespace（.. / /// / 空格）直接拒绝，不能清洗成 anon
        # 却仍被 is_namespace_explicit() 当显式。
        return validate_namespace_chars(explicit)

    branch = _git_branch() or "nobranch"
    pid = os.getpid()
    # #P1-61 兜底并入 hostname：PID 只在本机唯一，共享 NFS 下两个 host 的
    # pid=1234 会写进同一个 namespace，必须加 host 维度。
    host = _sanitize(socket.gethostname() or "host")
    return _sanitize(f"anon__{host}__{branch}__{pid}")


def set_session_namespace(namespace: str | None) -> None:
    """#37 设置当前 context 的 session namespace（``None`` 清除，回退环境变量）。

    用于 ``DataAccessSession(namespace=...)``：请求级绑定，路径 resolve 时再
    生效——不再依赖全局环境变量决定多租户路径。
    #P0-32：基于 ``contextvars.ContextVar``，随 asyncio task 传播。
    # #P1-62/#P1-63 显式 session namespace 非法字符 → 拒绝，不静默替换。
    """
    if namespace:
        validate_namespace_chars(namespace)
        _session.set(namespace)
    else:
        _session.set(None)


def session_namespace() -> str | None:
    return _session.get()


@contextmanager
def namespace_scope(namespace: str | None):
    """#37 上下文管理器：进入时绑定 session namespace，退出时恢复。"""
    if namespace:
        validate_namespace_chars(namespace)
    token = _session.set(namespace)
    try:
        yield
    finally:
        _session.reset(token)


class DataAccessSession:
    """#37 请求/session 级命名空间上下文。

    用法::

        with DataAccessSession(namespace="run_123"):
            store.write_arrow("factor_lake_staging", tbl, factor_id="x")

    内部所有 ``resolve_namespace()``（写路径目录绑定）在该作用域内使用
    ``run_123``，退出后恢复进入前的 namespace（嵌套场景不丢外层）。
    避免「loader 加载时就把 ${RUN_NAMESPACE} 替换死」导致长期 worker
    多租户串路径。
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace

    def __enter__(self) -> "DataAccessSession":
        if self.namespace:
            validate_namespace_chars(self.namespace)
        self._token = _session.set(self.namespace if self.namespace else None)
        return self

    def __exit__(self, *exc: Any) -> None:
        # #P0-32：restore 进入前的 namespace，而不是无条件清成 None。
        token = getattr(self, "_token", None)
        if token is not None:
            _session.reset(token)


_OPERATOR_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._@+-]+")

# #P1-64 request-scoped operator：async 多用户服务里 QUANT_OPERATOR 是进程级，
# 同一进程多 job/user 时需要 ContextVar 作用域（operator_scope）。
_operator_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "data_access_operator", default=None
)


@contextmanager
def operator_scope(operator: str | None):
    """#P1-64 请求级 operator 覆盖（ContextVar，随 asyncio task 传播）。"""
    if operator is not None and not isinstance(operator, str):
        raise ValidationError(f"operator 必须是字符串，收到 {type(operator).__name__}")
    token = _operator_ctx.set(operator)
    try:
        yield
    finally:
        _operator_ctx.reset(token)


def resolve_operator() -> str | None:
    """返回审计用的 operator 名字；没配就返回 None（由调用方决定要不要告警）。

    WHY：publish / 写 published_root 的动作必须记录「是谁干的」才有意义；
         共用 uid 下 os.getlogin() 永远是 yluel，没参考价值。

    #P1-64 优先级：request-scoped ``operator_scope`` context > 环境变量。
    operator 只进审计日志（不拼进路径），所以允许 `@` 和 `+` 这类邮箱/身份标识常用字符；
    namespace 则严格（走 _sanitize），因为会被拼进文件路径。
    """
    scoped = _operator_ctx.get()
    if scoped:
        return scoped
    explicit = os.environ.get(_OPERATOR_ENV, "").strip()
    if not explicit:
        return None
    cleaned = _OPERATOR_SANITIZE_RE.sub("_", explicit).strip("._-")
    return cleaned or None


def is_namespace_explicit() -> bool:
    """判断 namespace 是否来自显式来源（环境变量或 session context）。

    用于告警：如果写入了 namespaced 数据集但 namespace 是兜底生成的，
    说明用户可能忘记 export 了，日志里打个 warning。
    #P0-32：同时认可 session namespace（``DataAccessSession``/``namespace_scope``）。
    """
    if _session.get():
        return True
    return bool(os.environ.get(_NAMESPACE_ENV, "").strip())
