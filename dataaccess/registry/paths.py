"""
data_access.paths —— 路径白名单与解析

职责：
    1. 按 registry 注册的数据集 root，自动生成「允许的路径前缀集合」
    2. resolve_and_authorize(path) 做路径清洗 + 越界拒绝
    3. 对 namespaced / staging 数据集，把 RUN_NAMESPACE 自动插进路径

非职责：
    不负责路径是否真实存在（由 store.py / writer 处理）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Iterable

from data_access.core.exceptions import ValidationError


# 匹配 ${VAR} 或 ${VAR:-默认值}；不吞变量名里的 } 所以 default 值里可以含 /、空格等
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(template: str) -> str:
    """展开 ${VAR}、${VAR:-default}、$VAR、~。

    相比 os.path.expandvars，额外支持 ${VAR:-default} 这种 bash 风格默认值，
    让 datasets.yaml 能写 ${MASSIVE_PARQUET_ROOT:-/home/yluel/share/...} 这种
    "没设环境变量时用兜底路径"的配置。

    WHY：团队里每个人的 shell 设置不一样；不能强制大家都 export 所有 env。
         YAML 里写默认值，未设 env 的人也能直接跑。
    """

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)  # 可能是 None（没写默认）或 "" 或任意字符串
        value = os.environ.get(var_name)
        if value:
            return value
        if default is not None:
            return default
        # 既没 env 也没默认：保留原样，让后续路径解析报带上下文的错
        return match.group(0)

    # Never expand variables introduced by a default value; only the original
    # template's ${...} tokens are configuration inputs.
    expanded = _ENV_PATTERN.sub(replace, template)
    # #P1-69 docstring 承诺支持 `$VAR`：${...} 处理完再让 os.path.expandvars
    # 补 `$VAR`（裸形式），两者行为对齐文档。
    return os.path.expandvars(os.path.expanduser(expanded))


def canonicalize(path: str | Path) -> Path:
    """把路径转成「绝对、消解 .. 和软链接」的标准形式。

    WHY：白名单比较必须在 canonical 形式下做，否则 '/a/../b' 这类可以绕过。
         `.resolve()` 在路径不存在时仍返回绝对形式（strict=False 行为）。

    #R32-P0-121 TOCTOU 防护：使用 `resolve(strict=False)` 原子化解析 symlink——
    内核保证 resolve 过程中路径不会因 symlink 替换产生 TOCTOU 窗口。旧代码若用
    `is_symlink() + readlink()` 两步则存在窗口（check 后 symlink 可被替换）。
    """
    return Path(path).expanduser().resolve(strict=False)


def path_is_under(child: Path, parent: Path) -> bool:
    """检查 child 是否严格位于 parent 目录下（或就是 parent 本身）。

    两边都必须先 canonicalize；本函数不做清洗，由调用方保证。
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def canonicalize_strict(path: str | Path) -> Path:
    """#R32-P0-124 存在性强制的 canonical 解析：``Path.resolve(strict=True)``。

    与 ``canonicalize`` 的区别：路径（含所有父级组件）必须真实存在，否则抛
    ``ValidationError``。

    WHY fail-closed：``strict=False`` 对不存在的尾部组件只做**纯字符串**拼接，
    不经内核 symlink 解析——「先鉴权一个不存在的路径、随后在该位置创建 symlink
    指向沙箱外」这条 TOCTOU 路径因此绕过白名单。需要真实读/写既有文件时（DuckDB
    scan、open()）必须走 strict 解析，让内核在**同一次系统调用链**里完成
    symlink 解析。
    """
    p = Path(path).expanduser()
    try:
        return p.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        # RuntimeError: symlink 环路（Python < 3.13 的 resolve 会抛）
        raise ValidationError(
            f"路径无法 canonical 解析（不存在或 symlink 环路）：{p}（{type(exc).__name__}）"
        ) from exc


def assert_no_symlink_escape(path: str | Path, root: str | Path) -> Path:
    """#R32-P0-121 原子化 symlink 校验：确认 ``path`` 解析后仍在 ``root`` 内。

    TOCTOU 关键点：**先解析再比较，且比较的是解析结果**。两步式
    ``if not p.is_symlink(): use(p)`` 存在窗口——check 之后 ``p`` 可被替换成
    symlink，随后的 open 就跟到沙箱外。这里的做法：

    1. ``os.path.realpath()`` 一次性完成整链 symlink 解析（内核语义，无用户态窗口）；
    2. 白名单比较在 realpath 结果上做；
    3. 若尾部组件本身是 symlink，用 ``os.readlink()`` 读出目标一并校验，
       让错误信息能指出逃逸目标（诊断用，不参与授权判定）。

    返回 realpath 化的 ``Path``；逃逸则抛 ``ValidationError``。
    """
    raw = Path(path).expanduser()
    root_real = Path(os.path.realpath(str(Path(root).expanduser())))
    # realpath 不要求路径存在，但会解析所有已存在的 symlink 组件（原子、内核语义）
    real = Path(os.path.realpath(str(raw)))
    if path_is_under(real, root_real):
        return real
    # 逃逸：尽量给出 symlink 目标，便于定位配置错误 / 攻击。
    link_hint = ""
    try:
        # follow_symlinks=False：只看尾部组件自身，绝不跟随（诊断也不能被牵走）
        if os.path.islink(str(raw)):
            link_hint = f"（symlink → {os.readlink(str(raw))}）"
    except OSError:
        pass
    raise ValidationError(
        f"symlink 逃逸：{raw}{link_hint} 解析为 {real}，不在允许根 {root_real} 下。"
        "symlink 不能把沙箱外的路径引入已注册数据集根。"
    )


def _production_mode() -> bool:
    # #P1-final closure 17：全项目唯一严格模式判定。旧代码只认 QUANT_PRODUCTION_MODE
    # + FACTOR_ENGINE_RUN_MODE，漏了 DATA_ACCESS_STRICT_READ=1（strict 读模式同样
    # 应该收紧 PathAuthorizer / extra roots 的沙箱）。
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        # 导入期循环依赖兜底：保守按严格处理（不能因 import 顺序放宽松）
        return True


def extra_allowed_roots_from_env() -> list[Path]:
    """``DATA_ACCESS_EXTRA_ALLOWED_ROOTS``：逗号分隔的额外白名单根。

    其他服务器自选读/写目录时，把自定义根加到这里，否则 PathAuthorizer 会拒越界。
    例：``export DATA_ACCESS_EXTRA_ALLOWED_ROOTS=/data/my_ws,/data/my_cache``

    #P0-48 production 拒绝 ``/``、``$HOME``、过宽祖先——普通环境变量不能实质取消
    本地路径沙箱。额外根应由部署配置提供，不直接信任任意 env。
    """
    raw = os.environ.get("DATA_ACCESS_EXTRA_ALLOWED_ROOTS", "")
    roots: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        resolved = canonicalize(part)
        if _production_mode():
            if resolved == Path("/"):
                raise ValidationError(
                    "DATA_ACCESS_EXTRA_ALLOWED_ROOTS=/ 在 production 被拒绝："
                    "会实质取消本地路径沙箱。请改为具体数据集目录。"
                )
            home = canonicalize(Path.home())
            if resolved == home:
                raise ValidationError(
                    f"DATA_ACCESS_EXTRA_ALLOWED_ROOTS={resolved} 在 production 被拒绝"
                    "（$HOME 过宽）。请改为具体数据集目录。"
                )
        roots.append(resolved)
    return roots


def resolve_namespace_path(value: str) -> str:
    """把模板里的 ``${RUN_NAMESPACE}`` 替换为**当前 context** 的 namespace。

    #P1-final closure 12：registry load 时**不再**把 ``${RUN_NAMESPACE}`` 烘焙进
    ``root`` / ``root_template`` / ``static_root`` / ``authorized_root``——长期 worker
    里换 ``DataAccessSession`` / ``namespace_scope`` 后，真正 read/write 时才在此处
    解析当前 ``resolve_namespace()``，保证「请求级 namespace」真正生效。
    """
    if not isinstance(value, str) or "${RUN_NAMESPACE}" not in value:
        return value
    from data_access.core.namespace import resolve_namespace

    return value.replace("${RUN_NAMESPACE}", resolve_namespace())


def dataset_env_root(dataset_name: str, kind: str) -> str | None:
    """按数据集名读环境变量覆盖根。

    kind=read → ``DATA_ACCESS_READ_ROOT_<NAME>``
    kind=write → ``DATA_ACCESS_WRITE_ROOT_<NAME>``
    名称里 ``-`` 会变成 ``_``，并转大写。
    """
    key = f"DATA_ACCESS_{kind.upper()}_ROOT_{dataset_name.upper().replace('-', '_')}"
    val = os.environ.get(key)
    return val if val else None


class PathAuthorizer:
    """按一组允许的根目录做前缀校验；registry 加载时构造一次，store 内部用。

    #7 生命周期错配修复：PathAuthorizer 现在保存 **unresolved root template**
    （``${RUN_NAMESPACE}`` 占位符原样保留），``resolve_and_authorize`` 时按
    **当前 session 的 namespace** 解析。旧实现 store 构造时就把 namespace 烘焙进
    ``_roots``，长期 worker 换 ``DataAccessSession`` 后数据集路径解析成新
    namespace，authorizer 却仍只认识旧的——轻则合法路径被拒，重则（白名单配宽时）
    削弱 namespace 隔离。解析结果按 namespace 缓存（``resolve_namespace()`` 是
    ContextVar，不同请求不碰撞）。
    """

    def __init__(self, allowed_roots: Iterable[str | Path]) -> None:
        # 保存 raw template（string）。canonicalize 延迟到 authorize 时按当前
        # namespace 做——含占位符的模板此刻无法安全 canonicalize。
        self._templates: tuple[str, ...] = tuple(str(r) for r in allowed_roots)
        self._cache: dict[str, tuple[Path, ...]] = {}
        self._cache_lock = threading.Lock()

    def _effective_roots(self) -> tuple[Path, ...]:
        """当前 context namespace 下的有效白名单根（带 per-namespace 缓存）。"""
        from data_access.core.namespace import resolve_namespace

        ns = resolve_namespace()
        cached = self._cache.get(ns)
        if cached is not None:
            return cached
        seen: set[Path] = set()
        roots: list[Path] = []
        for tpl in self._templates:
            resolved = canonicalize(resolve_namespace_path(tpl))
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
        out = tuple(roots)
        with self._cache_lock:
            self._cache[ns] = out
        return out

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        """当前 namespace 下已解析的根（错误信息 / 诊断用）。"""
        return self._effective_roots()

    def resolve_and_authorize(self, path: str | Path) -> Path:
        """清洗路径并检查是否落在任一允许根下；不通过则抛 ValidationError。

        返回 canonical 路径（可直接交给 DuckDB/open()）。

        #R32-P0-121 Symlink TOCTOU 防护：canonicalize 内部用 Path.resolve() 原子化
        解析 symlink（内核保证），此处拿到的 resolved 已是 symlink-resolved 最终路径。
        白名单检查在 canonical 路径上做，攻击者无法通过 symlink 替换绕过。
        """
        resolved = canonicalize(path)
        for root in self._effective_roots():
            if path_is_under(resolved, root):
                return resolved
        roots = self._effective_roots()
        sensitive = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        if sensitive:
            root_hint = f"{len(roots)} registered roots"
            resolved_hint = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
            raise ValidationError(
                f"路径越界（canonical path fingerprint={resolved_hint}）；{root_hint}。"
                "如需新增，请在 data_access/config/datasets.yaml 注册数据集。"
            )
        roots_hint = "\n  ".join(str(r) for r in roots)
        raise ValidationError(
            f"路径越界（不在任何已注册数据集根下）：{resolved}\n"
            f"当前允许的数据根：\n  {roots_hint}\n"
            f"如需新增，请在 data_access/config/datasets.yaml 注册数据集。"
        )
