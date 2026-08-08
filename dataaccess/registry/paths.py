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
    return os.path.expanduser(expanded)


def canonicalize(path: str | Path) -> Path:
    """把路径转成「绝对、消解 .. 和软链接」的标准形式。

    WHY：白名单比较必须在 canonical 形式下做，否则 '/a/../b' 这类可以绕过。
         `.resolve()` 在路径不存在时仍返回绝对形式（strict=False 行为）。
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


def _production_mode() -> bool:
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if fe == "production":
        return True
    return os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}


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
    """按一组允许的根目录做前缀校验；registry 加载时构造一次，store 内部用。"""

    def __init__(self, allowed_roots: Iterable[Path]) -> None:
        # canonicalize 后去重，避免重复比较；保留原顺序仅用于错误信息排序
        seen: set[Path] = set()
        roots: list[Path] = []
        for raw in allowed_roots:
            resolved = canonicalize(raw)
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
        self._roots: tuple[Path, ...] = tuple(roots)

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return self._roots

    def resolve_and_authorize(self, path: str | Path) -> Path:
        """清洗路径并检查是否落在任一允许根下；不通过则抛 ValidationError。

        返回 canonical 路径（可直接交给 DuckDB/open()）。
        """
        resolved = canonicalize(path)
        for root in self._roots:
            if path_is_under(resolved, root):
                return resolved
        sensitive = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        if sensitive:
            root_hint = f"{len(self._roots)} registered roots"
            resolved_hint = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
            raise ValidationError(
                f"路径越界（canonical path fingerprint={resolved_hint}）；{root_hint}。"
                "如需新增，请在 data_access/config/datasets.yaml 注册数据集。"
            )
        roots_hint = "\n  ".join(str(r) for r in self._roots)
        raise ValidationError(
            f"路径越界（不在任何已注册数据集根下）：{resolved}\n"
            f"当前允许的数据根：\n  {roots_hint}\n"
            f"如需新增，请在 data_access/config/datasets.yaml 注册数据集。"
        )
