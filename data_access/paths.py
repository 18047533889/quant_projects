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

import os
import re
from pathlib import Path
from typing import Iterable

from .exceptions import ValidationError


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

    expanded = _ENV_PATTERN.sub(replace, template)
    # 再处理 $VAR 形式（无大括号）和 ~
    expanded = os.path.expandvars(expanded)
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
        # 错误消息列出白名单，便于排查。生产环境若敏感可只打哈希。
        roots_hint = "\n  ".join(str(r) for r in self._roots)
        raise ValidationError(
            f"路径越界（不在任何已注册数据集根下）：{resolved}\n"
            f"当前允许的数据根：\n  {roots_hint}\n"
            f"如需新增，请在 data_access/config/datasets.yaml 注册数据集。"
        )
