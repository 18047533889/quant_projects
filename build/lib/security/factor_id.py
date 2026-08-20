# -*- coding: utf-8 -*-
"""FactorId 文件系统安全 domain validator（R32-P0-035/036, P0-043）。

factor_id 是稳定机器 ID（区别于 display_name 人类名称 / description 说明）。
HTTP validator 历史用 ``[:128]`` 截断会让两个不同 ID 静默碰撞；Materializer /
Delete 直接把 factor_id 拼到 path 也没有统一 gate。本模块是**单一 domain 层
验证器** —— HTTP、Direct Python API、materializer、catalog、delete 全部复用。

规则（R32 §7）：
- 长度超限 → reject（绝不截断）；
- 禁止 path separator（``/``、``\\``）、``..``、控制字符；
- Unicode 统一 NFC 归一化（case 策略：保留大小写但 NFC 归一，禁止视觉同形歧义）；
- reserved ids（如 ``.`` / ``..`` / 空 / ``factors`` 等目录保留名）拒绝；
- 路径 ``resolve()`` 后必须仍在 allowed root 下（``confine_path``）；
- symlink escape 防护（resolve 后 realpath 仍在 root 内）；
- delete_files / service root / cache root / staging / artifact / checkpoint 全用
  同一个 ``confine_path``。
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

MAX_FACTOR_ID_LENGTH = 128

#: 控制字符 / 路径分隔符 / 相对穿越片段。
_FORBIDDEN_CHARS_RE = re.compile(r"[\x00-\x1f\x7f/\\]")
_FORBIDDEN_SEGMENTS = frozenset({".", "..", ""})

#: R40 #145: production factor_id 字符集白名单 —— NFC 归一化防不住视觉同形歧义
#: （西里尔 ``а`` vs 拉丁 ``a``、全角 ``：`` vs 半角 ``:``）。production 只接受
#: ASCII 字母/数字/``_``/``.``/``:``/``-``；research 放行但告警。
_PRODUCTION_FACTOR_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

#: 保留名：因子湖目录拓扑占用的名字，不可用作 factor_id。
_RESERVED_IDS = frozenset({
    "_catalog.sqlite",
    "factors",
    "checkpoints",
    "manifest",
    ".identity.json",
    ".lock",
    "catalog",
    "staging",
    "artifacts",
    "cache",
})


class FactorIdError(ValueError):
    """factor_id 不合法（长度超限 / 路径穿越 / 控制字符 / 保留名 / 越界路径）。"""


def normalize_factor_id(factor_id: Any) -> str:
    """Unicode NFC 归一化 + 强制字符串。

    R32 §7: Unicode NFC 策略 —— 组合字符与分解字符不得产生不同 factor_id。
    """
    text = str(factor_id or "").strip()
    return unicodedata.normalize("NFC", text)


def validate_factor_id(
    factor_id: Any,
    *,
    max_length: int = MAX_FACTOR_ID_LENGTH,
    production: bool | None = None,
) -> str:
    """R32-P0-035/036: 统一 FactorId domain gate。

    - 长度超限 → ``FactorIdError``（绝不截断 —— ``[:128]`` 会让两个不同 ID
      静默碰撞）；
    - 禁 path separator / ``..`` / 控制字符；
    - Unicode NFC 归一化后返回；
    - 保留名拒绝。
    - R40 #145: production 下额外执行 ASCII 字符集白名单（``^[A-Za-z0-9_.:-]+$``）
      —— 拒绝同形异义（homoglyph）字符；非 production 放行但告警。

    参数:
        factor_id: 待校验的因子 ID
        max_length: 长度上限（可选）
        production: 是否按 production 规则（``None`` 自动按 run mode 解析）

    返回:
        归一化后的 factor_id

    Raises:
        FactorIdError: 任一校验失败
    """
    if factor_id is None:
        raise FactorIdError("factor_id is required")
    text = normalize_factor_id(factor_id)
    if not text:
        raise FactorIdError("factor_id must not be empty")
    if len(text) > int(max_length):
        raise FactorIdError(
            f"factor_id too long: {len(text)} chars exceeds max {max_length}. "
            "Truncating would silently collide distinct IDs — reject instead. "
            f"value={text[:48]!r}…"
        )
    if _FORBIDDEN_CHARS_RE.search(text):
        raise FactorIdError(
            f"factor_id contains forbidden characters (path separators, control "
            f"chars): {text!r}"
        )
    if any(seg in _FORBIDDEN_SEGMENTS for seg in text.split(" ")):
        raise FactorIdError(f"factor_id contains forbidden segment: {text!r}")
    lowered = text.lower()
    if lowered in _RESERVED_IDS:
        raise FactorIdError(f"factor_id {text!r} is reserved")
    # ``..`` 作为连续片段：即使不在 / 分隔后也要拒绝。
    if ".." in text:
        raise FactorIdError(f"factor_id must not contain '..': {text!r}")
    # R40 #145: production 字符集白名单（homoglyph 防御）。
    if not _PRODUCTION_FACTOR_ID_RE.match(text):
        if production is None:
            from runtime.production_policy import is_production_mode

            production = is_production_mode()
        if production:
            raise FactorIdError(
                f"factor_id {text!r} contains characters outside the production "
                "whitelist [A-Za-z0-9_.:-]; homoglyph / non-ASCII letters are "
                "rejected in production (R40 #145)"
            )
        import warnings

        warnings.warn(
            f"factor_id {text!r} contains characters outside the production "
            "whitelist; non-ASCII / homoglyph IDs are research-only (R40 #145)",
            RuntimeWarning,
            stacklevel=2,
        )
    return text


def confine_path(root: str | Path, candidate: str | Path, *, label: str = "path") -> Path:
    """R32-P0-060: resolve-under-root 检查（所有写/删路径共用）。

    ``candidate`` ``resolve()``（含 symlink 解析）后必须仍在 ``root`` 内。
    返回解析后的绝对 Path。越界 → ``FactorIdError``。

    用于 factor lake、service root、cache root、staging、artifact、checkpoint
    全部路径。
    """
    root_resolved = Path(root).expanduser().resolve()
    candidate_resolved = Path(candidate).expanduser().resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        raise FactorIdError(
            f"{label} escapes allowed root: candidate={candidate_resolved} "
            f"root={root_resolved}"
        )
    return candidate_resolved


def factor_dir_for(
    lake_root: str | Path,
    factor_id: Any,
    *,
    max_length: int = MAX_FACTOR_ID_LENGTH,
) -> Path:
    """R32-P0-036: 因子目录解析 —— 先 domain-validate，再 confine 到 lake root。

    ``factors/`` 目录名固定，factor_id 作为二级目录段；``confine_path`` 保证
    resolve 后仍在 lake root 下（防 symlink / ``..`` 逃逸）。
    """
    fid = validate_factor_id(factor_id, max_length=max_length)
    lake = Path(lake_root).expanduser().resolve()
    candidate = lake / "factors" / fid
    return confine_path(lake, candidate, label="factor_dir")


def validate_delete_factor(
    lake_root: str | Path,
    factor_id: Any,
) -> Path:
    """R32-P0-060: delete_files 同样走 root confinement —— 删除路径越界=灾难。"""
    return factor_dir_for(lake_root, factor_id)


def is_safe_factor_id(factor_id: Any) -> bool:
    """快速布尔检查（不抛错），用于 audit / 扫描。"""
    try:
        validate_factor_id(factor_id)
        return True
    except FactorIdError:
        return False


__all__ = [
    "FactorIdError",
    "MAX_FACTOR_ID_LENGTH",
    "normalize_factor_id",
    "validate_factor_id",
    "confine_path",
    "factor_dir_for",
    "validate_delete_factor",
    "is_safe_factor_id",
]
