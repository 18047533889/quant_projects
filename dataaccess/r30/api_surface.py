"""data_access.r30.api_surface —— R30-P2-004 API Surface 收敛。

目标:**减少维护与用户心智成本**。把大量同义的 ``read_*`` 历史入口收敛到
一组 canonical 入口:

  - canonical: ``read / scan / session / plan / write / publish``;
  - compat 别名: ``read_arrow / read_auto / read_frame / read_uri /
    read_joined / read_factors`` 保留兼容,**但不再新增同义 public entry**。

用户只需记住 canonical;旧代码继续可用(别名映射到 canonical 归属)。
"""
from __future__ import annotations

__all__ = [
    "PREFERRED_PUBLIC_API",
    "public_api_surface",
    "is_preferred",
    "compat_aliases",
]

#: 推荐公开入口(维护/心智成本最小集合)。
PREFERRED_PUBLIC_API: tuple[str, ...] = (
    "read",
    "scan",
    "session",
    "plan",
    "write",
    "publish",
)

#: compat 别名 → 归属 canonical。read_arrow 等保留 compat 但不再新增同义入口。
_COMPAT_ALIASES: dict[str, str] = {
    "read_arrow": "read",    # read 返回 pyarrow Table(默认)
    "read_auto": "read",     # 自动探测格式
    "read_frame": "read",    # 返回 pandas Frame
    "read_uri": "read",      # 从 URI 直接读
    "read_joined": "read",   # 多数据集 join 读
    "read_factors": "read",  # factor lake wide 读
}


def public_api_surface() -> dict:
    """canonical 入口列表 + compat 别名映射(别名 → 归属 canonical)。"""
    return {
        "preferred_public_api": list(PREFERRED_PUBLIC_API),
        "canonical_entries": list(PREFERRED_PUBLIC_API),
        "compat_aliases": dict(_COMPAT_ALIASES),
        "aliases_by_canonical": {
            canonical: tuple(sorted(a for a, c in _COMPAT_ALIASES.items() if c == canonical))
            for canonical in PREFERRED_PUBLIC_API
        },
    }


def is_preferred(name: str) -> bool:
    """``name`` 是否为 canonical 推荐入口。"""
    return name in PREFERRED_PUBLIC_API


def compat_aliases(name: str) -> tuple[str, ...]:
    """返回 ``name`` 的 compat 别名归属。

    语义:
      - ``name`` 是 canonical 入口 → 返回映射到它的全部 compat 别名
        (例如 ``compat_aliases("read")`` 含 ``read_arrow`` 等 6 个);
      - ``name`` 是 compat 别名 → 返回它所属的 canonical(单元素 tuple),
        方便调用方把旧别名归一到 preferred 入口;
      - 其他 → 空 tuple。
    """
    canonical = _COMPAT_ALIASES.get(name, name)
    if canonical not in PREFERRED_PUBLIC_API:
        return ()
    if name == canonical:
        return tuple(sorted(a for a, c in _COMPAT_ALIASES.items() if c == canonical))
    return (canonical,)
