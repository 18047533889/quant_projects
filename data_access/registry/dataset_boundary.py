"""R32-P0-100: Dataset-specific path boundary verification.

每个 dataset 只能访问其注册的 root 范围内的路径，防止 dataset A 的写操作
越界到 dataset B 的 root。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    from data_access.registry.loader import DatasetBase


def verify_path_belongs_to_dataset(
    path: Path,
    dataset: DatasetBase,
    operation: str = "access",
) -> None:
    """验证路径属于指定 dataset 的授权边界。

    Args:
        path: 待验证的绝对路径（已 canonicalize）
        dataset: Dataset 对象（含 authorized_root / root / root_template）
        operation: 操作类型（用于错误消息）

    Raises:
        ValidationError: 路径不在 dataset 授权边界内
    """
    from data_access.registry.paths import canonicalize, path_is_under, resolve_namespace_path

    # 获取 dataset 的授权根（按优先级）
    authorized_roots: list[Path] = []
    
    # 1. 优先使用 authorized_root（显式授权边界）
    if hasattr(dataset, "authorized_root") and dataset.authorized_root:
        root_str = resolve_namespace_path(str(dataset.authorized_root))
        authorized_roots.append(canonicalize(root_str))
    
    # 2. 次选 root_template 或 root
    if hasattr(dataset, "root_template") and dataset.root_template:
        root_str = resolve_namespace_path(str(dataset.root_template))
        authorized_roots.append(canonicalize(root_str))
    elif hasattr(dataset, "root") and dataset.root:
        root_str = resolve_namespace_path(str(dataset.root))
        authorized_roots.append(canonicalize(root_str))
    
    # 3. 特殊处理：StaticDataset 的 static_root
    if hasattr(dataset, "static_root") and dataset.static_root:
        root_str = resolve_namespace_path(str(dataset.static_root))
        authorized_roots.append(canonicalize(root_str))
    
    if not authorized_roots:
        raise ValidationError(
            f"Dataset '{dataset.name}' 缺少 root 配置，无法验证路径边界（R32-P0-100）"
        )
    
    # 验证路径在任一授权根下
    path_canonical = canonicalize(path)
    for authorized_root in authorized_roots:
        if path_is_under(path_canonical, authorized_root):
            return  # 通过验证
    
    # 不在任何授权根下 → 拒绝
    roots_str = ", ".join(str(r) for r in authorized_roots)
    raise ValidationError(
        f"R32-P0-100 path boundary violation: {operation} 路径 '{path}' "
        f"不在 dataset '{dataset.name}' 的授权边界内。"
        f"授权根: [{roots_str}]"
    )


__all__ = ["verify_path_belongs_to_dataset"]
