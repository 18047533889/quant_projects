"""R32-P0-102/103: Generation-based atomic write with immutable generation + pointer.

替代两次 rename 的 missing-target 窗口风险，使用不可变 generation 目录 +
原子指针文件切换模式。
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from data_access.core.atomic import atomic_write_json
from data_access.core.exceptions import DataError

logger = logging.getLogger("data_access.write.generation")


def generate_generation_id() -> str:
    """生成唯一 generation ID（时间戳 + UUID）。"""
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"g{ts}_{uid}"


def get_current_generation_id(target_dir: Path) -> str | None:
    """读取当前 generation 指针。

    Returns:
        当前 generation ID，若不存在则返回 None
    """
    pointer_file = target_dir / ".generation_pointer"
    if not pointer_file.exists():
        return None
    try:
        with open(pointer_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("generation_id")
    except Exception:
        return None


def write_generation_pointer(
    target_dir: Path,
    generation_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """原子写入 generation 指针（单文件原子替换，无 missing-target 窗口）。

    Args:
        target_dir: 目标数据集目录
        generation_id: 新 generation ID
        metadata: 可选元数据（rows、timestamp 等）
    """
    pointer_file = target_dir / ".generation_pointer"
    pointer_data = {
        "generation_id": generation_id,
        "updated_at": time.time(),
        "metadata": metadata or {},
    }
    atomic_write_json(pointer_file, pointer_data)


def atomic_publish_with_generation(
    target_dir: Path,
    candidate_dir: Path,
    archive_parent: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """使用不可变 generation + 原子指针的发布模式（R32-P0-103）。

    流程：
        1. candidate → target/.generations/<generation_id>（不可变）
        2. 旧 generation 移至 archive（如果存在）
        3. 原子更新 .generation_pointer 指向新 generation
        4. 读者通过指针文件解析当前 generation，无 missing-target 窗口

    Args:
        target_dir: 目标数据集根目录
        candidate_dir: 已准备好的候选数据
        archive_parent: 归档目录父路径
        metadata: 发布元数据

    Returns:
        {"generation_id": str, "archive_path": str | None}
    """
    generation_id = generate_generation_id()
    generations_dir = target_dir / ".generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    
    new_generation_dir = generations_dir / generation_id
    old_generation_id = get_current_generation_id(target_dir)
    archive_path: Path | None = None
    
    # Step 1: 移动 candidate 到不可变 generation 目录
    if new_generation_dir.exists():
        raise DataError(
            f"Generation {generation_id} 已存在（UUID 碰撞或时钟回拨），拒绝发布"
        )
    os.rename(str(candidate_dir), str(new_generation_dir))
    
    # Step 2/3: commit the pointer while the old generation remains in place.
    # Archiving before this commit creates a dangling old pointer if the pointer
    # write times out (the gateway-timeout failure mode this path must avoid).
    try:
        write_generation_pointer(target_dir, generation_id, metadata)
    except Exception:
        # The old generation was intentionally left in place; remove only the
        # unpublished candidate so the existing pointer remains valid.
        try:
            if new_generation_dir.exists():
                shutil.rmtree(new_generation_dir)
        except OSError as rollback_exc:
            raise DataError("generation pointer 写入失败，候选 generation 清理失败") from rollback_exc
        raise

    # Step 4: archive old generation after the pointer commit.  Failure here is
    # non-destructive: readers already have a valid new generation, and the old
    # generation remains available for recovery/audit.
    if old_generation_id:
        old_generation_dir = generations_dir / old_generation_id
        if old_generation_dir.exists():
            archive_path = archive_parent / (
                f"gen_{old_generation_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(str(old_generation_dir), str(archive_path))
            except OSError:
                logger.warning("old generation archive failed; retaining it: %s", old_generation_dir)

    return {
        "generation_id": generation_id,
        "archive_path": str(archive_path) if archive_path else None,
    }


def read_current_generation_data(target_dir: Path) -> Path | None:
    """读取当前 generation 的数据目录。

    Returns:
        当前 generation 数据目录的 Path，若无则返回 None
    """
    generation_id = get_current_generation_id(target_dir)
    if not generation_id:
        return None
    
    generation_dir = target_dir / ".generations" / generation_id
    return generation_dir if generation_dir.exists() else None


__all__ = [
    "generate_generation_id",
    "get_current_generation_id",
    "write_generation_pointer",
    "atomic_publish_with_generation",
    "read_current_generation_data",
]
