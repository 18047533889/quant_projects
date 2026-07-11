"""COS 路径工具（factor_engine 内部版）。

通过 ``admin-cos`` 命令行操作腾讯云对象存储。
仅在目标路径以 ``cos://`` 开头时触发，否则透明地回退本地文件系统。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _is_cos(path: str | Path) -> bool:
    """判断路径是否为 COS 路径。"""
    return str(path).startswith("cos://")


def cos_read_parquet(cos_path: str) -> Any:
    """从 COS 读取 parquet 文件到 DataFrame。

    Args:
        cos_path: 形如 cos://bucket/prefix/data.parquet

    Returns:
        pd.DataFrame
    """
    import pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["admin-cos", "cat", cos_path],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                f"COS 读取失败: {cos_path}\n{result.stderr.decode()}"
            )
        with open(tmp_path, "wb") as f:
            f.write(result.stdout)
        return pd.read_parquet(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def cos_write_parquet(df: Any, cos_path: str) -> None:
    """将 DataFrame 写入 COS parquet 文件。

    流程: 本地临时 parquet → admin-cos cp → 清理临时文件。

    Args:
        df: pd.DataFrame
        cos_path: 形如 cos://bucket/prefix/data.parquet
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        result = subprocess.run(
            ["admin-cos", "cp", tmp_path, cos_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"COS 写入失败: {cos_path}\n{result.stderr}"
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def cos_exists(cos_path: str) -> bool:
    """检查 COS 路径是否存在。"""
    if not _is_cos(cos_path):
        return Path(cos_path).exists()

    result = subprocess.run(
        ["admin-cos", "ls", cos_path],
        capture_output=True, timeout=30,
    )
    return result.returncode == 0


def cos_ensure_dir(cos_dir: str) -> bool:
    """确保 COS 目录存在（通过上传 .empty 标记文件）。"""
    if not _is_cos(cos_dir):
        Path(cos_dir).mkdir(parents=True, exist_ok=True)
        return True

    # COS 目录通过上传一个空标记文件来"创建"
    marker = f"{cos_dir.rstrip('/')}/.empty"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".empty") as f:
        tmp = f.name
        f.write(b"")

    try:
        cp = subprocess.run(
            ["admin-cos", "cp", tmp, marker],
            capture_output=True, text=True, timeout=30,
        )
        return cp.returncode == 0
    finally:
        Path(tmp).unlink(missing_ok=True)


def cos_list_dir(cos_dir: str) -> list[str]:
    """列出 COS 目录下的文件/子目录名列表。

    Returns:
        文件名列表（不含路径前缀）。
    """
    if not _is_cos(cos_dir):
        if not Path(cos_dir).exists():
            return []
        return [p.name for p in Path(cos_dir).iterdir()]

    result = subprocess.run(
        ["admin-cos", "ls", f"{cos_dir}/"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []

    entries: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "|" not in line or "KEY" in line or "---" in line or "TOTAL" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0]:
            name = parts[0].rstrip("/").split("/")[-1]
            if name:
                entries.append(name)
    return entries
