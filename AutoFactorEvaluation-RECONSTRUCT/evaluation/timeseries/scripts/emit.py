"""时序评估产物写出工具。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：落盘下游主消费 parquet 产物，并生成治理所需 JSON（manifest/validation）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """将 DataFrame 以 parquet 格式写出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    """写出 JSON 产物（同目录临时文件 + replace 原子落盘）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
    tmp_path.replace(path)


def sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 哈希值。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    *,
    factor_id: str,
    eval_run_id: str,
    environment: str,
    price_field: str,
    cache_hit: bool,
    timezone_policy: str,
    config: dict[str, Any],
    artifacts: dict[str, Path],
) -> Path:
    """写出本次运行 manifest。

    说明：
        - 下游核心数值读取来自 parquet；
        - manifest 用于文件发现、哈希校验与运行元信息追溯。
    """
    artifact_meta = []
    for name, p in artifacts.items():
        if p.exists():
            artifact_meta.append(
                {
                    "name": name,
                    "path": str(p.resolve()),
                    "sha256": sha256_file(p),
                }
            )
    manifest = {
        "producer": "evaluation.timeseries",
        "factor_id": factor_id,
        "eval_run_id": eval_run_id,
        "environment": environment,
        "price_field": price_field,
        "forward_return_cache_hit": cache_hit,
        "timezone_policy": timezone_policy,
        "evaluation_config": config,
        "artifacts": artifact_meta,
    }
    path = out_dir / "manifest.json"
    write_json(manifest, path)
    return path

