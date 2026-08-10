"""R27-G —— generation_pointer 数据集的**原子代写**（append/upsert/overwrite/delete）。

读路径（``registry/loader.py: ParametricDataset.resolve_paths``）只消费
``root/manifest.json`` 的 ``generation`` 指针指向那一代 ``generation/<gid>/``，
**绝不用 ``generation/*`` 通配**。因此写路径只需：

    写完整新一代（不可变目录） → **原子 flip manifest.json.generation**（单指针）

读者要么看到旧代（旧 manifest），要么新代（新 manifest），**绝不看到新旧混合 /
半写状态**。这从根上消除三类非原子性：
  - append 直接写 live target → 读者看到「旧数据 + 前 7 个新 partition」；
  - upsert/delete 按分区逐个 read-merge-rename → mixed generation；
  - overwrite/publish「两次 rename」之间的 target 缺失窗口。

只对 ``generation_pointer: true`` 的 ParametricDataset 生效（当前 factor_matrix）。
非 generation 数据集保留原 mutation_lock + candidate-rename 语义（limitation 记录）。
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import DataError, ValidationError


def next_generation_id() -> str:
    """新一代 id（时间戳 + 随机后缀；同一代内写多文件不撞）。"""
    return f"g-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


def read_generation_manifest(root: Path) -> dict[str, Any]:
    """读 manifest.json；不存在/不可读 → {}（视为无指针）。"""
    mp = root / "manifest.json"
    if not mp.exists():
        return {}
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        raise DataError(
            f"generation manifest.json 不可读（{mp}）: {exc}（fail-closed，不拿旧代冒充）"
        ) from exc


def flip_generation_pointer(root: Path, gid: str) -> None:
    """**原子** flip manifest.json.generation = gid（temp 写 + os.replace 单指针切换）。

    这是整个代写的 commit point：替换前后读者各自读到完整旧代/完整新代。
    """
    root.mkdir(parents=True, exist_ok=True)
    mp = root / "manifest.json"
    manifest = read_generation_manifest(root)
    manifest["generation"] = gid
    manifest.setdefault("manifest_version", 1)
    tmp = root / f".manifest.json.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    os.replace(str(tmp), str(mp))


def current_generation_dir(root: Path) -> Path | None:
    """当前 manifest 指向的代目录；无 manifest/无指针/目录缺失 → None。"""
    manifest = read_generation_manifest(root)
    gid = manifest.get("generation")
    if not gid:
        return None
    gen_dir = root / "generation" / str(gid)
    return gen_dir if gen_dir.is_dir() else None


def copy_generation(src: Path, dst: Path) -> None:
    """把一代内容（含 hive 分区布局）拷进新代目录；跳过隐藏临时文件。"""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.rglob("*"):
        if entry.is_file() and not entry.name.startswith("."):
            rel = entry.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(entry), str(target))


def write_generation_files(
    table: Any,
    gen_dir: Path,
    partition_by: Sequence[str] | None,
) -> int:
    """把表写成 generation 布局：``<partition col=value>/data.parquet``。

    与上游 FactorEngine（test_r14_matrix_generation_resolver 的 ``_write_generation``）
    约定一致：每个 hive 分区写 ``data.parquet``，分区列进目录、不进文件
    （读路径 ``hive_partitioning=true`` 会从目录重建列）。返回总行数。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    gen_dir.mkdir(parents=True, exist_ok=True)
    if not partition_by:
        pq.write_table(table, gen_dir / "data.parquet")
        return table.num_rows

    pcols = [c for c in partition_by if c in table.column_names]
    if len(pcols) != len(partition_by):
        raise ValidationError(
            f"generation 分区列 {list(partition_by)} 不全在表中（缺 "
            f"{set(partition_by) - set(pcols)}）"
        )
    df = table.to_pandas()
    total = 0
    for key, group in df.groupby(list(pcols), dropna=False):
        if isinstance(key, tuple):
            parts = dict(zip(pcols, key))
        else:
            parts = {pcols[0]: key}
        dir_path = gen_dir
        for col in pcols:
            dir_path = dir_path / f"{col}={parts[col]}"
        dir_path.mkdir(parents=True, exist_ok=True)
        sub = group.drop(columns=list(pcols))
        sub_tbl = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(sub_tbl, dir_path / "data.parquet")
        total += sub_tbl.num_rows
    return total


def validate_generation(gen_dir: Path) -> int:
    """新代必须至少有一个可读 parquet footer；返回总行数。"""
    import pyarrow.parquet as pq

    files = sorted(gen_dir.rglob("*.parquet"))
    if not files:
        raise ValidationError(f"generation 目录无 parquet 文件，拒绝 flip：{gen_dir}")
    total = 0
    for fp in files:
        try:
            total += pq.read_metadata(str(fp)).num_rows
        except Exception as exc:
            raise ValidationError(
                f"generation 候选文件 footer 读取失败：{fp}（拒绝 flip）"
            ) from exc
    return total


def generation_layout(ds: Any, params: dict[str, Any]) -> tuple[Path, Path]:
    """解析 generation 数据集的 (root, glob_part)。

    root 为 ``resolve_root``（含 namespace/参数解析）；glob_part 为相对 root 的
    glob 尾段（``**/*.parquet`` / ``year=*/month=*/data.parquet`` 等）。
    """
    from data_access.registry.params_validation import (
        ParamSpec,
        validate_params,
    )
    from data_access.registry.paths import resolve_namespace_path

    specs = ds.param_specs or {
        k: ParamSpec(name=k, type=t) for k, t in ds.params_schema.items()
    }
    validated = validate_params(ds.name, specs, dict(params))
    root_tpl = resolve_namespace_path(ds.root_template).format(**validated)
    glob_tpl = resolve_namespace_path(ds.glob_template).format(**validated)
    return Path(root_tpl), Path(glob_tpl)


def merge_tables_by_keys(
    old_table: Any,
    new_table: Any,
    keys: Sequence[str],
) -> Any:
    """dataset 级 upsert 合并（generation 写用）：同 key 新值覆盖旧值。

    比逐分区 read-merge-rename 更彻底——整代合并后写入新一代，读者永远只看到
    完整一代，不存在「2024=new、2025=old」的 mixed generation。
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if old_table is None or old_table.num_rows == 0:
        return new_table
    if new_table is None or new_table.num_rows == 0:
        return old_table
    for k in keys:
        if k not in new_table.column_names or k not in old_table.column_names:
            raise ValidationError(f"upsert 合并键 {k!r} 不在表列中（generation 合并）")
    # 旧表去掉将被新表覆盖的 key 组合，再 concat 新表 → 整代幂等。
    # 多 key 用组合列做 is_in（单 key 直接用列）。
    if len(keys) == 1:
        col = keys[0]
        keep_mask = pc.invert(pc.is_in(old_table[col], value_set=new_table[col]))
        drop = old_table.filter(keep_mask)
    else:
        import pandas as pd

        old = old_table.to_pandas()
        new = new_table.to_pandas()
        old_keys = old[list(keys)].astype(str).agg("|".join, axis=1)
        new_keys = new[list(keys)].astype(str).agg("|".join, axis=1)
        old = old[~old_keys.isin(set(new_keys))]
        merged = pd.concat([old, new], ignore_index=True)
        return pa.Table.from_pandas(merged, preserve_index=False)
    return pa.concat_tables([drop, new_table], promote_options="default")
