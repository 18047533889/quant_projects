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

    R29-P0 #204：弃 ``to_pandas() + groupby``——分区枚举走 DuckDB ``SELECT
    DISTINCT``（typed）、分区筛选走 typed CAST，全程 Arrow，无 pandas 中转。
    """
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
    total = 0
    for rel_dir in sorted(partition_rel_dirs(table, pcols)):
        sub = filter_table_by_partition(table, rel_dir)
        if sub is None or sub.num_rows == 0:
            continue
        write_partition_drop_cols(gen_dir, rel_dir, sub, pcols)
        total += sub.num_rows
    return total


# R29-P0 #200：显式空 generation 的标记文件（delete-all 全分区删光时写入）。
_EMPTY_GENERATION_META = "generation.meta.json"


def validate_generation(gen_dir: Path, *, allow_empty: bool = False) -> int:
    """新代必须至少有一个可读 parquet footer；返回总行数。

    R29-P0 #200：``allow_empty=True`` 时允许显式空 generation（delete-all 全分区
    删光）——写入 ``generation.meta.json``（row_count=0, complete=true）标记，
    读者据此识别「权威空代」而非「代缺失」（不写空 parquet、不 glob 出旧数据）。
    """
    import pyarrow.parquet as pq

    files = sorted(gen_dir.rglob("*.parquet"))
    if not files:
        if allow_empty:
            gen_dir.mkdir(parents=True, exist_ok=True)
            meta = gen_dir / _EMPTY_GENERATION_META
            meta.write_text(
                json.dumps({"row_count": 0, "complete": True}, ensure_ascii=False),
                encoding="utf-8",
            )
            return 0
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


# ---- R28-26：generation copy-on-write（COW）-------------------------------

def _hive_value(v: Any) -> str:
    """分区值 → hive 目录段字符串（与 write_generation_files 的 pandas 输出一致）。"""
    if v is None:
        return "None"
    return str(v)


def _arrow_type_sql(arrow_type: Any) -> str:
    """pyarrow 类型 → DuckDB CAST 目标类型（分区字符串参数 typed 比较用）。"""
    import pyarrow as pa

    if pa.types.is_integer(arrow_type):
        return "BIGINT" if pa.types.is_int64(arrow_type) else "INTEGER"
    if pa.types.is_floating(arrow_type):
        return "DOUBLE"
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    if pa.types.is_date(arrow_type):
        return "DATE"
    return "VARCHAR"


def partition_rel_dirs(table: Any, partition_by: Sequence[str]) -> set[str]:
    """表中出现的分区相对目录集（``year=2024/month=01``），typed→str。

    R28-26：COW 需要知道「新表会碰哪些分区」，未碰的分区才能硬链接复用。
    用 DuckDB ``SELECT DISTINCT``（typed，不用 pandas 字符串拼接）。
    """
    pcols = [c for c in partition_by if c in getattr(table, "column_names", ())]
    if not pcols:
        return set()
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        con.register("_t", table)
        rows = con.execute(
            f"SELECT DISTINCT {', '.join(pcols)} FROM _t"
        ).fetchall()
    finally:
        con.close()
    out: set[str] = set()
    for vals in rows:
        out.add("/".join(f"{c}={_hive_value(v)}" for c, v in zip(pcols, vals)))
    return out


def hardlink_cow(
    src_gen: Path,
    dst_gen: Path,
    *,
    skip_rel_dirs: set[str],
) -> list[Path]:
    """COW：把 src 代**未变更**的分区 ``data.parquet`` 硬链接到 dst 代。

    更新时间复杂度从「O(整个历史数据集)」降到「O(变更分区)」：旧分区只加一个
    inode 引用（无数据拷贝），新分区由调用方全新写。跨文件系统/硬链接失败
    回退 copy。返回硬链接的 parquet 文件列表。
    """
    dst_gen.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    for data_file in sorted(src_gen.rglob("data.parquet")):
        rel_dir = data_file.parent.relative_to(src_gen).as_posix()
        if rel_dir in skip_rel_dirs:
            continue
        target = dst_gen / rel_dir / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(data_file), str(target))
        except OSError:
            shutil.copy2(str(data_file), str(target))
        linked.append(target)
    return linked


def _rel_dir_parts(rel_dir: str) -> dict[str, str]:
    return {
        seg.split("=", 1)[0]: seg.split("=", 1)[1]
        for seg in rel_dir.split("/")
        if "=" in seg
    }


def read_partition_typed(src_gen: Path, rel_dir: str) -> Any | None:
    """读 src 代某个分区，**用 DuckDB hive_partitioning 重建分区列**（typed）。

    分区列在目录名里、不在 parquet 文件里——``pq.read_table`` 读回会丢列。这里
    ``hive_partitioning=true`` 让 DuckDB 从目录重建 ``year=2024/month=01`` → 与
    新表 schema 一致，可直接 concat/merge。
    """
    import duckdb

    p = src_gen / rel_dir / "data.parquet"
    if not p.exists():
        return None
    con = duckdb.connect(":memory:")
    try:
        q = f"SELECT * FROM read_parquet('{p}', hive_partitioning=true)"
        return con.execute(q).to_arrow_table()
    finally:
        con.close()


def filter_table_by_partition(table: Any, rel_dir: str) -> Any:
    """从整表筛出属于某分区的行（typed——分区字符串参数按列类型 CAST 比较）。

    返回 None 表示 rel_dir 的分区列不全在表中（保守：调用方按全部行处理）。
    """
    import duckdb

    rel_parts = _rel_dir_parts(rel_dir)
    if not rel_parts:
        return table
    con = duckdb.connect(":memory:")
    try:
        con.register("_t", table)
        conds: list[str] = []
        params: list[Any] = []
        for col, val in rel_parts.items():
            if col not in table.column_names:
                continue
            conds.append(
                f"_t.{col} = CAST(? AS {_arrow_type_sql(table[col].type)})"
            )
            params.append(val)
        if not conds:
            return table
        return con.execute(
            f"SELECT * FROM _t WHERE {' AND '.join(conds)}", params
        ).to_arrow_table()
    finally:
        con.close()


def rows_in_partition(table: Any, rel_dir: str) -> int:
    """表中某分区的行数（delete COW 判定部分删除用）。"""
    sub = filter_table_by_partition(table, rel_dir)
    return 0 if sub is None else sub.num_rows


def write_partition_drop_cols(
    dst_gen: Path,
    rel_dir: str,
    table: Any,
    partition_by: Sequence[str],
) -> None:
    """把合并后的表写入 dst 代某分区：**丢掉分区列**（与 write_generation_files
    约定一致——分区列进目录、不进文件），写 ``data.parquet``。"""
    import pyarrow.parquet as pq

    d = dst_gen / rel_dir
    d.mkdir(parents=True, exist_ok=True)
    pcols = [c for c in partition_by if c in table.column_names]
    if pcols:
        table = table.drop_columns(pcols)
    pq.write_table(table, d / "data.parquet")


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
    # R28-26：多 key 合并**不再用 pandas** ``astype(str)+"|" join + isin``——那套
    # 既有 key collision/类型丢失风险又慢。改 DuckDB **typed anti-join**：
    # ``NOT EXISTS ... IS NOT DISTINCT FROM``（NULL-safe，不做字符串拼接）。
    if len(keys) == 1:
        col = keys[0]
        keep_mask = pc.invert(pc.is_in(old_table[col], value_set=new_table[col]))
        drop = old_table.filter(keep_mask)
    else:
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.register("_old", old_table)
            con.register("_new", new_table)
            conds = " AND ".join(
                f"_old.{k} IS NOT DISTINCT FROM _new.{k}" for k in keys
            )
            drop = con.execute(
                "SELECT _old.* FROM _old WHERE NOT EXISTS "
                f"(SELECT 1 FROM _new WHERE {conds})"
            ).to_arrow_table()
        finally:
            con.close()
    return pa.concat_tables([drop, new_table], promote_options="default")
