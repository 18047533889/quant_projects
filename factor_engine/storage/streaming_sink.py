# -*- coding: utf-8 -*-
"""FE 100k GO P0#12 —— streaming sink（GO prompt §21 + §72）。

10 万 × 5000 × 多年结果**禁止一次性常驻内存后整表写出**。正确路径是：

```
root done → validate → encode → partition write → release root buffer
```

本模块用 pyarrow ``ParquetWriter`` 做分片（shard）流式落盘：

- 每个 ``factor`` 写一个独立 shard 文件（``<campaign>/shards/<root_id>.parquet``）；
- 每篇 shard 由多个 RecordBatch 流式追加，只缓存当前 batch（不物化全因子）；
- 写盘完成后立即更新 checkpoint（见 ``dry_run_checkpoint.py``）；
- 崩溃最多丢「正在写的那一个 shard」，其余已完成 shard 全部保留。

契约：
- 每根 root 独立可校验（checksum 写入 sidecar），最终 ``load_all()`` 可逐 shard
  重读并与 in-memory 基线对拍（见 tests/test_dry_run_ladder.py 的 (b) 与
  tests/test_production_factor_ladder.py）；
- 禁止 ``dict[factor_name] = full_history_dataframe`` 形式的一站式大内存组装。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 确定性 schema：date / asset / float value
# ---------------------------------------------------------------------------
_SINK_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("asset", pa.string()),
        ("value", pa.float64()),
    ]
)


@dataclass(frozen=True)
class StreamRow:
    """单个待写出的结果行。"""

    date: str
    asset: str
    value: float


@dataclass
class ShimWriteResult:
    """一篇 shard 的写盘元数据 + checksum。"""

    root_id: str
    path: str
    num_rows: int
    bytes_written: int
    checksum: str  # sha256 over sorted rows


class StreamingSink:
    """分片、可续写、只缓存单 batch 的结果落盘器。

    用法::

        sink = StreamingSink(output_dir=Path("/tmp/.../lake"), campaign_id="run1")
        with sink.open(root_id="factor_00042") as shard:
            shard.add(StreamRow("2024-01-02", "A0001", 0.5))
            shard.add_many([...])
        # 退出 with 块 = 该 shard 已写出 + checkpoint 已更新
    """

    def __init__(
        self,
        *,
        output_dir: Path | str,
        campaign_id: str,
        row_group_size: int = 65536,
        compression: str = "SNAPPY",
        checksum_policy: str = "sorted",  # sorted | row-order
    ) -> None:
        self.output_dir = Path(output_dir)
        self.campaign_id = str(campaign_id)
        self.row_group_size = int(row_group_size)
        self.compression = compression
        self.checksum_policy = checksum_policy
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    # -- 单 shard 流式写入器 -----------------------------------------------
    @dataclass
    class _ShardWriter:
        sink: "StreamingSink"
        root_id: str

        _pq_writer: Any = field(default=None, init=False, repr=False)
        _rows: list[StreamRow] = field(default_factory=list, init=False, repr=False)
        _all_rows: list[StreamRow] | None = field(default=None, init=False, repr=False)
        _num_rows: int = field(default=0, init=False)
        _path: Path = field(default=None, init=False, repr=False)  # type: ignore[assignment]

        def __enter__(self) -> "StreamingSink._ShardWriter":
            self._path = self.sink.shard_dir / f"{self.root_id}.parquet"
            self._pq_writer = pq.ParquetWriter(
                self._path,
                _SINK_SCHEMA,
                compression=self.sink.compression,
                write_statistics=True,
            )
            return self

        def add(self, date: str, asset: str, value: float) -> None:
            self._rows.append(StreamRow(str(date), str(asset), float(value)))
            if len(self._rows) >= self.sink.row_group_size:
                self._flush_batch()

        def add_many(self, rows: Iterable[StreamRow]) -> None:
            for row in rows:
                self.add(row.date, row.asset, row.value)

        def _flush_batch(self) -> None:
            tbl = pa.table(
                {
                    "date": [r.date for r in self._rows],
                    "asset": [r.asset for r in self._rows],
                    "value": [r.value for r in self._rows],
                },
                schema=_SINK_SCHEMA,
            )
            self._pq_writer.write_table(tbl)
            self._num_rows += len(self._rows)
            # 需要确定性 sorted checksum 时保留全部行（内存中排序哈希，不重读盘）
            if self.sink.checksum_policy == "sorted":
                if self._all_rows is None:
                    self._all_rows = list(self._rows)
                else:
                    self._all_rows.extend(self._rows)
            self._rows = []  # 释放当前 batch，绝不累积全文

        def __exit__(self, exc_type, exc, tb) -> bool:
            if exc is not None:
                # 异常：丢弃未 flush 的当前 batch，close 已写部分，不写 checkpoint。
                self._pq_writer.close()
                self.sink._fail_shard(self.root_id, str(exc))
                return False
            try:
                self._flush_batch()
                self._pq_writer.close()
            except BaseException as e:  # noqa: BLE001 -- 落盘失败要诚实上报
                self.sink._fail_shard(self.root_id, str(e))
                return False
            checksum = self._sha()
            stats = self._path.stat()
            result = ShimWriteResult(
                root_id=self.root_id,
                path=str(self._path),
                num_rows=self._num_rows,
                bytes_written=int(stats.st_size),
                checksum=checksum,
            )
            self.sink._record_shard(result)
            return False

        def _sha(self) -> str:
            """确定性 checksum：行排序后哈希。

            排序保证跨重跑确定性（同一数据 → 同一 checksum），且与
            in-memory 基线对拍（tests 直接对该值 1e-12 级比对）。
            在 ``checksum_policy="sorted"`` 下使用**内存累积行**（``_all_rows``）
            而不是重读刚写完的 parquet —— 后者在共享进程（多 worker 线程持 GIL）
            下会触发 pyarrow.dataset 的再 import / /proc 遍历停顿（环境问题），
            拖慢整个 sink 提交路径。
            """
            if self.sink.checksum_policy == "sorted" and self._all_rows is not None:
                triples = sorted(
                    ((r.date, r.asset, r.value) for r in self._all_rows),
                    key=lambda t: (t[0], t[1], str(t[2])),
                )
                hasher = hashlib.sha256()
                for d, a, v in triples:
                    hasher.update(f"{d}|{a}|{v!r}\n".encode("utf-8"))
                return hasher.hexdigest()
            tbl = pq.read_table(self._path)
            cols = {"date": tbl["date"].to_pylist(), "asset": tbl["asset"].to_pylist(),
                    "value": tbl["value"].to_pylist()}
            triples = sorted(zip(cols["date"], cols["asset"], cols["value"]), key=lambda t: (t[0], t[1], str(t[2])))
            hasher = hashlib.sha256()
            for d, a, v in triples:
                hasher.update(f"{d}|{a}|{v!r}\n".encode("utf-8"))
            return hasher.hexdigest()

    # -- 顶层接口 -----------------------------------------------------------
    def open(self, root_id: str) -> "_ShardWriter":
        """打开一篇 root 的流式 shard（with 块结束即提交该 shard）。"""
        return StreamingSink._ShardWriter(self, root_id=str(root_id))

    def _record_shard(self, result: ShimWriteResult) -> None:
        """写完成一篇 shard：持久化 sidecar manifest + 推进 checkpoint。"""
        sidecar = self.shard_dir / f"{result.root_id}.sidecar.json"
        sidecar.write_text(
            json.dumps(
                {
                    "root_id": result.root_id,
                    "path": result.path,
                    "num_rows": result.num_rows,
                    "bytes_written": result.bytes_written,
                    "checksum": result.checksum,
                    "campaign_id": self.campaign_id,
                    "checksum_policy": self.checksum_policy,
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _fail_shard(self, root_id: str, reason: str) -> None:
        """失败登记（供 checkpoint 的 failed root 表使用）。"""
        fail_mark = self.shard_dir / f"{root_id}.failed"
        fail_mark.write_text(json.dumps({"root_id": root_id, "reason": reason}, indent=2))

    def read_all(self) -> pa.Table:
        """把全部已完成 shard 重读为一张 Arrow 表（对拍用途）。"""
        shards = sorted(self.shard_dir.glob("*.parquet"))
        tables = [pq.read_table(str(s)) for s in shards]
        if not tables:
            return pa.table({"date": [], "asset": [], "value": []}, schema=_SINK_SCHEMA)
        return pa.concat_tables(tables, promote_options="default")
