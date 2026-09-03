# -*- coding: utf-8 -*-
"""FE 100k GO P0#12 —— 生产 checkpoint / 断点续跑（GO prompt §72）。

支持 ``campaign manifest`` / ``completed root IDs`` / ``failed root IDs`` /
``source snapshot`` / ``factor hash`` / ``partition completion``。断点重启
**不得重算全部** —— 已完成 root 直接跳过。

文件布局（``checkpoint_dir/``）::

    campaign.json            campaign manifest（HEAD、时间、参数指纹、root 总数等）
    completed.json           {completed_root_id: {checkpoint marker}}
    failed.json              {failed_root_id: {code, bucket, reason, at}}
    partitions.json          {partition_key: {completed shard ids, …}}  (partition completion)

写盘是「atomic 提交」：先写 ``*.tmp`` 再 ``os.replace``，避免半写 corruption。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from factor_engine.runtime.failure_classification import (
    FailureReport,
    FailureCode,
    classify_exception,
)

_EXTRA_TMP_JSON_KW: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


@dataclass
class CampaignMeta:
    """campaign manifest 的元信息。"""

    campaign_id: str
    total_roots: int
    source_snapshot: str
    factor_hash: str | None = None
    git_head: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "total_roots": self.total_roots,
            "source_snapshot": self.source_snapshot,
            "factor_hash": self.factor_hash,
            "git_head": self.git_head,
            "created_at": self.created_at,
        }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, **_EXTRA_TMP_JSON_KW))
    os.replace(tmp, path)  # atomic


class DryRunCheckpoint:
    """可续写、可查询 completed/failed、可记录 partition completion 的 checkpoint。

    用法::

        ckpt = DryRunCheckpoint(checkpoint_dir=Path("/tmp/.../ckpt"))
        ckpt.init_campaign(CampaignMeta(campaign_id="run1", total_roots=100, source_snapshot="ss1"))
        if not ckpt.is_completed(root_id):
            ... compute ...
            ckpt.mark_completed(root_id, marker={"checksum": "…"})
        ckpt.mark_failed(root_id, exc)
    """

    def __init__(self, *, checkpoint_dir: Path | str) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.campaign_path = self.dir / "campaign.json"
        self.completed_path = self.dir / "completed.json"
        self.failed_path = self.dir / "failed.json"
        self.partitions_path = self.dir / "partitions.json"

    # -- campaign manifest --------------------------------------------------
    def init_campaign(self, meta: CampaignMeta) -> None:
        _atomic_write(self.campaign_path, meta.to_dict())

    def campaign(self) -> dict[str, Any]:
        if not self.campaign_path.exists():
            return {}
        return json.loads(self.campaign_path.read_text())

    def update_campaign(self, **fields: Any) -> None:
        data = self.campaign()
        data.update(fields)
        _atomic_write(self.campaign_path, data)

    # -- completed / failed roots ------------------------------------------
    def _completed(self) -> dict[str, Any]:
        if not self.completed_path.exists():
            return {}
        return json.loads(self.completed_path.read_text())

    def _failed(self) -> dict[str, Any]:
        if not self.failed_path.exists():
            return {}
        return json.loads(self.failed_path.read_text())

    def is_completed(self, root_id: str) -> bool:
        return root_id in self._completed()

    def is_failed(self, root_id: str) -> bool:
        return root_id in self._failed()

    def pending_roots(self, all_root_ids: list[str]) -> list[str]:
        done = set(self._completed()) | set(self._failed())
        return [r for r in all_root_ids if r not in done]

    def mark_completed(self, root_id: str, marker: dict[str, Any] | None = None) -> None:
        data = self._completed()
        data[str(root_id)] = {"checkpoint": marker or {}, "done": True}
        _atomic_write(self.completed_path, data)

    def mark_failed(self, root_id: str, exc: BaseException) -> None:
        code, bucket = classify_exception(exc)
        data = self._failed()
        entry = data.get(str(root_id), {})
        entry.update(
            {
                "code": str(code),
                "bucket": bucket,
                "reason": (str(exc) or "")[:2000],
                "count": entry.get("count", 0) + 1,
            }
        )
        data[str(root_id)] = entry
        _atomic_write(self.failed_path, data)

    @property
    def completed_roots(self) -> list[str]:
        return sorted(self._completed().keys())

    @property
    def failed_roots(self) -> list[str]:
        return sorted(self._failed().keys())

    def completed_count(self) -> int:
        return len(self._completed())

    def failed_count(self) -> int:
        return len(self._failed())

    # -- partition completion ----------------------------------------------
    def mark_partition_shard(self, partition_key: str, shard_id: str) -> None:
        data = self._partitions()
        part = data.setdefault(str(partition_key), {})
        shards = part.setdefault("completed_shards", [])
        if str(shard_id) not in shards:
            shards.append(str(shard_id))
            part["completed_shards"] = sorted(shards)
            part["count"] = len(shards)
        _atomic_write(self.partitions_path, data)

    def _partitions(self) -> dict[str, Any]:
        if not self.partitions_path.exists():
            return {}
        return json.loads(self.partitions_path.read_text())

    def partition(self, partition_key: str) -> dict[str, Any]:
        return self._partitions().get(str(partition_key), {})

    # -- summary ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign(),
            "completed_count": self.completed_count(),
            "failed_count": self.failed_count(),
            "completed_roots": self.completed_roots,
            "failed_roots": self.failed_roots,
            "partitions": self._partitions(),
        }

    def report(self, failure_report: FailureReport | None = None) -> FailureReport:
        """把 checkpoint 内已登记 failed 重建为 FailureReport（按类型回灌）。"""
        if failure_report is None:
            failure_report = FailureReport()
        for entry in self._failed().values():
            code = entry.get("code")
            for _ in range(int(entry.get("count", 1))):
                failure_report.by_code[str(code)] += 1
                bucket = entry.get("bucket")
                if bucket:
                    failure_report.by_bucket[bucket] += 1
        return failure_report
