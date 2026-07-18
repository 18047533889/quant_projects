"""Pipeline 多机分片与文件任务队列。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from runtime.shard_materialize import shard_by_hash

JOB_TYPE_CONFIG = "config"
JOB_TYPE_DATA_EVENT = "data_event"


def shard_config_paths(
    config_paths: Iterable[Path],
    *,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    """按 config 路径稳定哈希分片，供多机并行消费同一目录。"""
    return shard_by_hash(
        config_paths,
        shard_index=shard_index,
        shard_count=shard_count,
        key_fn=lambda p: str(Path(p).resolve()),
    )


@dataclass(frozen=True)
class QueueJob:
    """任务队列中的单条作业记录。"""

    job_id: str
    config_path: str
    payload: dict[str, Any]
    status: str
    created_at: str
    updated_at: str


class FileTaskQueue:
    """基于目录的文件任务队列（pending/running/done/failed）。"""

    def __init__(self, root: str | Path) -> None:
        """初始化队列目录（pending/running/done/failed 子目录）。"""
        self.root = Path(root)
        for name in ("pending", "running", "done", "failed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def enqueue(self, config_path: str | Path, **payload: Any) -> QueueJob:
        """将新任务写入 pending 目录并返回作业对象。"""
        now = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().hex
        job = QueueJob(
            job_id=job_id,
            config_path=str(config_path),
            payload=payload,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        path = self.root / "pending" / f"{job_id}.json"
        path.write_text(json.dumps(job.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return job

    def _read_job(self, path: Path) -> QueueJob:
        data = json.loads(path.read_text(encoding="utf-8"))
        return QueueJob(**data)

    def claim(self) -> QueueJob | None:
        """原子认领一条 pending 任务，移至 running 并返回。"""
        pending_dir = self.root / "pending"
        for path in sorted(pending_dir.glob("*.json")):
            running_path = self.root / "running" / path.name
            try:
                os.rename(path, running_path)
            except FileNotFoundError:
                continue
            job = self._read_job(running_path)
            updated = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=job.payload,
                status="running",
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            running_path.write_text(
                json.dumps(updated.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return updated
        return None

    def complete(self, job_id: str, *, result: dict[str, Any] | None = None) -> None:
        """将 running 任务标记为 done 并写入可选结果。"""
        self._finalize(job_id, bucket="done", result=result)

    def fail(self, job_id: str, *, error: str) -> None:
        """将 running 任务标记为 failed 并记录错误信息。"""
        self._finalize(job_id, bucket="failed", result={"error": error})

    def retry_or_fail(
        self,
        job_id: str,
        *,
        error: str,
        max_retries: int = 0,
    ) -> str:
        """失败时若未超重试上限则回到 pending，否则进入 failed。返回 ``requeued`` / ``failed``。"""
        running_path = self.root / "running" / f"{job_id}.json"
        if not running_path.exists():
            raise FileNotFoundError(f"running job not found: {job_id}")
        job = self._read_job(running_path)
        payload = dict(job.payload)
        attempts = int(payload.get("attempts", 0)) + 1
        payload["attempts"] = attempts
        payload["last_error"] = error
        if max_retries > 0 and attempts <= max_retries:
            restored = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=payload,
                status="pending",
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            pending_path = self.root / "pending" / f"{job_id}.json"
            pending_path.write_text(
                json.dumps(restored.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            running_path.unlink(missing_ok=True)
            return "requeued"
        payload["result"] = {"error": error, "attempts": attempts}
        final = QueueJob(
            job_id=job.job_id,
            config_path=job.config_path,
            payload=payload,
            status="failed",
            created_at=job.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        target = self.root / "failed" / f"{job_id}.json"
        target.write_text(json.dumps(final.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        running_path.unlink(missing_ok=True)
        return "failed"

    def _finalize(self, job_id: str, *, bucket: str, result: dict[str, Any] | None) -> None:
        running_path = self.root / "running" / f"{job_id}.json"
        if not running_path.exists():
            raise FileNotFoundError(f"running job not found: {job_id}")
        job = self._read_job(running_path)
        payload = dict(job.payload)
        if result is not None:
            payload["result"] = result
        final = QueueJob(
            job_id=job.job_id,
            config_path=job.config_path,
            payload=payload,
            status=bucket,
            created_at=job.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        target = self.root / bucket / f"{job_id}.json"
        target.write_text(json.dumps(final.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        running_path.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        """各状态任务数量。"""
        return {
            bucket: len(list((self.root / bucket).glob("*.json")))
            for bucket in ("pending", "running", "done", "failed")
        }

    def requeue_stale_running(self, *, max_age_seconds: float | None = None) -> int:
        """将 running 任务移回 pending（worker 崩溃恢复）。"""
        moved = 0
        now = datetime.now(timezone.utc)
        for path in sorted((self.root / "running").glob("*.json")):
            job = self._read_job(path)
            if max_age_seconds is not None:
                updated = _parse_ts(job.updated_at)
                if updated is not None:
                    age = (now - updated).total_seconds()
                    if age < max_age_seconds:
                        continue
            pending_path = self.root / "pending" / path.name
            restored = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=job.payload,
                status="pending",
                created_at=job.created_at,
                updated_at=now.isoformat(),
            )
            pending_path.write_text(
                json.dumps(restored.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            path.unlink(missing_ok=True)
            moved += 1
        return moved


class ObjectStoreTaskQueue:
    """对象存储任务队列（S3 等）；本地路径无 ``://`` 时委托 ``FileTaskQueue``。"""

    def __init__(self, root: str | Path) -> None:
        """本地路径委托 ``FileTaskQueue``；``s3://`` 等走 fsspec 对象存储。"""
        text = str(root).strip()
        if "://" not in text:
            self._delegate: FileTaskQueue | None = FileTaskQueue(text)
            self._fs = None
            self._base = ""
            return
        try:
            import fsspec
        except ImportError as exc:
            raise ImportError(
                "ObjectStoreTaskQueue 需要 fsspec：pip install fsspec s3fs"
            ) from exc
        protocol = text.split("://", 1)[0]
        self._delegate = None
        self._fs = fsspec.filesystem(protocol)
        self._base = text.split("://", 1)[1].rstrip("/")
        for bucket in ("pending", "running", "done", "failed"):
            path = self._path(bucket)
            if not self._fs.exists(path):
                self._fs.mkdirs(path, exist_ok=True)

    def _path(self, *parts: str) -> str:
        return "/".join(p for p in (self._base, *parts) if p)

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        assert self._fs is not None
        with self._fs.open(path, "w") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2))

    def _read_json(self, path: str) -> dict[str, Any]:
        assert self._fs is not None
        with self._fs.open(path, "r") as fh:
            return json.loads(fh.read())

    def enqueue(self, config_path: str | Path, **payload: Any) -> QueueJob:
        """将新任务写入 pending 并返回作业对象。"""
        if self._delegate is not None:
            return self._delegate.enqueue(config_path, **payload)
        now = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().hex
        job = QueueJob(
            job_id=job_id,
            config_path=str(config_path),
            payload=payload,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        self._write_json(self._path("pending", f"{job_id}.json"), job.__dict__)
        return job

    def claim(self) -> QueueJob | None:
        """原子认领一条 pending 任务，移至 running 并返回。"""
        if self._delegate is not None:
            return self._delegate.claim()
        assert self._fs is not None
        pending = sorted(self._fs.glob(self._path("pending", "*.json")))
        for path in pending:
            name = path.rsplit("/", 1)[-1]
            running = self._path("running", name)
            try:
                self._fs.mv(path, running)
            except Exception:
                continue
            data = self._read_json(running)
            job = QueueJob(**data)
            updated = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=job.payload,
                status="running",
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._write_json(running, updated.__dict__)
            return updated
        return None

    def complete(self, job_id: str, *, result: dict[str, Any] | None = None) -> None:
        """将 running 任务标记为 done 并写入可选结果。"""
        if self._delegate is not None:
            return self._delegate.complete(job_id, result=result)
        self._finalize(job_id, bucket="done", result=result)

    def fail(self, job_id: str, *, error: str) -> None:
        """将 running 任务标记为 failed 并记录错误信息。"""
        if self._delegate is not None:
            return self._delegate.fail(job_id, error=error)
        self._finalize(job_id, bucket="failed", result={"error": error})

    def _finalize(self, job_id: str, *, bucket: str, result: dict[str, Any] | None) -> None:
        assert self._fs is not None
        running = self._path("running", f"{job_id}.json")
        if not self._fs.exists(running):
            raise FileNotFoundError(f"running job not found: {job_id}")
        job = QueueJob(**self._read_json(running))
        payload = dict(job.payload)
        if result is not None:
            payload["result"] = result
        final = QueueJob(
            job_id=job.job_id,
            config_path=job.config_path,
            payload=payload,
            status=bucket,
            created_at=job.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        target = self._path(bucket, f"{job_id}.json")
        self._write_json(target, final.__dict__)
        self._fs.rm(running, missing_ok=True)

    def stats(self) -> dict[str, int]:
        """各状态任务数量。"""
        if self._delegate is not None:
            return self._delegate.stats()
        assert self._fs is not None
        return {
            bucket: len(self._fs.glob(self._path(bucket, "*.json")))
            for bucket in ("pending", "running", "done", "failed")
        }

    def requeue_stale_running(self, *, max_age_seconds: float | None = None) -> int:
        """将超时的 running 任务移回 pending（worker 崩溃恢复）。"""
        if self._delegate is not None:
            return self._delegate.requeue_stale_running(max_age_seconds=max_age_seconds)
        assert self._fs is not None
        moved = 0
        now = datetime.now(timezone.utc)
        for path in sorted(self._fs.glob(self._path("running", "*.json"))):
            job = QueueJob(**self._read_json(path))
            if max_age_seconds is not None:
                updated = _parse_ts(job.updated_at)
                if updated is not None:
                    age = (now - updated).total_seconds()
                    if age < max_age_seconds:
                        continue
            name = path.rsplit("/", 1)[-1]
            pending = self._path("pending", name)
            restored = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=job.payload,
                status="pending",
                created_at=job.created_at,
                updated_at=now.isoformat(),
            )
            self._write_json(pending, restored.__dict__)
            self._fs.rm(path, missing_ok=True)
            moved += 1
        return moved


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


class RedisTaskQueue:
    """Redis 任务队列 backend（Phase 10）；未安装 redis 时 raise ImportError。"""

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "factor_engine:queue",
        reset: bool = False,
    ) -> None:
        """连接 Redis；仅在显式 ``reset=True`` 时清理已有队列。"""
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisTaskQueue 需要 redis 包：pip install redis"
            ) from exc

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix.rstrip(":")
        try:
            self._redis.ping()
        except Exception as exc:
            raise ImportError(
                f"RedisTaskQueue backend unavailable at {redis_url}: {exc}"
            ) from exc
        if reset:
            for bucket in ("pending", "running", "done", "failed"):
                self._redis.delete(f"{self._prefix}:{bucket}")

    def _key(self, bucket: str) -> str:
        return f"{self._prefix}:{bucket}"

    def enqueue(self, config_path: str | Path, **payload: Any) -> QueueJob:
        """将新任务写入 Redis pending 哈希并返回作业对象。"""
        now = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().hex
        job = QueueJob(
            job_id=job_id,
            config_path=str(config_path),
            payload=payload,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        self._redis.hset(self._key("pending"), job_id, json.dumps(job.__dict__, ensure_ascii=False))
        return job

    def claim(self) -> QueueJob | None:
        """从 Redis pending 哈希中原子认领一条任务。"""
        pending_key = self._key("pending")
        for job_id, raw in self._redis.hgetall(pending_key).items():
            if self._redis.hdel(pending_key, job_id) == 0:
                continue
            data = json.loads(raw)
            job = QueueJob(**data)
            updated = QueueJob(
                job_id=job.job_id,
                config_path=job.config_path,
                payload=job.payload,
                status="running",
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._redis.hset(self._key("running"), job_id, json.dumps(updated.__dict__, ensure_ascii=False))
            return updated
        return None

    def complete(self, job_id: str, *, result: dict[str, Any] | None = None) -> None:
        """将 running 任务标记为 done 并写入可选结果。"""
        self._finalize(job_id, bucket="done", result=result)

    def fail(self, job_id: str, *, error: str) -> None:
        """将 running 任务标记为 failed 并记录错误信息。"""
        self._finalize(job_id, bucket="failed", result={"error": error})

    def _finalize(self, job_id: str, *, bucket: str, result: dict[str, Any] | None) -> None:
        running_key = self._key("running")
        raw = self._redis.hget(running_key, job_id)
        if raw is None:
            raise FileNotFoundError(f"running job not found: {job_id}")
        self._redis.hdel(running_key, job_id)
        data = json.loads(raw)
        job = QueueJob(**data)
        payload = dict(job.payload)
        if result is not None:
            payload["result"] = result
        final = QueueJob(
            job_id=job.job_id,
            config_path=job.config_path,
            payload=payload,
            status=bucket,
            created_at=job.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._redis.hset(self._key(bucket), job_id, json.dumps(final.__dict__, ensure_ascii=False))

    def stats(self) -> dict[str, int]:
        """各状态任务数量。"""
        return {
            bucket: self._redis.hlen(self._key(bucket))
            for bucket in ("pending", "running", "done", "failed")
        }


def build_task_queue(
    *,
    backend: str = "file",
    root: str | Path | None = None,
    redis_url: str | None = None,
) -> FileTaskQueue | RedisTaskQueue | ObjectStoreTaskQueue:
    """工厂：file / object_store（S3 等）/ redis 队列。"""
    normalized = backend.strip().lower()
    root_text = str(root).strip() if root is not None else ""
    if normalized in {"object_store", "s3"} or (root_text and "://" in root_text):
        if not root_text:
            raise ValueError("object_store backend 需要 root（如 s3://bucket/prefix/queue）")
        return ObjectStoreTaskQueue(root_text)
    if normalized == "redis":
        if not redis_url:
            raise ValueError("redis backend 需要 redis_url")
        return RedisTaskQueue(redis_url)
    if root is None:
        raise ValueError("file backend 需要 root")
    return FileTaskQueue(root)


def enqueue_config_directory(
    queue: FileTaskQueue | RedisTaskQueue | ObjectStoreTaskQueue,
    config_dir: str | Path,
    *,
    pattern: str = "*.yaml",
) -> list[QueueJob]:
    """将目录内 YAML config 批量入队。"""
    root = Path(config_dir)
    jobs: list[QueueJob] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            jobs.append(queue.enqueue(path))
    return jobs


def enqueue_data_event(
    queue: FileTaskQueue | RedisTaskQueue | ObjectStoreTaskQueue,
    *,
    dataset: str,
    column: str,
    updated_date: str,
    config_path: str | Path | None = None,
    **payload: Any,
) -> QueueJob:
    """将 ``DataEvent`` 入队供 worker 消费。"""
    ref = str(config_path) if config_path is not None else f"event:{dataset}:{column}"
    return queue.enqueue(
        ref,
        job_type=JOB_TYPE_DATA_EVENT,
        dataset=dataset,
        column=column,
        updated_date=updated_date,
        **payload,
    )
