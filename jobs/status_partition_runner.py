#!/usr/bin/env python3
"""Recoverable, partition-by-partition entry point for StockDailyBar.

This module owns orchestration only. Feature computation and source validation
remain in :mod:`ashare_feature_pipeline` so the runner can change without
changing feature definitions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq

from ashare_feature_pipeline import (
    BUCKET,
    SOURCE_ROOT,
    build_daily_features,
    load_registry,
    read_table,
    validate_daily,
)

COS_EXECUTABLE = "/home/sunhaiwei/quantsociety/bin/cos-api"
STATE_VERSION = 1
TRANSIENT_COS_CODES = ("400", "524")


def _error_text(error: BaseException) -> str:
    parts = [str(error)]
    for name in ("stdout", "stderr"):
        value = getattr(error, name, None)
        if value:
            parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def is_transient_cos_error(error: BaseException) -> bool:
    """Return whether a COS command failure contains a retryable HTTP code."""
    text = _error_text(error)
    return any(
        re.search(rf"(?<!\d){re.escape(code)}(?!\d)", text)
        for code in TRANSIENT_COS_CODES
    )


def cos_cli(
    *args: str,
    capture: bool = True,
    executable: str = COS_EXECUTABLE,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    wait_seconds: float = 120.0,
    max_attempts: int = 3,
) -> str:
    """Run ``cos-api`` and retry transient 400/524 failures.

    ``max_attempts`` counts the initial request, so the default is at most
    three requests total. Non-transient command failures are raised directly.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    command = [executable, *args]
    for attempt in range(1, max_attempts + 1):
        try:
            result = run(command, check=True, text=True, capture_output=capture)
            return result.stdout if capture else ""
        except subprocess.CalledProcessError as error:
            if attempt == max_attempts or not is_transient_cos_error(error):
                raise
            sleep(wait_seconds)
    raise AssertionError("unreachable")


def cos_cp(source: str, destination: Path | str, **kwargs: Any) -> None:
    """Copy one object through the retrying COS wrapper.

    Upload direction: the local ``source`` is a path and ``destination`` is a
    ``cos://`` URI, so neither end is converted to a local :class:`Path`.
    """
    destination_text = str(destination)
    if destination_text.startswith("cos://"):
        cos_cli("cp", source, destination_text, capture=False, **kwargs)
        return
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cos_cli("cp", source, str(destination), capture=False, **kwargs)


def cos_ls(prefix: str, **kwargs: Any) -> str:
    """List a COS prefix through the retrying COS wrapper."""
    return cos_cli("ls", prefix, "--recursive", **kwargs)


def list_partition_keys(
    dataset: str,
    start: dt.date,
    end: dt.date,
    **retry_kwargs: Any,
) -> list[str]:
    """Return date-suffixed parquet keys in the requested range."""
    output = cos_ls(f"{SOURCE_ROOT}/{dataset}/", **retry_kwargs)
    keys = []
    for line in output.splitlines():
        match = re.search(r"(\S+/)(\d{4}-\d{2}-\d{2})\.parquet(?:\s|$)", line)
        if not match:
            continue
        date = dt.date.fromisoformat(match.group(2))
        if start <= date <= end:
            keys.append(match.group(0).split()[0])
    return sorted(set(keys))


def _memory_snapshot() -> dict[str, int]:
    process = psutil.Process()
    virtual = psutil.virtual_memory()
    return {
        "rss_bytes": int(process.memory_info().rss),
        "available_bytes": int(virtual.available),
        "total_bytes": int(virtual.total),
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


@dataclass(frozen=True)
class PartitionContext:
    key: str
    date: dt.date | None
    input_path: Path
    output_dir: Path
    workdir: Path
    partition_index: int


PartitionProcessor = Callable[[PartitionContext], Mapping[str, Any] | None]


class PartitionRunner:
    """Execute independent partitions with an atomic, resumable checkpoint."""

    def __init__(
        self,
        workdir: Path,
        *,
        run_id: str | None = None,
        dataset: str = "StockDailyBar",
        resume: bool = False,
        fail_fast: bool = False,
    ) -> None:
        self.workdir = Path(workdir)
        self.state_path = self.workdir / "manifests" / "partition_status.json"
        self.run_id = run_id or self.workdir.name
        self.dataset = dataset
        self.resume = resume
        self.fail_fast = fail_fast
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "inputs").mkdir(exist_ok=True)
        (self.workdir / "outputs").mkdir(exist_ok=True)
        (self.workdir / "manifests").mkdir(exist_ok=True)
        self.state = self._load_state() if resume else self._new_state()

    def _new_state(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "state_version": STATE_VERSION,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "started_at": now,
            "updated_at": now,
            "partitions": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._new_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("state_version") != STATE_VERSION:
            raise ValueError(
                f"unsupported partition state version: {state.get('state_version')!r}"
            )
        state.setdefault("partitions", {})
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                self.state,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    @staticmethod
    def _partition_date(key: str) -> dt.date | None:
        match = re.search(r"/(\d{4}-\d{2}-\d{2})\.parquet$", key)
        return dt.date.fromisoformat(match.group(1)) if match else None

    def run(
        self,
        partition_keys: Iterable[str],
        processor: PartitionProcessor,
        *,
        download: Callable[[str, Path], None] | None = None,
    ) -> dict[str, Any]:
        keys = list(dict.fromkeys(partition_keys))
        download = download or (
            lambda key, path: cos_cp(f"cos://{BUCKET}/{key}", path)
        )
        self.state["partition_count"] = len(keys)
        self._save_state()

        for index, key in enumerate(keys):
            previous = self.state["partitions"].get(key, {})
            if previous.get("status") == "completed":
                continue
            entry = dict(previous)
            entry.update(
                {
                    "status": "running",
                    "attempts": int(previous.get("attempts", 0)) + 1,
                    "started_at": _utc_now(),
                    "failure_reason": None,
                    "memory_before": _memory_snapshot(),
                }
            )
            self.state["partitions"][key] = entry
            self._save_state()
            local = self.workdir / "inputs" / Path(key).name
            context = PartitionContext(
                key,
                self._partition_date(key),
                local,
                self.workdir / "outputs",
                self.workdir,
                index,
            )
            try:
                if not local.exists():
                    download(key, local)
                result = dict(processor(context) or {})
                entry.update(
                    {
                        "status": "completed",
                        "finished_at": _utc_now(),
                        "result": result,
                        "memory_after": _memory_snapshot(),
                    }
                )
            except Exception as error:  # checkpoint before moving on
                entry.update(
                    {
                        "status": "failed",
                        "finished_at": _utc_now(),
                        "failure_reason": f"{type(error).__name__}: {_error_text(error)}",
                        "traceback": traceback.format_exc(limit=12),
                        "memory_after": _memory_snapshot(),
                    }
                )
                self.state["partitions"][key] = entry
                self._save_state()
                if self.fail_fast:
                    raise
                continue
            self.state["partitions"][key] = entry
            self._save_state()

        completed = sum(
            item.get("status") == "completed"
            for item in self.state["partitions"].values()
        )
        failed = sum(
            item.get("status") == "failed"
            for item in self.state["partitions"].values()
        )
        summary = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "partition_count": len(keys),
            "completed": completed,
            "failed": failed,
            "state_path": str(self.state_path),
            "status": "completed" if failed == 0 else "completed_with_failures",
        }
        self.state["summary"] = summary
        self._save_state()
        return summary


def _write_partition_features(
    context: PartitionContext, registry: list[Any]
) -> Mapping[str, Any]:
    table = read_table(context.input_path)
    validation = validate_daily(table)
    features, quality = build_daily_features(table, registry)
    suffix = context.date.isoformat() if context.date else f"index-{context.partition_index:05d}"
    output = context.output_dir / f"part-{suffix}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(features, output, compression="zstd", use_dictionary=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "input_rows": table.num_rows,
        "validation": validation,
        "quality": quality,
        "output": {
            "local_path": str(output),
            "rows": features.num_rows,
            "sha256": digest,
            "bytes": output.stat().st_size,
        },
    }


def run_stock_daily_bar(
    start: dt.date,
    end: dt.date,
    workdir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run StockDailyBar without building a whole-history Arrow table."""
    keys = list_partition_keys("StockDailyBar", start, end)
    if not keys:
        raise ValueError("no StockDailyBar partitions in requested range")
    registry = load_registry()
    runner = PartitionRunner(workdir, dataset="StockDailyBar", resume=resume)
    return runner.run(keys, lambda context: _write_partition_features(context, registry))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_stock_daily_bar(args.start, args.end, args.workdir, resume=args.resume),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
