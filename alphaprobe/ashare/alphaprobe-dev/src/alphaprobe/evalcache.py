"""Evaluation Cache（任务书 §12）。

cache key = canonical_formula_hash + orientation + data_snapshot_id +
universe_snapshot_id + FE version + operator semantics version + label spec hash
+ segment + fidelity + evaluator version + preprocess profile version。
任何一项变化必须 cache miss。
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


def build_cache_key(
    *,
    canonical_formula_hash: str,
    orientation: int,
    data_snapshot_id: str,
    universe_snapshot_id: str,
    factor_engine_version: str,
    operator_semantics_version: str,
    label_spec_hash: str,
    segment: str,
    fidelity: str,
    evaluator_version: str,
    preprocess_profile_version: str = "none",
) -> str:
    payload = json.dumps(
        {
            "cfh": canonical_formula_hash,
            "ori": orientation,
            "data": data_snapshot_id,
            "uni": universe_snapshot_id,
            "fe": factor_engine_version,
            "ops": operator_semantics_version,
            "label": label_spec_hash,
            "seg": segment,
            "fid": fidelity,
            "ev": evaluator_version,
            "pp": preprocess_profile_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvaluationCache:
    """§7 Candidate Once Computed, Many Consumers。

    线程安全；缺省进程内 dict。可挂 disk backend（jsonl/parquet）后续替换。
    """

    def __init__(self, max_entries: int = 200_000) -> None:
        self._store: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._store:
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: str, record: Any) -> None:
        with self._lock:
            if len(self._store) >= self._max:
                # 简单溢出策略：丢最早插入的一半（7×24 下由 disk backend 接管）
                for k in list(self._store)[: self._max // 2]:
                    del self._store[k]
            self._store[key] = record

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses)
                    else 0.0
                ),
            }