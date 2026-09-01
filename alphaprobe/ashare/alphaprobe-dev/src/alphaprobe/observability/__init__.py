"""observability（任务书 §72-§74）：run manifest + 事件流 + 成本账本。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    """§72：一个 run 保存完整可复现信息。"""

    run_id: str
    round_id: str
    campaign_id: str
    git_commit: str = "unknown"
    factor_engine_build: str = "unknown"
    data_access_build: str = "unknown"
    evaluator_build: str = "unknown"
    refinement_build: str = "unavailable"
    consolidation_build: str = "unavailable"
    data_snapshot_id: str = "auto"
    universe_snapshot_id: str = "auto"
    label_spec: str = ""
    split_spec: str = ""
    pit_contract: str = "pubdate_asof"
    prompt_version: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    seed: int = 0
    knowledge_cutoff: str | None = None
    survival_cutoff: str | None = None
    search_config_hash: str = ""
    fitness_config_hash: str = ""
    created_at: float = field(default_factory=time.time)


def collect_git_commit(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_run_manifest(out_dir: str | os.PathLike, manifest: RunManifest) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "run_manifest.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=1)
    return p


@dataclass
class CostLedger:
    """§30：记录真实 prompt/completion tokens、latency、model、action、factor_id。"""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        action: str,
        factor_id: str | None,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        estimated_cost: float = 0.0,
    ) -> None:
        self.entries.append(
            {
                "action": action,
                "factor_id": factor_id,
                "model": model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "latency_ms": int(latency_ms),
                "estimated_cost": float(estimated_cost),
                "ts": time.time(),
            }
        )

    def totals(self) -> dict[str, float]:
        return {
            "calls": len(self.entries),
            "prompt_tokens": sum(e["prompt_tokens"] for e in self.entries),
            "completion_tokens": sum(e["completion_tokens"] for e in self.entries),
            "estimated_cost": sum(e["estimated_cost"] for e in self.entries),
        }

    def write(self, out_dir: str | os.PathLike) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "cost_ledger.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"entries": self.entries, "totals": self.totals()}, f, indent=1)
        return p


class ObservabilityCounters:
    """§74 运行面板统计。"""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    def write(self, out_dir: str | os.PathLike) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "budget_stats.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=1)
        return p


def write_events_jsonl(out_dir: str | os.PathLike, events: list[dict[str, Any]], name: str = "attempts") -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    return p