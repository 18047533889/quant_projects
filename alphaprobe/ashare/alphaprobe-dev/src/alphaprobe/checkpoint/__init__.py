"""checkpoint v2（任务书 §57 / §57.1 / §89.6）。

不把海量 factor matrix 塞 checkpoint；恢复不重算全池。
v1 migration：old expression → FE canonical；丢弃/隔离 test metrics，不进搜索记忆。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = 2


def save_checkpoint_v2(
    checkpoint_dir: str | os.PathLike,
    *,
    run_id: str,
    round_id: str,
    campaign_id: str,
    generation: int,
    active_pool_factor_ids: list[str],
    pending_actions: list[dict[str, Any]],
    scheduler_state: dict[str, Any],
    budget_state: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
    calibrator_version: str = "",
    knowledge_cutoff: str | None = None,
    survival_cutoff: str | None = None,
    global_memory_snapshot_id: str = "",
    evaluation_cache_snapshot_id: str = "",
    config_hash: str = "",
    system_version_key: str = "",
) -> Path:
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": run_id,
        "round_id": round_id,
        "campaign_id": campaign_id,
        "generation": generation,
        "active_pool_factor_ids": list(active_pool_factor_ids),
        "pending_actions": list(pending_actions),
        "scheduler_state": scheduler_state,
        "budget_state": budget_state,
        "rng_state": rng_state or {},
        "calibrator_version": calibrator_version,
        "knowledge_cutoff": knowledge_cutoff,
        "survival_cutoff": survival_cutoff,
        "global_memory_snapshot_id": global_memory_snapshot_id,
        "evaluation_cache_snapshot_id": evaluation_cache_snapshot_id,
        "config_hash": config_hash,
        "system_version_key": system_version_key,
    }
    path = d / "checkpoint_v2.json"
    tmp = d / "checkpoint_v2.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def load_checkpoint_v2(checkpoint_dir: str | os.PathLike) -> dict[str, Any] | None:
    path = Path(checkpoint_dir) / "checkpoint_v2.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# §57.1 v1 migration
# ---------------------------------------------------------------------------


def migrate_v1(
    v1_checkpoint_path: str | os.PathLike,
    *,
    canonicalize_fn: Any | None = None,
    output_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """old node → FactorNode fields；single parent → one-element list；
    **test metrics 丢弃/隔离，不进入搜索记忆**。

    canonicalize_fn(formula) -> (canonical, ast_hash, signal_id, family_id)。
    """
    src = Path(v1_checkpoint_path)
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("graph", {}).get("nodes", [])
    migrated_nodes = []
    for node in nodes:
        formula = node.get("formula") or node.get("payload", {}).get("source", "")
        if canonicalize_fn is not None:
            canonical, ast_hash, signal_id, family_id = canonicalize_fn(formula)
        else:
            canonical, ast_hash, signal_id, family_id = formula, "", "", None
        parent = node.get("parent")
        migrated_nodes.append(
            {
                "factor_id": node.get("key") or node.get("id") or canonical,
                "canonical_formula": canonical,
                "canonical_ast_hash": ast_hash,
                "signal_equivalence_id": signal_id or ast_hash,
                "parameter_family_id": family_id,
                "parent_ids": [parent] if parent else [],  # single → list
                "topic": node.get("topic"),
                "description": node.get("description"),
                # 旧单值指标：保留 train/valid 侧 raw 值，test 一律隔离
                "legacy_ic": node.get("ic"),
                "legacy_icir": node.get("icir"),
            }
        )
    # v1 里 node.test_ic / test_icir 不迁移：隔离为 sealed 历史附件
    out = {
        "checkpoint_version": 2,
        "migrated_from": str(src),
        "active_pool_factor_ids": [n["factor_id"] for n in migrated_nodes],
        "migrated_nodes": migrated_nodes,
        "test_metrics_isolated": True,  # §57.1：test metrics 不进入搜索记忆
    }
    dst = Path(output_path) if output_path else src.parent / "checkpoint_v2_migrated.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out