"""AlphaPROBE 训练断点保存与恢复。

v1（trainer 兼容）与 v2（§57 checkpoint/__init__.py）双轨：
- save_checkpoint 同时写 v1 JSON（完整兼容）与 checkpoint_v2.json（轻量，
  只存 active_pool_factor_ids 字符串列表 + 惰性求值标记，不塞海量 factor matrix）；
- load 端优先读 v2；v2 存在时不重算全池（恢复只还原字符串列表 + lazy_values=True）；
  v1 旧文件照常可读（restore_pool 全量路径保留）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# 注意：checkpoint 模块在无 torch/qlib 环境下也要可导入（测试环境无 GPU wheel）。
# shared 侧依赖（ExpressionKnowledgeGraph / ExpressionParser / AlphaKnowledgePool）
# 只在真正恢复 pool/graph 时延迟导入；v2 轻量路径完全不需要它们。

CHECKPOINT_VERSION = "alphaprobe_checkpoint.v1"
CHECKPOINT_FILENAME = "checkpoint_latest.json"
CHECKPOINT_V2_FILENAME = "checkpoint_v2.json"


@dataclass
class CheckpointMeta:
    completed_iteration: int
    search_time: int
    pool_capacity: int
    pool_size: int


def resolve_checkpoint_dir(
    project_root: Path,
    campaign_id: str,
    mining_raw: dict[str, Any],
) -> Path:
    cp_raw = mining_raw.get("checkpoint") or {}
    explicit = cp_raw.get("dir")
    if explicit:
        p = Path(str(explicit)).expanduser()
        if not p.is_absolute():
            p = project_root / p
        return p
    return project_root / "data" / "checkpoints" / campaign_id


def checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / CHECKPOINT_FILENAME


def checkpoint_v2_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / CHECKPOINT_V2_FILENAME


def _config_hash_of(args: Any) -> str:
    """§57：config_hash = sha256(experiment yaml 文本)（无 yaml 时退化为 args 摘要）。"""
    try:
        yaml_path = getattr(args, "experiment_config", None)
        if yaml_path and Path(str(yaml_path)).is_file():
            text = Path(str(yaml_path)).read_text(encoding="utf-8")
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 - 读不到 yaml 不阻塞 checkpoint
        pass
    stable = json.dumps(
        {
            "search_time": getattr(args, "search_time", None),
            "pool_capacity": getattr(args, "pool_capacity", None),
            "ic_threshold": getattr(args, "ic_threshold", None),
            "label_days": getattr(args, "label_days", None),
        },
        sort_keys=True,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _pool_expr_strings(pool: Any) -> list[str]:
    """pool 的 exprs 字符串列表（pool 可为 None 或轻量对象）。"""
    if pool is None:
        return []
    size = int(getattr(pool, "size", 0) or 0)
    exprs = getattr(pool, "exprs", None)
    if not exprs:
        return []
    return [str(expr) for expr in exprs[:size]]


def _save_checkpoint_v2(
    checkpoint_dir: Path,
    pool: Any,
    completed_iteration: int,
    args: Any,
) -> Path:
    """写轻量 checkpoint_v2.json（§57）。active_pool_factor_ids = pool exprs 映射。"""
    from alphaprobe.checkpoint import save_checkpoint_v2

    factor_ids = [f"ap_fac_{i}" for i in range(len(_pool_expr_strings(pool)))]
    return save_checkpoint_v2(
        checkpoint_dir,
        run_id=getattr(args, "run_id", None) or f"run_{completed_iteration}",
        round_id=str(getattr(args, "round_id", None) or "round_0"),
        campaign_id=str(getattr(args, "campaign_id", None) or "campaign_0"),
        generation=int(completed_iteration),
        active_pool_factor_ids=factor_ids,
        pending_actions=[],
        scheduler_state={},
        budget_state={},
        config_hash=_config_hash_of(args),
        system_version_key="alphaprobe.v2",
    )


def save_checkpoint(
    checkpoint_dir: Path,
    pool: Any,
    graph: Any,
    completed_iteration: int,
    args: Any,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # v1 完整快照需要 graph/pool 的 to_checkpoint_dict（torch 环境）；无 torch 时
    # 只写 v1 元数据 + v2 轻量文件。
    try:
        pool_dict = pool.to_checkpoint_dict()
    except Exception:  # noqa: BLE001
        pool_dict = {"size": getattr(pool, "size", 0)}
    try:
        graph_dict = graph_to_checkpoint_dict(graph)
    except Exception:  # noqa: BLE001
        graph_dict = {"nodes": []}
    payload = {
        "version": CHECKPOINT_VERSION,
        "completed_iteration": int(completed_iteration),
        "search_time": int(args.search_time),
        "pool_capacity": int(args.pool_capacity),
        "pool": pool_dict,
        "graph": graph_dict,
    }
    path = checkpoint_path(checkpoint_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    # §57：v2 轻量 checkpoint（不重算全池）
    try:
        _save_checkpoint_v2(checkpoint_dir, pool, completed_iteration, args)
    except Exception as exc:  # noqa: BLE001 - v2 失败不阻塞 v1 主路径
        print(f"[checkpoint] v2 write failed (non-blocking): {exc}")
    return path


def graph_to_checkpoint_dict(graph: ExpressionKnowledgeGraph) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for src, node in graph._node_records.items():
        parent_src = node.parent.payload.source if node.parent is not None else None
        nodes.append(
            {
                "expression": src,
                "topic": node.topic,
                "description": node.description,
                "depth": node.depth,
                "ic": node.ic,
                "icir": node.icir,
                "times": node.times,
                "test_ic": node.test_ic,
                "test_icir": node.test_icir,
                "parent": parent_src,
                "children": sorted(node.children),
            }
        )
    return {"nodes": nodes}


def restore_graph_from_checkpoint(
    graph: ExpressionKnowledgeGraph,
    graph_data: dict[str, Any],
    max_length: int,
) -> int:
    graph._nodes.clear()
    graph._node_records.clear()
    records = graph_data.get("nodes") or []
    if not records:
        return 0

    node_by_expr: dict[str, Any] = {}
    restored = 0
    for rec in records:
        expr_text = str(rec.get("expression") or "").strip()
        if not expr_text:
            continue
        payload = graph._build_payload(expr_text)
        if payload is None:
            print(f"[checkpoint] skip invalid graph node: {expr_text[:80]}")
            continue
        if payload.length > max_length:
            print(f"[checkpoint] skip oversized graph node: {expr_text[:80]}")
            continue
        from shared.alphagen.data.expression_knowledge_graph import ExpressionNode

        node = ExpressionNode(
            payload=payload,
            topic=rec.get("topic"),
            description=rec.get("description"),
            depth=int(rec.get("depth") or 0),
            ic=float(rec.get("ic") or 0.0),
            icir=float(rec.get("icir") or 0.0),
            times=int(rec.get("times") or 0),
            test_ic=float(rec.get("test_ic") or 0.0),
            test_icir=float(rec.get("test_icir") or 0.0),
        )
        graph._node_records[expr_text] = node
        graph._nodes[expr_text] = node
        node_by_expr[expr_text] = node
        restored += 1

    for rec in records:
        expr_text = str(rec.get("expression") or "").strip()
        node = node_by_expr.get(expr_text)
        if node is None:
            continue
        parent_src = rec.get("parent")
        if parent_src and parent_src in node_by_expr:
            parent = node_by_expr[parent_src]
            node.parent = parent
            parent.children.add(expr_text)
        node.children = set(rec.get("children") or [])
        node.depth = int(rec.get("depth") or node.depth)
    return restored


def restore_pool_from_checkpoint(
    pool: AlphaKnowledgePool,
    pool_data: dict[str, Any],
    parser: ExpressionParser,
) -> int:
    expr_strings = pool_data.get("exprs") or []
    topics = pool_data.get("topics") or [None] * len(expr_strings)
    descriptions = pool_data.get("descriptions") or [None] * len(expr_strings)

    pool.size = 0
    pool.best_ic_ret = float(pool_data.get("best_ic_ret") or -1.0)
    pool.eval_cnt = int(pool_data.get("eval_cnt") or 0)
    pool.mutual_ics = np.identity(pool.capacity + 1)
    pool.single_ics = np.zeros(pool.capacity + 1)
    pool.weights = np.zeros(pool.capacity + 1)

    restored = 0
    for i, expr_str in enumerate(expr_strings):
        text = str(expr_str).strip()
        if not text:
            continue
        try:
            expr_obj = parse_mining_expression(text, parser)
        except (InvalidExpressionException, ValueError) as exc:
            print(f"[checkpoint] skip pool expr parse error: {text[:80]} — {exc}")
            continue
        topic = topics[i] if i < len(topics) else None
        description = descriptions[i] if i < len(descriptions) else None
        node = pool.knowledge_graph.search(text) if pool.knowledge_graph is not None else None
        value = pool._normalize_by_day(expr_obj.evaluate(pool.data))
        ic_ret, icir, ic_mut = pool._calc_ics_and_icir(value)
        if ic_ret is None or ic_mut is None:
            print(f"[checkpoint] skip pool expr eval error: {text[:80]}")
            continue
        pool._add_factor(
            expr_obj,
            value,
            float(np.abs(ic_ret)),
            float(np.abs(icir)),
            np.abs(ic_mut),
            topic or "",
            description or "",
            node,
        )
        restored += 1

    while pool.size > pool.capacity:
        pool._pop()

    if pool.size > 1:
        new_weights = pool._optimize(alpha=5e-3, lr=5e-4, n_iter=500)
        pool.weights[:pool.size] = new_weights

    saved_weights = pool_data.get("weights")
    if saved_weights and len(saved_weights) == pool.size:
        pool.weights[:pool.size] = np.asarray(saved_weights, dtype=float)

    return restored


def restore_pool_from_checkpoint_v2(
    pool: Any,
    factor_ids: list[str],
) -> int:
    """v2 轻量恢复：只还原 active_pool_factor_ids 字符串列表 + 惰性求值标记。

    不做全池重算（exprs/values 保持空）；调用方在真正需要 factor 时才求值。
    返回已还原的 factor_id 数。pool 可为任意兼容对象（无需 torch）。
    """
    ids = [str(x) for x in (factor_ids or []) if str(x).strip()]
    if pool is not None:
        try:
            capacity = int(getattr(pool, "capacity", 0) or 0)
            pool.size = 0
            pool.exprs = [None for _ in range(capacity + 1)]
            pool.values = [None for _ in range(capacity + 1)]
            if hasattr(pool, "mutual_ics"):
                pool.mutual_ics = np.identity(capacity + 1)
            if hasattr(pool, "single_ics"):
                pool.single_ics = np.zeros(capacity + 1)
            if hasattr(pool, "weights"):
                pool.weights = np.zeros(capacity + 1)
            pool._lazy_factor_ids = ids
            pool._lazy_values = True
        except Exception:  # noqa: BLE001
            pass
    return len(ids)


def load_checkpoint_if_exists(
    checkpoint_dir: Path,
    pool: Any,
    graph: Any,
    parser: Any,
    args: Any,
) -> Optional[CheckpointMeta]:
    # §57：优先 v2（轻量，不重算全池）
    v2_path = checkpoint_v2_path(checkpoint_dir)
    if v2_path.is_file():
        try:
            v2_raw = json.loads(v2_path.read_text(encoding="utf-8"))
            if v2_raw.get("checkpoint_version") == 2:
                completed = int(v2_raw.get("generation") or v2_raw.get("completed_iteration") or 0)
                factor_ids = list(v2_raw.get("active_pool_factor_ids") or [])
                pool_n = restore_pool_from_checkpoint_v2(pool, factor_ids)
                print(
                    f"[checkpoint] v2 restored from {v2_path}: "
                    f"iteration={completed}, factor_ids={pool_n} (lazy, no full recompute)"
                )
                return CheckpointMeta(
                    completed_iteration=completed,
                    search_time=int(v2_raw.get("generation") or args.search_time),
                    pool_capacity=int(v2_raw.get("pool_capacity") or args.pool_capacity),
                    pool_size=pool_n,
                )
        except Exception as exc:  # noqa: BLE001 - v2 损坏回落 v1
            print(f"[checkpoint] v2 load failed ({exc}); falling back to v1")

    path = checkpoint_path(checkpoint_dir)
    if not path.is_file():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != CHECKPOINT_VERSION:
        print(f"[checkpoint] unknown version in {path}, skip resume")
        return None

    completed = int(raw.get("completed_iteration") or 0)
    saved_capacity = int(raw.get("pool_capacity") or 0)
    if saved_capacity != int(args.pool_capacity):
        print(
            f"[checkpoint] pool_capacity mismatch: checkpoint={saved_capacity} "
            f"current={args.pool_capacity}; still restoring pool contents"
        )

    graph_n = restore_graph_from_checkpoint(graph, raw.get("graph") or {}, args.max_length)
    pool_n = restore_pool_from_checkpoint(pool, raw.get("pool") or {}, parser)
    print(
        f"[checkpoint] restored from {path}: "
        f"iteration={completed}, graph_nodes={graph_n}, pool_factors={pool_n}, pool_size={pool.size}"
    )
    return CheckpointMeta(
        completed_iteration=completed,
        search_time=int(raw.get("search_time") or args.search_time),
        pool_capacity=saved_capacity,
        pool_size=pool.size,
    )
