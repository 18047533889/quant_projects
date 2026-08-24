"""AlphaPROBE 训练断点保存与恢复。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from shared.alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph
from shared.alphagen.data.tree import ExpressionParser, InvalidExpressionException
from alphaprobe.fe_bridge.expr_parse import parse_mining_expression
from alphaprobe.trainer.pool import AlphaKnowledgePool

CHECKPOINT_VERSION = "alphaprobe_checkpoint.v1"
CHECKPOINT_FILENAME = "checkpoint_latest.json"


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


def save_checkpoint(
    checkpoint_dir: Path,
    pool: AlphaKnowledgePool,
    graph: ExpressionKnowledgeGraph,
    completed_iteration: int,
    args: Any,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "completed_iteration": int(completed_iteration),
        "search_time": int(args.search_time),
        "pool_capacity": int(args.pool_capacity),
        "pool": pool.to_checkpoint_dict(),
        "graph": graph_to_checkpoint_dict(graph),
    }
    path = checkpoint_path(checkpoint_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
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


def load_checkpoint_if_exists(
    checkpoint_dir: Path,
    pool: AlphaKnowledgePool,
    graph: ExpressionKnowledgeGraph,
    parser: ExpressionParser,
    args: Any,
) -> Optional[CheckpointMeta]:
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
