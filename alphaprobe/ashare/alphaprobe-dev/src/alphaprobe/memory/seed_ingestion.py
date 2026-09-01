"""冷启动 seed 摄取（任务书 §35 / §52）。

把 cold_start_library yaml 里的公式（无迭代过程，纯 FORMULA_ONLY）灌进
GlobalMemoryStore 作为 lineage_root 种子。兼容两种数据来源：

- ``cold_start_library.load_cold_start_yaml``（项目内包可用时，权威路径）
- 否则直接用 pyyaml 读 ``entries[].{expression,formula}`` + topic + explanation

关键约束（§35 / 冷启动现状）：
- yaml 不存在 / 目录为空 / 没有可解析条目是**已知状态** → 返回 0 并打印
  提示，绝不抛异常、绝不阻塞后续挖掘。
- identity_fn 用 alphaprobe.dedup（canonical / sign id / family）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from alphaprobe.memory import GlobalMemoryStore


def _identity_default(
    formula: str,
) -> tuple[str, str, str, str | None]:
    """默认 identity：raw 兜底（caller 未提供 alphaprobe.dedup identity 时）。"""
    from alphaprobe.dedup import canonical_ast_hash, canonicalize_dsl, signal_equivalence_id

    canonical = canonicalize_dsl(formula)
    return (
        canonical,
        canonical_ast_hash(formula),
        signal_equivalence_id(formula),
        None,
    )


def _try_load_with_cold_start_library(
    yaml_path: Path,
) -> list[tuple[str, str, str]] | None:
    """尝试用 cold_start_library.load_cold_start_yaml 读；不可用/无法解析返回 None。"""
    try:
        from alphaprobe.cold_start import load_cold_start_yaml
    except Exception:
        return None
    try:
        entries = load_cold_start_yaml(yaml_path)
    except Exception:
        return None
    if not entries:
        # 文件存在但无条目：回落 pyyaml 宽松解析（兼容非标准 keys）
        return None
    out: list[tuple[str, str, str]] = []
    for e in entries:
        expr = str(getattr(e, "expr", "") or "").strip()
        if not expr:
            continue
        topic = str(getattr(e, "topic", "") or "").strip()
        desc = str(getattr(e, "description", "") or "").strip()
        out.append((expr, topic, desc))
    return out


def _try_load_with_pyyaml(
    yaml_path: Path,
) -> list[tuple[str, str, str]] | None:
    """宽松解析：yaml 顶层 entries 列表，keys 兼容 expression/formula + topic + explanation。"""
    try:
        import yaml
    except Exception:
        return None
    try:
        raw = yaml.safe_load(yaml_path.expanduser().read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    entries_raw = raw.get("entries") or raw.get("factors") or raw.get("library") or []
    if not isinstance(entries_raw, list):
        return []
    out: list[tuple[str, str, str]] = []
    for item in entries_raw:
        # 宽松解析：没有 expression/formula 字段的条目
        if isinstance(item, dict):
            formula = str(item.get("expression") or item.get("formula") or "").strip()
            topic = str(item.get("topic") or "").strip()
            explanation = str(item.get("explanation") or "").strip()
        else:
            formula, topic, explanation = "", "", ""
        if not formula:
            continue
        out.append((formula, topic, explanation))
    return out


def ingest_cold_start_yaml(
    yaml_path: str | Path,
    store: GlobalMemoryStore,
    identity_fn: Callable[[str], tuple[str, str, str, str | None]] | None = None,
    *,
    source_snapshot: str = "cold_start_v9",
) -> int:
    """把冷启动 yaml 条目灌入 GlobalMemoryStore。返回实际新建的 seed 数。

    兼容行为（§35 / REFACTOR_STATE）：yaml 不存在、目录为空、或解析不出
    条目 → 返回 0 并打印提示，绝不抛异常、绝不阻塞。
    """
    path = Path(yaml_path).expanduser()
    if not path.is_file():
        print(
            f"[memory.seed_ingestion] cold-start yaml 不存在: {path} — "
            "跳过 seed 摄取（已知状态，不阻塞）"
        )
        return 0

    entries = _try_load_with_cold_start_library(path)
    if entries is not None and not entries:
        # cold_start_library 读到空条目 → 用 pyyaml 宽松解析兜底（兼容
        # 非 expr 字段的条目）；pyyaml 也解析不出才算真正空。
        entries = _try_load_with_pyyaml(path)
    if entries is None:
        entries = _try_load_with_pyyaml(path)
    if entries is None:
        print(
            f"[memory.seed_ingestion] 无法解析 {path}（缺 pyyaml）— 跳过，不阻塞"
        )
        return 0
    if not entries:
        print(
            f"[memory.seed_ingestion] {path} 无可用条目（目录为空/空文件）— "
            "摄取 0 条，不阻塞"
        )
        return 0

    if identity_fn is None:
        identity_fn = _identity_default
    created = store.ingest_seed_library(
        iter(entries),
        source_snapshot=source_snapshot,
        identity_fn=identity_fn,
    )
    print(
        f"[memory.seed_ingestion] ingested {created}/{len(entries)} seeds "
        f"from {path} (snapshot={source_snapshot})"
    )
    return created
