#!/usr/bin/env python3
"""汇总 production 运行中的 Polars→Pandas fallback 记录。

用法::

    # 从 engine run() 返回的 dict 读取
    python scripts/report_pandas_fallbacks.py --json result.json

    # 从 stdin 读 JSON（批跑聚合）
    cat batch_results.json | python scripts/report_pandas_fallbacks.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _extract_fallbacks(payload: object) -> list[dict[str, str]]:
    """从 engine 结果 JSON（单条或批跑嵌套结构）递归提取 production_pandas_fallbacks。"""
    if isinstance(payload, dict):
        if "production_pandas_fallbacks" in payload:
            raw = payload["production_pandas_fallbacks"]
            if isinstance(raw, list):
                return [dict(x) for x in raw if isinstance(x, dict)]
        if "results" in payload and isinstance(payload["results"], list):
            out: list[dict[str, str]] = []
            for item in payload["results"]:
                out.extend(_extract_fallbacks(item))
            return out
        if "runtime_stats" in payload and isinstance(payload["runtime_stats"], dict):
            raw = payload["runtime_stats"].get("production_pandas_fallbacks") or []
            return [dict(x) for x in raw if isinstance(x, dict)]
    if isinstance(payload, list):
        out: list[dict[str, str]] = []
        for item in payload:
            out.extend(_extract_fallbacks(item))
        return out
    return []


def _summarize(fallbacks: list[dict[str, str]]) -> dict[str, object]:
    """按算子名聚合 fallback 事件计数。"""
    by_op = Counter(str(x.get("op", "?")) for x in fallbacks)
    return {
        "total_events": len(fallbacks),
        "unique_ops": len(by_op),
        "by_op": dict(sorted(by_op.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def main() -> int:
    """汇总 production 运行中的 Polars→Pandas fallback 记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="单条或批跑 engine 结果 JSON")
    parser.add_argument("--fail-if-any", action="store_true", help="存在 fallback 时 exit 1")
    args = parser.parse_args()

    if args.json:
        payload = json.loads(args.json.read_text(encoding="utf-8"))
    else:
        payload = json.load(sys.stdin)

    fallbacks = _extract_fallbacks(payload)
    summary = _summarize(fallbacks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_if_any and summary["total_events"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
