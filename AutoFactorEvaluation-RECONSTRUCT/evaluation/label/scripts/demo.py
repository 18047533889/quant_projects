"""使用 mock 数据库路径运行 evaluation.label 演示。

从仓库根目录运行:
    uv run python -m evaluation.label.demo

演示流程使用与正式环境相同的入口函数。它注入一个确定性的语义客户端，
因此无需外部 API 凭证即可运行命令。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schemas import JsonDict, default_mock_label_module_config
from .service import run_label_pipeline

# 不使用真实api测试启用以下部分
# class DemoDeepSeekClient:
#     """仅用于本地 mock 数据校验的确定性语义客户端。"""

    # def request_semantic_tags(
    #     self,
    #     expression: str,
    #     charts_summary: JsonDict,
    #     label_registry_excerpt: JsonDict,
    # ) -> JsonDict:
    #     """返回一个已有标签，避免演示流程修改标签词表。"""

    #     return {
    #         "primary_label": "proxy_momentum",
    #         "confidence": 0.82,
    #         "explanation": "Mock demo selects the existing proxy_momentum label.",
    #         "suggested_new_label": "",
    #         "similar_candidates": [],
    #     }


def main(argv: list[str] | None = None) -> int:
    """运行一次 mock 单因子贴标流水线。"""

    parser = argparse.ArgumentParser(description="Run evaluation.label demo.")
    parser.add_argument("--project-root", default=".", help="Repository root path.")
    parser.add_argument(
        "--mock-root",
        default="database/mock",
        help="Mock database root with tier/cache/log structure.",
    )
    parser.add_argument(
        "--factor-id",
        default="alpha_101_001",
        help="Factor folder name under mock tier1 purification base.",
    )
    parser.add_argument(
        "--evaluation-summary",
        default="database/mock/evaluation/label/evaluation_summary/alpha_101_001.json",
        help="Mock upstream evaluation summary JSON path.",
    )
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    config = default_mock_label_module_config(project_root, args.mock_root)
    source_factor_dir = config.tier1_pure_factor_base_dir / args.factor_id

    result = run_label_pipeline(
        source_factor_dir=source_factor_dir,
        evaluation_summary_path=project_root / args.evaluation_summary,
        config=config,
        operator="evaluation.label.demo",
        # deepseek_client=DemoDeepSeekClient(),
    )
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    return 0


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    """返回演示运行后用户最需要查看的紧凑字段。"""

    return {
        "factor_id": result["factor_id"],
        "tier": result["admission_decision"]["tier"],
        "primary_label": result["tag_package"]["deepseek_tags"]["primary_label"],
        "target_factor_dir": result["route_record"]["target_factor_dir"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
