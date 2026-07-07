# 主入口：启动研报 → YAML → 评分 → 迭代 流程
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

# 保证从项目根可导入 config / tools / verifiers 等
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_message_anthropic(model: str, system: str, messages: list, tools: list, max_tokens: int):
    """Anthropic API：需安装 anthropic，并设置 ANTHROPIC_API_KEY。"""
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )
    return resp


def create_message_minimax(model: str, system: str, messages: list, tools: list, max_tokens: int):
    """Minimax(Anthropic兼容) API：需安装 anthropic，并设置 MINIMAX_API_KEY。"""
    anthropic = importlib.import_module("anthropic")

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    resp = client.messages.create(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )
    return resp


def main() -> None:
    from config.settings import MODEL, MAX_TOKENS
    from core.agent_loop import agent_loop

    parser = argparse.ArgumentParser(description="factor_agent main entry")
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "minimax"),
        choices=["anthropic", "minimax"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Model name. For minimax, set a valid minimax model name.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="请根据当前上下文生成首版配置文件 YAML，并写入 output/config_v1.yaml，然后调用 run_eval 评分。",
    )
    args = parser.parse_args()

    if args.provider == "anthropic":
        if importlib.util.find_spec("anthropic") is None:
            print("未安装 anthropic，请先执行: pip install anthropic")
            sys.exit(1)
        create_fn = create_message_anthropic
        selected_model = args.model or MODEL
    else:
        if importlib.util.find_spec("anthropic") is None:
            print("未安装 anthropic，请先执行: pip install anthropic")
            sys.exit(1)
        if not os.getenv("MINIMAX_API_KEY"):
            print("缺少 MINIMAX_API_KEY，请先设置环境变量")
            sys.exit(1)
        create_fn = create_message_minimax
        selected_model = args.model or os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")

    response, state = agent_loop(
        args.query,
        create_fn,
        model=selected_model,
        max_tokens=MAX_TOKENS,
    )
    print("rounds:", state.round, "passed:", state.passed)
    if response is not None:
        print("response:", response)
    if state.final_output is not None:
        print("final_output:", state.final_output)


if __name__ == "__main__":
    main()
