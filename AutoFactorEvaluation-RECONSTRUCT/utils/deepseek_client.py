"""DeepSeek API 客户端（Gateway 专用）。

从项目根目录的 .env 文件读取 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_api_key() -> str:
    """从项目根 .env 读取 DEEPSEEK_API_KEY。"""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY":
                    return v.strip()
    raise RuntimeError(
        f"DEEPSEEK_API_KEY 未在 {env_path} 中找到"
    )


_API_KEY: str | None = None


def get_api_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = _load_api_key()
    return _API_KEY


def deepseek_chat(
    messages: list[dict[str, str]],
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> str:
    """调用 DeepSeek V4 Flash API。

    Args:
        messages: OpenAI-format 消息列表。
        model: 模型名。
        temperature: 采样温度（0 = 确定输出）。
        max_tokens: 最大输出 Token。

    Returns:
        API 返回的文本内容。
    """
    import httpx

    api_key = get_api_key()
    resp = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
