"""AlphaPROBE LLM 提示词 — 正文见 configs/prompts/*.txt"""

from __future__ import annotations

from shared.utils.prompt_loader import (
    build_generation_user_prompt,
    load_prompt,
    set_prompts_dir,
)

PROMPT_HEAD = load_prompt("PROMPT_HEAD")
PROMPT_FEATURES_AND_OPERATORS = load_prompt("PROMPT_FEATURES_AND_OPERATORS")
PROMPT_COMPARE = load_prompt("PROMPT_COMPARE")
PROMPT_DIMENSION_REDUCTION = load_prompt("PROMPT_DIMENSION_REDUCTION")
PROMPT_GENERARTION = load_prompt("PROMPT_GENERARTION")
PROMPT_SEPARATION = load_prompt("PROMPT_SEPARATION")


def reload_prompts() -> None:
    """重新从磁盘加载（set_prompts_dir 之后调用）。"""
    global PROMPT_HEAD, PROMPT_FEATURES_AND_OPERATORS, PROMPT_COMPARE
    global PROMPT_DIMENSION_REDUCTION, PROMPT_GENERARTION, PROMPT_SEPARATION
    from shared.utils.prompt_loader import _load_prompt_file

    _load_prompt_file.cache_clear()
    PROMPT_HEAD = load_prompt("PROMPT_HEAD")
    PROMPT_FEATURES_AND_OPERATORS = load_prompt("PROMPT_FEATURES_AND_OPERATORS")
    PROMPT_COMPARE = load_prompt("PROMPT_COMPARE")
    PROMPT_DIMENSION_REDUCTION = load_prompt("PROMPT_DIMENSION_REDUCTION")
    PROMPT_GENERARTION = load_prompt("PROMPT_GENERARTION")
    PROMPT_SEPARATION = load_prompt("PROMPT_SEPARATION")


__all__ = [
    "PROMPT_HEAD",
    "PROMPT_FEATURES_AND_OPERATORS",
    "PROMPT_COMPARE",
    "PROMPT_DIMENSION_REDUCTION",
    "PROMPT_GENERARTION",
    "PROMPT_SEPARATION",
    "build_generation_user_prompt",
    "reload_prompts",
    "set_prompts_dir",
]
