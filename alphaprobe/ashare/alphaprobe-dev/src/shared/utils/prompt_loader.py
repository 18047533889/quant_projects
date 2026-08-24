"""从 configs/prompts/ 加载 AlphaPROBE LLM 提示词。

编辑 ``configs/prompts/*.txt`` 即可调 prompt，无需改 Python。
实验 yaml 可通过 ``mining.prompts.dir`` 覆盖目录。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMPTS_DIR = _PROJECT_ROOT / "configs" / "prompts"

# 文件名 → 导出的常量名（供 from shared.utils.prompt import ...）
_PROMPT_FILES: dict[str, str] = {
    "PROMPT_HEAD": "system_head.txt",
    "PROMPT_FEATURES_AND_OPERATORS": "features_operators_fe_dsl.txt",
    "PROMPT_DIMENSION_REDUCTION": "validity_fe_dsl.txt",
    "PROMPT_GENERARTION": "generation.txt",
    "PROMPT_COMPARE": "compare_fe_dsl.txt",
    "PROMPT_SEPARATION": "separation_fe_dsl.txt",
}

_active_dir: Optional[Path] = None


def project_root() -> Path:
    return _PROJECT_ROOT


def default_prompts_dir() -> Path:
    return _DEFAULT_PROMPTS_DIR


def set_prompts_dir(path: str | Path | None) -> None:
    """运行时切换 prompt 目录（由 train.py / trainer 在启动时调用）。"""
    global _active_dir
    if path is None:
        _active_dir = None
        _load_prompt_file.cache_clear()
        return
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    _active_dir = p
    _load_prompt_file.cache_clear()


def prompts_dir() -> Path:
    return _active_dir if _active_dir is not None else _DEFAULT_PROMPTS_DIR


@lru_cache(maxsize=32)
def _load_prompt_file(filename: str) -> str:
    path = prompts_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = []
    in_header_comments = True
    for line in text.splitlines():
        stripped = line.strip()
        if in_header_comments:
            if stripped.startswith("#"):
                continue
            if stripped == "":
                continue
            in_header_comments = False
        lines.append(line)
    body = "\n".join(lines).strip()
    return body + ("\n" if body else "")


def load_prompt(const_name: str) -> str:
    filename = _PROMPT_FILES.get(const_name)
    if filename is None:
        raise KeyError(f"unknown prompt constant: {const_name}")
    return _load_prompt_file(filename)


def build_generation_user_prompt(
    topic: str,
    expressions: str,
    explanations: str,
    traces: str,
    num: int,
) -> str:
    """拼装生成任务的 user prompt（算子说明 + 有效性 + 生成任务）。"""
    parts = [
        load_prompt("PROMPT_FEATURES_AND_OPERATORS"),
        load_prompt("PROMPT_DIMENSION_REDUCTION"),
        load_prompt("PROMPT_GENERARTION").format(
            topic=topic,
            expressions=expressions,
            explanations=explanations,
            traces=traces,
            num=num,
        ),
    ]
    return "\n".join(p.strip() for p in parts if p.strip()) + "\n"


def init_prompt_constants() -> dict[str, str]:
    """加载全部 prompt 常量到 dict。"""
    return {name: load_prompt(name) for name in _PROMPT_FILES}
