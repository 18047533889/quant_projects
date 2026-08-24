#!/usr/bin/env python3
"""将 ``factor_engine_llm_prompt.md`` 同步为纯文本 ``factor_engine_llm_prompt.txt``。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = FE_ROOT / "factor_engine" / "docs" / "factor_engine_llm_prompt.md"
TXT_PATH = FE_ROOT / "factor_engine" / "docs" / "factor_engine_llm_prompt.txt"


def _strip_inline_md(text: str) -> str:
    """去除 Markdown 内联语法（链接、粗体、行内代码）。"""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def md_to_txt(md: str) -> str:
    """将 Markdown 文档转换为纯文本格式（标题、引用、分隔线等）。"""
    out: list[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip() and (not out or out[-1] != ""):
            out.append("")
            continue
        if line.startswith("# "):
            title = _strip_inline_md(line[2:].strip())
            out.extend(["=" * 80, title.upper(), "=" * 80, ""])
            continue
        if line.startswith("## "):
            title = _strip_inline_md(line[3:].strip())
            out.extend(["", title, "-" * len(title), ""])
            continue
        if line.startswith("### "):
            title = _strip_inline_md(line[4:].strip())
            out.extend(["", f"--- {title} ---", ""])
            continue
        if line.startswith("#### "):
            title = _strip_inline_md(line[5:].strip())
            out.extend(["", title, ""])
            continue
        if line.startswith("> "):
            out.append("NOTE: " + _strip_inline_md(line[2:].strip()))
            continue
        if line.strip() == "---":
            out.append("")
            continue
        out.append(_strip_inline_md(line))
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def main() -> int:
    """同步 ``factor_engine_llm_prompt.md`` 为纯文本 ``.txt``；``--check`` 用于 CI。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验 txt 已与 md 同步（CI 门禁）",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=MD_PATH,
        help="源 Markdown 路径",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=TXT_PATH,
        help="目标纯文本路径",
    )
    args = parser.parse_args()

    if not args.md.is_file():
        print(f"FAIL: missing {args.md}", file=sys.stderr)
        return 1

    payload = md_to_txt(args.md.read_text(encoding="utf-8"))
    if args.check:
        if not args.txt.is_file():
            print(f"FAIL: missing {args.txt}", file=sys.stderr)
            return 1
        if args.txt.read_text(encoding="utf-8") != payload:
            print(
                f"FAIL: {args.txt.name} out of date; run sync_factor_engine_llm_prompt_txt.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.txt.name} synced with {args.md.name}")
        return 0

    args.txt.parent.mkdir(parents=True, exist_ok=True)
    args.txt.write_text(payload, encoding="utf-8")
    print(f"Wrote {args.txt} ({len(payload.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
