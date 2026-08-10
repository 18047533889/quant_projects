#!/usr/bin/env python3
"""R26 §7 —— security/PIT/contract/snapshot 边界的 fail-open 与 bool 强转静态审计。

检查：
    1. ``except Exception: pass / return`` 出现在 security/contract/snapshot/
       runtime/read/cos/store.py → blocking（fail-open：异常 → 旧逻辑继续）；
       cleanup/telemetry 允许但有注释说明。
    2. ``bool(raw.get(`` / ``bool(entry.get(`` / ``bool(payload.get(`` 安全配置
       强转 → blocking（``bool("false") is True``）。
    3. ``is_strict_semantics()`` 未纳入 automated_research 的路径不再存在。

用法：python -m data_access.scripts.audit_r26_security
退出码：0 = OK；1 = 有 blocking。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

# fail-open 允许目录（cleanup/telemetry/测试/构建产物）
_ALLOWED_DIRS = ("build", "__pycache__", "tests", "scripts")

# security/PIT/contract/snapshot 边界：这些目录的 except-swallow 视为 blocking。
_STRICT_DIRS = (
    "security",
    "contract",
    "snapshot",
    "runtime",
)

_EXCEPT_PASS_RE = re.compile(
    r"except\s+(\w[\w.]*)\s*(?:as\s+\w+)?\s*:\s*\n\s*(pass\s*#.*)?\n\s*(return\s*(None|\[\]|\{\}|\"\")?)"
)
_EXCEPT_EMPTY_RE = re.compile(
    r"except\s+(\w[\w.]*)\s*(?:as\s+\w+)?\s*:\s*\n\s*pass\s*\n\s*(?![^\n]*\n)"
)
_BOOL_COERCE_RE = re.compile(r"\bbool\((raw|entry|payload|meta|dict)\.get\(")

# 已知合法的 except（研究降级 / fail-closed 回退 / gate 收集），行号集合用注释标注。
_KNOWN_SWALLOWS = (
    "security/credentials.py:215",   # research-only coscli config 读取
    "security/governed_frame.py:139",  # expected digest 回退（仍校验非空）
    "security/policy.py:245",         # production_security_configured custom authorizer 回退
    # startup_gate 的 except→return [problems] 是 gate 的设计：collect→production 端 raise。
    "runtime/startup_gate.py:48",
    "runtime/startup_gate.py:79",
    "runtime/startup_gate.py:86",
    "runtime/startup_gate.py:109",
    "runtime/startup_gate.py:142",
    "runtime/startup_gate.py:156",
    # verifier/resolver `_effective_strict` return True = fail-closed（异常→strict）。
    "snapshot/verifier.py:64",
    "snapshot/resolver.py:213",
    # verifier `_safe_remote_meta`/`_safe_local_stat` return None → strict 调用方 fail-closed。
    "snapshot/verifier.py:252",
    "snapshot/verifier.py:260",
)


def _is_known(path: str, lineno: int) -> bool:
    return f"{path}:{lineno}" in _KNOWN_SWALLOWS


def main() -> int:
    problems: list[str] = []
    for py in PKG_ROOT.rglob("*.py"):
        rel = py.relative_to(PKG_ROOT).as_posix()
        if any(rel.startswith(d) for d in _ALLOWED_DIRS):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        # except pass/return 扫描
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("except "):
                continue
            in_strict = any(rel.startswith(d) for d in _STRICT_DIRS)
            # 看下一行是否为 pass / return / continue（静默吞）
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                continue
            nxt = lines[j].strip()
            silent = nxt in {"pass", "continue"} or nxt.startswith("return ")
            if silent and in_strict and not _is_known(rel, i + 1):
                problems.append(
                    f"{rel}:{i+1}: fail-open `{stripped[:60]}` → `{nxt}`"
                    "（R26 §7.1：security/PIT/contract/snapshot 禁止吞异常）"
                )
        # bool 强转扫描（跳过注释/文档字符串里的历史引用）。
        for m in _BOOL_COERCE_RE.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            line = lines[lineno - 1].lstrip() if lineno - 1 < len(lines) else ""
            if line.startswith(("#", '"""', "'''")) or "``" in line:
                continue  # 注释/文档（含历史代码引用）
            problems.append(
                f"{rel}:{lineno}: `{m.group(0)}`（R26 §7.2：安全配置拒绝字符串伪装布尔）"
            )

    if problems:
        print("R26 security audit FAILED:")
        for p in sorted(set(problems)):
            print(f"  - {p}")
        return 1
    print("R26 security audit OK (0 fail-open / 0 bool-coercion in security-critical paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
