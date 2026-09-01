"""Dedup audit 报告（§61 Phase C）。

raw count / exact / sign / family / parse 失败计数。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.seen.store import SeenStore


class AuditReport:
    """§61 去重审计报告。"""

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    def raw_count(self) -> int:
        """原始 factor 行数（每 version 每 signal 一行）。"""
        return self._store.count_factors()

    def alias_count(self) -> int:
        return self._store.count_aliases()

    def parse_failure_count(self) -> int:
        return self._store.count_parse_failures()

    def exact_duplicate_count(self) -> int:
        """EXACT_DUPLICATE 判定次数（累计 meta）。"""
        rows = self._store.q(
            "SELECT key, value FROM seen_meta WHERE key LIKE 'verdict:EXACT_DUPLICATE:%'"
        )
        return sum(int(v) for _, v in rows)

    def sign_equivalent_count(self) -> int:
        rows = self._store.q(
            "SELECT key, value FROM seen_meta WHERE key LIKE 'verdict:SIGN_EQUIVALENT_DUPLICATE:%'"
        )
        return sum(int(v) for _, v in rows)

    def new_count(self) -> int:
        rows = self._store.q(
            "SELECT key, value FROM seen_meta WHERE key LIKE 'verdict:NEW:%'"
        )
        return sum(int(v) for _, v in rows)

    def family_count(self) -> int:
        return self._store.count_families()

    def report(self) -> dict[str, Any]:
        """生成完整 audit 报告 dict。"""
        return {
            "raw_factor_count": self.raw_count(),
            "alias_count": self.alias_count(),
            "exact_duplicate_count": self.exact_duplicate_count(),
            "sign_equivalent_count": self.sign_equivalent_count(),
            "new_count": self.new_count(),
            "family_count": self.family_count(),
            "parse_failure_count": self.parse_failure_count(),
            "total_actions": self._store.count_actions(),
            "total_subtrees": self._store.count_subtrees(),
        }

    def render_text(self) -> str:
        """渲染可读文本报告。"""
        r = self.report()
        lines = [
            "Dedup Audit Report (§61)",
            "=========================",
            f"Raw factor rows      : {r['raw_factor_count']}",
            f"Alias rows           : {r['alias_count']}",
            f"NEW verdicts         : {r['new_count']}",
            f"EXACT_DUPLICATE      : {r['exact_duplicate_count']}",
            f"SIGN_EQUIVALENT      : {r['sign_equivalent_count']}",
            f"Families             : {r['family_count']}",
            f"Parse failures       : {r['parse_failure_count']}",
            f"Actions seen         : {r['total_actions']}",
            f"Subtrees             : {r['total_subtrees']}",
        ]
        return "\n".join(lines)