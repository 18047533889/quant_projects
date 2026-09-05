#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""monorepo 根入口 wrapper：data_access T8 多 artifact 原子发布验收 smoke。

验收命令兼容（UPSTREAM_FIX_PLAN §验收命令）：

    python scripts/smoke_t8_remote_publish.py [--tmp-base DIR] [--factor-prefix PREFIX]

默认纯本地（LocalObjectStore + 临时 staging，DATA_ACCESS 环境无关）；真实 COS
smoke 需 ``T8_REAL_COS=1`` 且 ``cos://qs-cold/factor_pool/_smoke/<run_id>/`` 可写
（opt-in）。退出码：全部 PASS=0，任一 FAIL 非零。详见被 wrapper 的实现脚本头部。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA_ACCESS = _REPO / "data_access"
_TARGET = _DATA_ACCESS / "scripts" / "smoke_t8_remote_publish.py"
for _p in (str(_REPO), str(_DATA_ACCESS)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv: list[str] | None = None) -> int:
    if not _TARGET.is_file():
        print(
            f"找不到实现脚本: {_TARGET}（本 wrapper 依赖 data_access/scripts/"
            f"smoke_t8_remote_publish.py）",
            file=sys.stderr,
        )
        return 1
    sys.argv = [str(_TARGET)] + list(argv or sys.argv[1:])
    import runpy

    namespace = runpy.run_path(str(_TARGET), run_name="__main__")
    return int(namespace.get("main")(sys.argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
