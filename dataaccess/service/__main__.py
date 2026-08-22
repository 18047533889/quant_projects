# -*- coding: utf-8 -*-
"""``python -m data_access.service`` 或 ``data-access-server`` 入口。"""
from __future__ import annotations

import argparse
import os
import sys


def _run_production_startup_gate() -> None:
    """R26-P0-020：服务启动 lifecycle——listen socket 之前先跑 production startup gate。

    任一 critical fail → 进程非零退出（不能带着「security 未配置 / calendar 不可
    证明 / single-worker 违反」的状态对外服务）。
    """
    try:
        from data_access.runtime.startup_gate import run_startup_gate
        from data_access.store import get_store
    except Exception as exc:
        print(f"startup gate: 无法加载 store：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        run_startup_gate(get_store())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"startup gate failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="data_access 只读 HTTP 服务")
    parser.add_argument(
        "--host",
        default=os.environ.get("DATA_ACCESS_API_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DATA_ACCESS_API_PORT", "8765")),
    )
    parser.add_argument(
        "--skip-startup-gate",
        action="store_true",
        help="跳过 production startup gate（仅本地开发/调试）",
    )
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "缺少 uvicorn。请先: pip install 'data-access[service]'"
        ) from exc

    if not args.skip_startup_gate:
        _run_production_startup_gate()

    uvicorn.run(
        "data_access.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
