# -*- coding: utf-8 -*-
"""``python -m data_access.service`` 或 ``data-access-server`` 入口。"""
from __future__ import annotations

import argparse
import os


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
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "缺少 uvicorn。请先: pip install 'data-access[service]'"
        ) from exc

    uvicorn.run(
        "data_access.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
