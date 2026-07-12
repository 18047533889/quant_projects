#!/usr/bin/env python3
"""启动 data_access 只读 HTTP 服务。

用法（在有 COS/数据盘的服务器上）::

    source env.sh
    pip install -r requirements-service.txt
    python scripts/run_data_access_server.py --host 0.0.0.0 --port 8765

团队客户端（无需 clone 全仓库，只需 pip install httpx pyarrow pandas pydantic）::

    from data_access.service.client import DataAccessClient
    client = DataAccessClient("http://读数服务器:8765", api_key="...")
    df = client.read_frame("ashare_stock_daily", columns=["Close"], time_range=("2024-01-01","2024-01-31"))
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="data_access 只读 HTTP 服务")
    parser.add_argument("--host", default=os.environ.get("DATA_ACCESS_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DATA_ACCESS_API_PORT", "8765")))
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "缺少 uvicorn。请先: pip install -r requirements-service.txt"
        ) from exc

    uvicorn.run(
        "data_access.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
