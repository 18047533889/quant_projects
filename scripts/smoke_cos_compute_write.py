#!/usr/bin/env python3
"""真实 COS 小批量：remote 读 A股/美股 clean_data → 运算 → 落 staging。

用法::

    source env.sh
    export DATA_ACCESS_COS_READ_MODE=remote
    export DATA_ACCESS_COS_REMOTE_BACKEND=cli   # 本机无 httpfs 时用 cli
    export QUANT_RUN_NAMESPACE=cos_smoke
    python3 scripts/smoke_cos_compute_write.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_env() -> None:
    os.environ.setdefault("DATA_ACCESS_COS_READ_MODE", "remote")
    os.environ.setdefault("DATA_ACCESS_COS_REMOTE_BACKEND", "cli")
    os.environ.setdefault("QUANT_RUN_NAMESPACE", "cos_smoke")
    os.environ.setdefault("QUANT_SCHEMA_CHECK", "off")
    # 独立 cache，避免污染永久 mirror
    cache = ROOT / "data" / ".cos_remote_cache_smoke"
    os.environ.setdefault("DATA_ACCESS_COS_CACHE_ROOT", str(cache))


def main() -> None:
    _setup_env()
    from data_access import get_store, reset_store
    from data_access.cos_mirror import DATASET_MIRROR_REGISTRY
    from data_access.cos_remote import cos_cache_root, cos_remote_backend, load_cos_cli_credentials_into_env

    reset_store()
    load_cos_cli_credentials_into_env()
    store = get_store()

    print("==> COS remote backend:", cos_remote_backend())
    print("==> COS cache root:", cos_cache_root())
    print(
        "==> Registered COS datasets:",
        len(DATASET_MIRROR_REGISTRY),
        "(ashare + us_massive + us_clean)",
    )
    ashare = sorted(n for n in DATASET_MIRROR_REGISTRY if n.startswith("ashare_"))
    us = sorted(n for n in DATASET_MIRROR_REGISTRY if n.startswith("us_"))
    print("    ashare:", len(ashare), "eg", ashare[:5])
    print("    us:", len(us), "eg", us[:5])

    # ---- A 股：1 个交易日 ----
    print("\n==> A-share ashare_stock_daily remote read + compute_and_write")
    a_result = store.compute_and_write(
        """
        SELECT
          CAST(TradeDate AS TIMESTAMP) AS datetime,
          Symbol AS asset,
          Close AS value,
          2016 AS year
        FROM {{ashare_stock_daily}}
        WHERE Symbol = '000001.SZ'
        """,
        read_datasets=["ashare_stock_daily"],
        read_time_ranges={"ashare_stock_daily": ("2016-01-04", "2016-01-04")},
        write_dataset="factor_lake_staging",
        factor_id="smoke_ashare_close_000001",
        mode="overwrite",
        partition_by=["year"],
    )
    print("    write rows_computed:", a_result.get("rows_computed"))
    print("    write path:", a_result.get("path") or a_result.get("files"))
    print("    keys:", sorted(a_result.keys()))
    a_df = store.read_frame(
        "factor_lake_staging",
        factor_id="smoke_ashare_close_000001",
        columns=["datetime", "asset", "value"],
    )
    print("    staging rows:", len(a_df))
    print(a_df.head().to_string(index=False))

    # ---- 美股：1 个交易日 ----
    print("\n==> US us_stock_daily remote read + compute_and_write")
    u_result = store.compute_and_write(
        """
        SELECT
          CAST(TradeDate AS TIMESTAMP) AS datetime,
          Ticker AS asset,
          Close AS value,
          2024 AS year
        FROM {{us_stock_daily}}
        WHERE Ticker = 'AAPL'
        """,
        read_datasets=["us_stock_daily"],
        read_time_ranges={"us_stock_daily": ("2024-01-02", "2024-01-02")},
        write_dataset="factor_lake_staging",
        factor_id="smoke_us_close_aapl",
        mode="overwrite",
        partition_by=["year"],
    )
    print("    keys:", sorted(u_result.keys()))
    u_df = store.read_frame(
        "factor_lake_staging",
        factor_id="smoke_us_close_aapl",
        columns=["datetime", "asset", "value"],
    )
    print("    staging rows:", len(u_df))
    print(u_df.head().to_string(index=False))

    # ---- 美股 clean_data 衍生层（单文件小表）----
    print("\n==> US clean_data us_is_early_close remote read")
    early = store.read_frame(
        "us_is_early_close",
        columns=None,
        time_range=None,
    )
    print("    rows:", len(early), "cols:", list(early.columns)[:8])

    print("\nOK: COS remote smoke finished. Source COS untouched; results in factor_lake_staging.")


if __name__ == "__main__":
    main()
