#!/usr/bin/env python3
"""刷新 datasets.yaml 静态数据集的 sidecar 统计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新 data_access 数据集 sidecar 统计")
    parser.add_argument("--dataset", required=True, help="datasets.yaml 中的数据集名")
    parser.add_argument("--config", default=None, help="datasets.yaml 路径")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    from data_access.registry import StaticDataset, load_registry
    from data_access.stats import build_dataset_stats_snapshot, write_stats_sidecar
    from data_access.store import DataAccessStore
    from data_access.engine import DuckDBEngine

    registry = load_registry(args.config)
    ds = registry.get(args.dataset)
    if not isinstance(ds, StaticDataset):
        print(f"仅支持 static 数据集，收到 kind={ds.kind}", file=sys.stderr)
        return 2

    store = DataAccessStore(registry, DuckDBEngine())
    snapshot = build_dataset_stats_snapshot(store, args.dataset)
    out = write_stats_sidecar(ds.root, snapshot)
    print(f"已写入 {out} rows={snapshot.num_rows} files={snapshot.num_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
