from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_MONOREPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))

from raw_data_layer.raw_data_fetching.pipeline import download_all_history, download_history, validate_parquet


def _add_download_all_history_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("download-all-history", help="Download Massive REST historical datasets")
    parser.add_argument("--api-key", default=None, help="Massive API key; defaults to MASSIVE_API_KEY env")
    parser.add_argument("--root-dir", default="/home/yluel/share/projects/massive_parquet", help="Output root directory")
    parser.add_argument("--dataset-workers", type=int, default=8, help="Parallel workers across datasets")
    parser.add_argument("--partition-workers", type=int, default=12, help="Parallel workers within one dataset")
    parser.add_argument("--limit", type=int, default=5000, help="Page size limit for pageable endpoints")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per parquet write chunk")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page cap for debugging")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip partitions that already have .ok marker")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Force re-download existing partitions")
    parser.add_argument("--results-csv", default="download_results.csv", help="Output summary CSV path")


def _add_download_history_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("download-history", help="Download flatfiles by prefix and convert to parquet")
    parser.add_argument("--access-key", default=None)
    parser.add_argument("--secret-key", default=None)
    parser.add_argument("--endpoint", default="https://files.massive.com")
    parser.add_argument("--bucket", default="flatfiles")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--prefix-template", default="us_stocks_sip/trades_v1/{year}/{month:02d}/")
    parser.add_argument("--day-prefix-template", default="us_stocks_sip/trades_v1/{year}/{month:02d}/{year}-{month:02d}-{day:02d}")
    parser.add_argument("--local-root", default="./massive_parquet")
    parser.add_argument("--workers", type=int, default=103)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=500000)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--list-prefixes", type=int, default=0, help="List N common prefixes under --prefix and exit")
    parser.add_argument("--list-keys", type=int, default=0, help="List N keys under --prefix and exit")


def _add_validate_parquet_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validate-parquet", help="Validate parquet files under a directory")
    parser.add_argument("directory", help="Parquet root directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw_data_fetching jobs through a unified pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_all_history_parser(subparsers)
    _add_download_history_parser(subparsers)
    _add_validate_parquet_parser(subparsers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download-all-history":
        payload = download_all_history(
            api_key=args.api_key,
            root_dir=args.root_dir,
            dataset_workers=args.dataset_workers,
            partition_workers=args.partition_workers,
            limit=args.limit,
            chunk_size=args.chunk_size,
            max_pages=args.max_pages,
            skip_existing=args.skip_existing,
            results_csv=args.results_csv,
            emit_logs=False,
        )
    elif args.command == "download-history":
        payload = download_history(
            access_key=args.access_key,
            secret_key=args.secret_key,
            endpoint=args.endpoint,
            bucket=args.bucket,
            year=args.year,
            month=args.month,
            day=args.day,
            prefix=args.prefix,
            prefix_template=args.prefix_template,
            day_prefix_template=args.day_prefix_template,
            local_root=args.local_root,
            workers=args.workers,
            max_files=args.max_files,
            chunk_rows=args.chunk_rows,
            list_prefixes=args.list_prefixes,
            list_keys=args.list_keys,
            skip_existing=args.skip_existing,
            emit_logs=False,
        )
    else:
        payload = validate_parquet(args.directory, workers=args.workers, emit_logs=False)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()