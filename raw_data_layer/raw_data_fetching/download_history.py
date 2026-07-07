# massive_stream_convert_parquet_with_checksum_and_backoff.py
# 说明: 并行下载 Massive flatfiles（S3-compatible）并流式转换为 Parquet（zstd）
#       增强功能：
#         - 先下载到本地临时文件（boto3.download_fileobj + TransferConfig）
#         - 按 chunk 将 CSV -> parquet（.part 临时文件），原子替换为 .parquet
#         - 计算并写入 SHA-256 checksum（.parquet.sha256），每次跳过前会校验 checksum
#         - 针对 429 / Throttling 做指数退避 + honor Retry-After + jitter
# 依赖: boto3, pandas, pyarrow, tqdm
# pip install boto3 pandas pyarrow tqdm

import os
import sys
import time
import math
import argparse
import traceback
import multiprocessing as mp
from functools import partial
import tempfile
import shutil
import hashlib
import random
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from boto3.s3.transfer import TransferConfig

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ----------------- USER CONFIG -----------------
# 推荐把密钥放到环境变量 MASSIVE_AWS_KEY / MASSIVE_AWS_SECRET
ACCESS_KEY = os.environ.get("MASSIVE_AWS_KEY", "f04c088e-d96f-4944-a048-8e419c0bd524")
SECRET_KEY = os.environ.get("MASSIVE_AWS_SECRET", "0_STHVfnT0CLigISYj9Oo0SIWVVpC9vO")

ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"

# Default to a small subset (one month) when PREFIX is empty.
DEFAULT_YEAR = 2025
DEFAULT_MONTH = 1
DEFAULT_PREFIX_TEMPLATE = "us_stocks_sip/trades_v1/{year}/{month:02d}/"
DEFAULT_DAY = 0
DEFAULT_DAY_PREFIX_TEMPLATE = "us_stocks_sip/trades_v1/{year}/{month:02d}/{year}-{month:02d}-{day:02d}"

# S3 前缀：按需改。建议先用小前缀测试，例如 "us_stocks_sip/quotes/2024/"
PREFIX = ""

# 本地存储根目录（脚本会在此下创建与 key 相同的子目录）
LOCAL_ROOT = "./massive_parquet"

# 并行 worker 数（根据 CPU / 网络 / 内存调整）
# 默认保守：最多 4 worker，避免被限流。可按需改成更高。
# WORKERS = min(4, max(1, mp.cpu_count() - 1))
WORKERS = 103
# 每个文件的最大重试次数（包括处理/下载阶段）
MAX_RETRIES = 5

# Limit total files processed (0 = no limit)
MAX_FILES = 0

# Only handle CSV/CSV.GZ objects
ALLOWED_SUFFIXES = (".csv", ".csv.gz")

# pandas.read_csv 的参数（可根据 file schema 扩展）
PANDAS_READ_CSV_KWARGS = {
    "sep": ",",
    "dtype": None,
    "low_memory": False,
    "encoding": "utf-8",
}

# Read CSV in chunks to reduce memory; set 0 to disable chunking
CHUNK_ROWS = 500_000

# parquet 写入选项
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3

# 是否跳过已有的 parquet 文件（用于断点续传）
# 当跳过时会校验 parquet + .sha256 一致性；若不一致则重新下载覆盖
SKIP_EXISTING = True

# Analyze storage savings from CSV.GZ -> Parquet
ANALYZE_SAVINGS = True

# Convert integer columns to float to avoid schema mismatch across chunks
CAST_INT_TO_FLOAT = True

# Force specific columns to string to stabilize schema across chunks
FORCE_STRING_COLUMNS = ["ticker", "conditions", "indicators"]

# Backoff / throttling config
BACKOFF_BASE = 1.0       # seconds
BACKOFF_CAP = 120.0      # seconds
JITTER = True            # use full jitter for backoff
# ------------------------------------------------

def create_s3_client(access_key, secret_key, endpoint=ENDPOINT):
    session = boto3.session.Session()
    cfg = Config(
        signature_version="s3v4",
        connect_timeout=60,
        read_timeout=300,
        max_pool_connections=64,
        retries={
            "max_attempts": 10,
            "mode": "adaptive",
        },
    )
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=cfg,
    )
    return client

def key_to_local_path(local_root, key):
    base = os.path.join(local_root, key)
    if base.endswith(".gz"):
        base = base[:-3]
    base = os.path.splitext(base)[0]
    out = base + ".parquet"
    return out

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def normalize_chunk_dtypes(df):
    if FORCE_STRING_COLUMNS:
        for col in FORCE_STRING_COLUMNS:
            if col in df.columns:
                try:
                    df[col] = df[col].astype("string")
                except Exception:
                    df[col] = df[col].astype(str)
    if CAST_INT_TO_FLOAT:
        int_cols = df.select_dtypes(include=["int64", "int32", "Int64"]).columns
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype("float64")
    return df

def list_keys_streaming(s3_client, bucket, prefix="", max_keys=0):
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(ALLOWED_SUFFIXES):
                continue
            yield key
            count += 1
            if max_keys and count >= max_keys:
                return

def list_common_prefixes(s3_client, bucket, prefix="", delimiter="/", max_prefixes=0):
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
        for obj in page.get("CommonPrefixes", []):
            p = obj.get("Prefix")
            if p is None:
                continue
            yield p
            count += 1
            if max_prefixes and count >= max_prefixes:
                return

def format_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def normalize_prefix(prefix, bucket):
    if not prefix:
        return prefix
    p = prefix.lstrip("/")
    if p.startswith(bucket + "/"):
        p = p[len(bucket) + 1:]
    return p

# ---------- checksum helpers ----------
def sha256_file_hex(path, buf_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_checksum_file(parquet_path, hex_digest):
    chk_path = parquet_path + ".sha256"
    with open(chk_path, "w", encoding="utf-8") as f:
        f.write(hex_digest)
    return chk_path

def read_checksum_file(parquet_path):
    chk_path = parquet_path + ".sha256"
    if not os.path.exists(chk_path):
        return None
    try:
        with open(chk_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt
    except Exception:
        return None

# ---------- backoff helper ----------
def backoff_sleep(attempt, err=None):
    """
    attempt: 1-based attempt number
    err: exception instance (may be botocore ClientError)
    Behavior:
      - If err contains Retry-After header, use it (with jitter)
      - Otherwise exponential backoff with full jitter, capped
    """
    # honor Retry-After if present in ClientError HTTP headers
    if isinstance(err, ClientError):
        try:
            resp_meta = err.response.get("ResponseMetadata", {}) or {}
            headers = resp_meta.get("HTTPHeaders") or {}
            ra = headers.get("retry-after") or headers.get("x-retry-after")
            if ra:
                try:
                    sec = int(float(ra))
                    jitter = random.uniform(0.2, 1.5)
                    wait = min(BACKOFF_CAP, sec + jitter)
                    time.sleep(wait)
                    return
                except Exception:
                    pass
        except Exception:
            pass

    # fallback: exponential backoff with full jitter
    exp = BACKOFF_BASE * (2 ** (attempt - 1))
    if JITTER:
        sleep = random.uniform(0, min(BACKOFF_CAP, exp))
    else:
        sleep = min(BACKOFF_CAP, exp)
    time.sleep(sleep)

# ---------- core worker ----------
def process_one_object(s3_params, key):
    """
    Robust implementation with:
      - throttling handling (429)
      - download -> tmp_file -> chunked CSV->parquet(.part) -> atomic replace -> checksum
    Returns: (key, status, message, src_bytes, parquet_bytes)
    """
    access_key, secret_key, endpoint, bucket, local_root, skip_existing, parquet_compression, parquet_compression_level, analyze_savings, chunk_rows = s3_params
    try:
        client = create_s3_client(access_key, secret_key, endpoint=endpoint)
    except Exception as e:
        return (key, "error", f"s3_client_create_failed: {e}", 0, 0)

    local_path = key_to_local_path(local_root, key)
    chk_path = local_path + ".sha256"

    # If target exists and SKIP_EXISTING, verify checksum if available.
    if skip_existing and os.path.exists(local_path):
        stored = read_checksum_file(local_path)
        if stored:
            try:
                current = sha256_file_hex(local_path)
                if current == stored:
                    return (key, "skipped", "parquet exists and checksum matches", 0, 0)
                else:
                    # mismatch -> remove and redownload
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(chk_path):
                            os.remove(chk_path)
                    except Exception:
                        pass
            except Exception:
                # can't compute -> remove and redownload
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                try:
                    if os.path.exists(chk_path):
                        os.remove(chk_path)
                except Exception:
                    pass
        else:
            # no stored checksum: attempt to create one (fast path)
            try:
                current = sha256_file_hex(local_path)
                write_checksum_file(local_path, current)
                return (key, "skipped", "parquet exists (checksum created)", 0, 0)
            except Exception:
                # fail to checksum -> remove and redownload
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                try:
                    if os.path.exists(chk_path):
                        os.remove(chk_path)
                except Exception:
                    pass

    # small jitter before starting to reduce thundering herd across workers
    time.sleep(random.uniform(0.0, 0.12))

    # ensure tmp dir exists
    tmp_dir = os.path.join(local_root, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # TransferConfig: conservative to reduce per-download parallelism and throttle risk
    transfer_cfg = TransferConfig(
        multipart_threshold=50 * 1024 * 1024,   # 50MB
        multipart_chunksize=8 * 1024 * 1024,   # 8MB part size (smaller parts tolerate network issues)
        max_concurrency=2,                     # reduce concurrency per-download
        use_threads=True,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        tmp_file = None
        tmp_parquet = None
        writer = None
        try:
            # retrieve head to check content length
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                content_length = head.get("ContentLength", None)
            except Exception:
                content_length = None

            # download to tmp file
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="massive_dl_", dir=tmp_dir, suffix=".gz" if key.endswith(".gz") else "")
            os.close(tmp_fd)
            tmp_file = tmp_path
            with open(tmp_file, "wb") as fobj:
                client.download_fileobj(Bucket=bucket, Key=key, Fileobj=fobj, Config=transfer_cfg)

            # verify size if possible
            src_bytes = os.path.getsize(tmp_file)
            if content_length is not None and src_bytes != content_length:
                raise IOError(f"downloaded size mismatch: got {src_bytes}, expected {content_length}")

            # prepare tmp parquet .part
            tmp_parquet = local_path + ".part"
            ensure_dir_for_file(tmp_parquet)
            try:
                if os.path.exists(tmp_parquet):
                    os.remove(tmp_parquet)
            except Exception:
                pass

            # read CSV in chunks -> write parquet .part
            if chunk_rows and chunk_rows > 0:
                reader = pd.read_csv(tmp_file, chunksize=chunk_rows, **PANDAS_READ_CSV_KWARGS)
                for chunk in reader:
                    chunk = normalize_chunk_dtypes(chunk)
                    if writer is None:
                        table = pa.Table.from_pandas(chunk, preserve_index=False)
                        writer = pq.ParquetWriter(
                            tmp_parquet,
                            table.schema,
                            compression=parquet_compression,
                            compression_level=parquet_compression_level,
                            use_dictionary=True,
                        )
                    else:
                        table = pa.Table.from_pandas(chunk, schema=writer.schema, preserve_index=False)
                    writer.write_table(table)
                if writer is None:
                    # empty CSV -> write empty parquet
                    empty = pa.Table.from_pandas(pd.DataFrame(), preserve_index=False)
                    pq.write_table(
                        empty,
                        tmp_parquet,
                        compression=parquet_compression,
                        compression_level=parquet_compression_level,
                        use_dictionary=True,
                    )
            else:
                df = pd.read_csv(tmp_file, **PANDAS_READ_CSV_KWARGS)
                df = normalize_chunk_dtypes(df)
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(
                    table,
                    tmp_parquet,
                    compression=parquet_compression,
                    compression_level=parquet_compression_level,
                    use_dictionary=True,
                )

            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

            # atomic replace
            ensure_dir_for_file(local_path)
            os.replace(tmp_parquet, local_path)

            # compute checksum and write .sha256
            parquet_checksum = sha256_file_hex(local_path)
            write_checksum_file(local_path, parquet_checksum)

            parquet_bytes = os.path.getsize(local_path) if os.path.exists(local_path) else 0

            # cleanup tmp file
            try:
                if tmp_file and os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass

            return (key, "ok", f"wrote {local_path}", content_length if content_length is not None else src_bytes, parquet_bytes)

        except Exception as e:
            last_err = e

            # detect 429 / throttling from ClientError
            is_client_429 = False
            if isinstance(e, ClientError):
                try:
                    status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    code = e.response.get("Error", {}).get("Code", "")
                    if status_code == 429 or str(code).lower() in ("throttling", "slowdown", "too many requests"):
                        is_client_429 = True
                except Exception:
                    pass

            if is_client_429:
                tqdm.write(f"[WARN][{key}] throttled (429). attempt {attempt}/{MAX_RETRIES}. backing off...")
                backoff_sleep(attempt, e)
            else:
                # cleanup tmp artifacts
                try:
                    if tmp_file and os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except Exception:
                    pass
                try:
                    if tmp_parquet and os.path.exists(tmp_parquet):
                        os.remove(tmp_parquet)
                except Exception:
                    pass
                # small backoff for general errors
                time.sleep(1.0 * attempt)

            # continue retry loop

        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

    # all attempts failed
    tb = "".join(traceback.format_exception_only(type(last_err), last_err))
    return (key, "error", f"failed after {MAX_RETRIES} attempts: {tb}", 0, 0)

# ---------- CLI and main ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Download Massive CSV/CSV.GZ and convert to Parquet (with checksum & backoff)")
    parser.add_argument("--access-key", default=ACCESS_KEY)
    parser.add_argument("--secret-key", default=SECRET_KEY)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--month", type=int, default=DEFAULT_MONTH)
    parser.add_argument("--day", type=int, default=DEFAULT_DAY)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--prefix-template", default=DEFAULT_PREFIX_TEMPLATE)
    parser.add_argument("--day-prefix-template", default=DEFAULT_DAY_PREFIX_TEMPLATE)
    parser.add_argument("--local-root", default=LOCAL_ROOT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--skip-existing", action="store_true", default=SKIP_EXISTING)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--list-prefixes", type=int, default=0, help="List N common prefixes under --prefix and exit")
    parser.add_argument("--list-keys", type=int, default=0, help="List N keys under --prefix and exit")
    return parser.parse_args()

def run_download_history(
    *,
    access_key: str,
    secret_key: str,
    endpoint: str = ENDPOINT,
    bucket: str = BUCKET,
    year: int = DEFAULT_YEAR,
    month: int = DEFAULT_MONTH,
    day: int = DEFAULT_DAY,
    prefix: str = PREFIX,
    prefix_template: str = DEFAULT_PREFIX_TEMPLATE,
    day_prefix_template: str = DEFAULT_DAY_PREFIX_TEMPLATE,
    local_root: str = LOCAL_ROOT,
    workers: int = WORKERS,
    max_files: int = MAX_FILES,
    chunk_rows: int = CHUNK_ROWS,
    list_prefixes: int = 0,
    list_keys: int = 0,
    skip_existing: bool = SKIP_EXISTING,
    analyze_savings: bool = ANALYZE_SAVINGS,
    emit_logs: bool = True,
) -> dict[str, Any]:
    workers = max(1, workers)
    max_files = max(0, max_files)
    chunk_rows = max(0, chunk_rows)

    if not prefix:
        if day and day > 0:
            prefix = day_prefix_template.format(year=year, month=month, day=day)
        else:
            prefix = prefix_template.format(year=year, month=month)

    normalized = normalize_prefix(prefix, bucket)
    if normalized != prefix and emit_logs:
        print(f"Normalized prefix: '{prefix}' -> '{normalized}'")
    prefix = normalized

    if not access_key or not secret_key:
        raise ValueError("ACCESS_KEY / SECRET_KEY 未设置。请设置环境变量 MASSIVE_AWS_KEY/MASSIVE_AWS_SECRET 或直接传参。")

    s3 = create_s3_client(access_key, secret_key, endpoint=endpoint)

    if list_prefixes > 0:
        items = list(list_common_prefixes(s3, bucket, prefix, max_prefixes=list_prefixes))
        return {
            "command": "download-history",
            "mode": "list-prefixes",
            "prefix": prefix,
            "items": items,
            "count": len(items),
        }

    if list_keys > 0:
        items = list(list_keys_streaming(s3, bucket, prefix, max_keys=list_keys))
        return {
            "command": "download-history",
            "mode": "list-keys",
            "prefix": prefix,
            "items": items,
            "count": len(items),
        }

    s3_params = (
        access_key,
        secret_key,
        endpoint,
        bucket,
        local_root,
        skip_existing,
        PARQUET_COMPRESSION,
        PARQUET_COMPRESSION_LEVEL,
        analyze_savings,
        chunk_rows,
    )

    processed = 0
    errors = 0
    skipped = 0
    start_ts = time.time()

    if emit_logs:
        print("Counting objects (fast pass) ...")
    total = 0
    for _ in list_keys_streaming(s3, bucket, prefix, max_keys=max_files):
        total += 1
    if emit_logs:
        print(f"Found {total:,} objects under prefix '{prefix}'")
    if total == 0 and emit_logs:
        print("Hint: prefix may be wrong. Try: --list-prefixes 50 (or set --prefix).")

    iterator = list_keys_streaming(s3, bucket, prefix, max_keys=max_files)

    if emit_logs:
        print(f"Starting processing with {workers} workers...")
    pool = mp.Pool(workers)
    total_src_bytes = 0
    total_parquet_bytes = 0
    savings_samples = 0
    try:
        func = partial(process_one_object, s3_params)
        results = pool.imap_unordered(func, iterator, chunksize=1)

        for res in tqdm(results, total=total, disable=not emit_logs):
            key, status, msg, src_bytes, parquet_bytes = res
            processed += 1
            if status == "error":
                errors += 1
                if emit_logs:
                    tqdm.write(f"[ERROR] {key} -> {msg}")
            elif status == "skipped":
                skipped += 1
            else:
                if analyze_savings and src_bytes and parquet_bytes:
                    total_src_bytes += src_bytes
                    total_parquet_bytes += parquet_bytes
                    savings_samples += 1

    except KeyboardInterrupt:
        if emit_logs:
            print("Interrupted by user, terminating pool...")
        pool.terminate()
        pool.join()
        raise
    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - start_ts
    output: dict[str, Any] = {
        "command": "download-history",
        "mode": "download",
        "prefix": prefix,
        "local_root": str(Path(local_root).resolve()),
        "total_objects": total,
        "processed": processed,
        "ok": max(0, processed - errors - skipped),
        "skipped": skipped,
        "errors": errors,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "throughput_files_per_sec": processed / max(1, elapsed),
    }
    if analyze_savings and savings_samples > 0 and total_src_bytes > 0:
        saved = total_src_bytes - total_parquet_bytes
        saved_pct = (1.0 - (total_parquet_bytes / total_src_bytes)) * 100.0
        output["storage_savings"] = {
            "csv_gz_bytes": total_src_bytes,
            "parquet_bytes": total_parquet_bytes,
            "saved_bytes": saved,
            "saved_pct": saved_pct,
        }
    return output


def main_run():
    args = parse_args()
    try:
        result = run_download_history(
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
            analyze_savings=ANALYZE_SAVINGS,
            emit_logs=True,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    if result["mode"] in {"list-prefixes", "list-keys"}:
        label = "Listing prefixes" if result["mode"] == "list-prefixes" else "Listing keys"
        print(f"{label} under '{result['prefix'] or '/'}' ...")
        for item in result["items"]:
            print(item)
        return

    print("\n=== Summary ===")
    print(f"Processed: {result['processed']:,} (skipped {result['skipped']:,})")
    print(f"Errors: {result['errors']:,}")
    print(
        f"Elapsed: {result['elapsed_seconds']:.1f}s  "
        f"throughput: {result['throughput_files_per_sec']:.2f} files/sec"
    )
    if "storage_savings" in result:
        savings = result["storage_savings"]
        print("\n=== Storage Savings (processed files only) ===")
        print(f"CSV.GZ total: {format_bytes(savings['csv_gz_bytes'])}")
        print(f"Parquet total: {format_bytes(savings['parquet_bytes'])}")
        print(f"Savings: {format_bytes(savings['saved_bytes'])} ({savings['saved_pct']:.2f}%)")

if __name__ == "__main__":
    main_run()