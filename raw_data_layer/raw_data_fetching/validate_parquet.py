# validate_parquet.py
# 说明: 递归扫描指定目录下的 Parquet 文件，并进行一系列完整性和数据质量检查。

import os
import argparse
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import traceback
import multiprocessing as mp
from typing import Any

# --- CONFIG ---
# 并行 worker 数
WORKERS = max(1, mp.cpu_count() - 1)
# 内存保护：一次读取的行数
CHUNK_ROWS = 1_000_000
# --- END CONFIG ---

def get_date_from_path(file_path):
    """从文件路径中解析出 'YYYY-MM-DD' 格式的日期。"""
    try:
        # 从后向前查找路径部分，直到找到形如 'YYYY-MM-DD.parquet' 的文件名
        for part in reversed(file_path.split(os.sep)):
            part_name = os.path.splitext(part)[0]
            if len(part_name) == 10 and part_name[4] == '-' and part_name[7] == '-':
                return pd.to_datetime(part_name)
    except Exception:
        return None
    return None

def validate_file(file_path):
    """
    对单个 Parquet 文件进行分块验证，返回一个包含 (file_path, report_list) 的元组。
    """
    report = []
    total_rows = 0
    
    # 1. 检查文件可读性和是否为空
    try:
        pq_file = pq.ParquetFile(file_path)
        total_rows = pq_file.metadata.num_rows
        if total_rows == 0:
            report.append("[EMPTY] 文件为空，不包含任何数据行。")
            return (file_path, report)
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e))
        report.append(f"[CORRUPT] 文件损坏或无法读取: {tb.strip()}")
        return (file_path, report)

    # 初始化聚合变量
    file_date = get_date_from_path(file_path)
    ts_cols = [col for col in pq_file.schema.names if 'ts' in col or 'timestamp' in col]
    ts_col = ts_cols[0] if ts_cols else None
    
    unique_dates = set()
    nan_counts = {col: 0 for col in pq_file.schema.names}
    has_negative = {col: False for col in pq_file.schema.names}
    cols_to_check = ['ask_price', 'bid_price', 'price', 'size']

    # 2. 分块迭代和验证
    try:
        for batch in pq_file.iter_batches(batch_size=CHUNK_ROWS):
            df = batch.to_pandas()
            
            # 时间戳验证
            if file_date and ts_col and ts_col in df.columns:
                unique_dates.update(pd.to_datetime(df[ts_col]).dt.date)
            
            # 数据质量检查
            for col in cols_to_check:
                if col in df.columns:
                    nan_counts[col] += df[col].isna().sum()
                    if not has_negative[col]:
                        if df[col].dropna().lt(0).any():
                            has_negative[col] = True
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e))
        report.append(f"[CORRUPT] 处理文件块时出错: {tb.strip()}")
        return (file_path, report)

    # 3. 生成最终报告
    if file_date and ts_col:
        unique_dates_list = sorted(list(unique_dates))
        if file_date.date() not in unique_dates:
            report.append(f"[DATES_MISMATCH] 文件名日期 {file_date.date()} 不在时间戳列 '{ts_col}' 中。实际包含的日期: {unique_dates_list}")
        elif len(unique_dates_list) > 1:
            report.append(f"[DATES_WARNING] 时间戳列 '{ts_col}' 中包含多个日期: {unique_dates_list}")

    for col in cols_to_check:
        if nan_counts.get(col, 0) > 0:
            nan_pct = (nan_counts[col] / total_rows) * 100
            report.append(f"[QUALITY_WARNING] 列 '{col}' 包含 {nan_counts[col]} 个 NaN 值 ({nan_pct:.2f}%)。")
        if has_negative.get(col, False):
            report.append(f"[QUALITY_WARNING] 列 '{col}' 包含负值。")

    return (file_path, report)


def validate_directory(directory: str, *, workers: int = WORKERS, emit_logs: bool = False) -> dict[str, Any]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录 '{directory}' 不存在。")

    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))

    parquet_files.sort()
    file_count = len(parquet_files)
    issue_files: list[dict[str, Any]] = []
    error_files = 0
    warning_files = 0

    if file_count == 0:
        return {
            "command": "validate-parquet",
            "directory": os.path.abspath(directory),
            "total_files": 0,
            "error_files": 0,
            "warning_files": 0,
            "healthy_files": 0,
            "issue_files": issue_files,
        }

    with mp.Pool(workers) as pool:
        with tqdm(total=file_count, desc="验证进度", unit="file", disable=not emit_logs) as pbar:
            results = pool.imap_unordered(validate_file, parquet_files)

            for file_path, report in results:
                if report:
                    is_error = any("[CORRUPT]" in msg or "[EMPTY]" in msg or "[DATES_MISMATCH]" in msg for msg in report)
                    is_warning = any("[WARNING]" in msg for msg in report)

                    if is_error:
                        error_files += 1
                    elif is_warning:
                        warning_files += 1

                    issue = {
                        "file_path": file_path,
                        "messages": report,
                        "severity": "error" if is_error else "warning",
                    }
                    issue_files.append(issue)

                    if emit_logs:
                        tqdm.write("\n" + "=" * 30)
                        tqdm.write(f"发现问题: {file_path}")
                        tqdm.write("=" * 30)
                        for msg in report:
                            tqdm.write(f"  - {msg}")

                pbar.update(1)

    healthy_files = file_count - error_files - warning_files
    return {
        "command": "validate-parquet",
        "directory": os.path.abspath(directory),
        "total_files": file_count,
        "error_files": error_files,
        "warning_files": warning_files,
        "healthy_files": healthy_files,
        "issue_files": issue_files,
    }

def main():
    parser = argparse.ArgumentParser(description="并行验证 Parquet 文件的完整性和数据质量。")
    parser.add_argument("directory", default="./massive_parquet/us_stocks_sip/trades_v1", nargs="?", help="需要扫描的 Parquet 文件根目录 (默认为: ./massive_parquet)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"并行 worker 数 (默认为: {WORKERS})")
    args = parser.parse_args()

    try:
        result = validate_directory(args.directory, workers=args.workers, emit_logs=True)
    except FileNotFoundError as exc:
        print(f"错误: {exc}")
        return

    if result["total_files"] == 0:
        print("未找到任何 .parquet 文件。")
        return

    print("\n\n=== 验证摘要 ===")
    print(f"总共检查文件数: {result['total_files']}")
    print(f"存在严重错误的文件 (损坏、为空、日期不匹配): {result['error_files']}")
    print(f"存在质量警告的文件 (NaNs, 负值等): {result['warning_files']}")
    print(f"确认健康的文件: {result['healthy_files']}")
    print("====================")

if __name__ == "__main__":
    main()