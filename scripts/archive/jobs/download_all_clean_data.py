#!/usr/bin/env python3
"""下载 COS 全部 A 股清洗数据到本地 cos_data（增量续传，全量到 2026-08-24）。

数据集: StockDailyBar, StockValuationDaily, StockIndustry, StockStatus,
        IndexDailyBar, IndexConstituent, StockList, StockCapitalDaily.
用 clean-cos-ro 逐文件 cp，ThreadPool 并行 + 瞬时错误重试，内存安全（无整库加载）。
"""
import subprocess, time, os, re, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

COS = "/home/sunhaiwei/quantsociety/bin/clean-cos-ro"
BASE = "cos://qs-cold/clean_data/ashare/lqtp_data"
LOCAL = Path("/home/sunhaiwei/cos_data")

# 需要全量落地的数据集（前4个与回测/中性化相关，其余为扩展）
DATASETS = [
    "StockDailyBar", "StockValuationDaily", "StockIndustry", "StockStatus",
    "IndexDailyBar", "IndexConstituent", "StockList", "StockCapitalDaily",
]
# 日志
LOG = Path("/home/sunhaiwei/quant_projects/runs/download_all.log")
LOG.parent.mkdir(parents=True, exist_ok=True)

KEY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.parquet")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cos_cli(*args, max_attempts=5, wait=90):
    for attempt in range(1, max_attempts + 1):
        try:
            r = subprocess.run([COS, *args], check=True, text=True, capture_output=True, timeout=180)
            return r.stdout
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or "").lower()
            transient = any(x in msg for x in ("400", "503", "502", "524", "77", "timeout", "internal error", "tempor", "throttl"))
            if attempt == max_attempts or not transient:
                raise
            time.sleep(wait)
        except subprocess.TimeoutExpired:
            time.sleep(wait)


def list_all_keys(dataset):
    """返回 COS 上该数据集所有 parquet key（含日期）。"""
    out = cos_cli("ls", f"{BASE}/{dataset}/", "--recursive", "--limit", "100000")
    keys = []
    for line in out.splitlines():
        if dataset in line and ".parquet" in line:
            m = re.search(r"/(\d{4}-\d{2}-\d{2})\.parquet", line)
            if m:
                keys.append(m.group(1))
    return sorted(set(keys))


def download_one(dataset, day):
    dst = LOCAL / dataset / f"{day}.parquet"
    if dst.exists() and dst.stat().st_size > 1000:
        return "skip"
    (LOCAL / dataset).mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            cos_cli("cp", f"{BASE}/{dataset}/{day}.parquet", str(dst))
            return "ok"
        except Exception:
            if attempt == 3:
                return "FAIL"
            time.sleep(15)
    return "FAIL"


def main():
    workers = int(os.environ.get("DOWNLOAD_WORKERS", "10"))
    for ds in DATASETS:
        try:
            days = list_all_keys(ds)
        except Exception as e:
            log(f"[{ds}] list 失败: {e}")
            continue
        (LOCAL / ds).mkdir(parents=True, exist_ok=True)
        # 增量：已存在且>1KB 的跳过
        todo = [d for d in days if not ((LOCAL / ds / f"{d}.parquet").exists()
                                        and (LOCAL / ds / f"{d}.parquet").stat().st_size > 1000)]
        log(f"[{ds}] 共 {len(days)} 天, 待下载 {len(todo)}")
        ok = fail = skip = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(download_one, ds, d) for d in todo]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                ok += r == "ok"; fail += r == "FAIL"; skip += r == "skip"
                if i % 300 == 0:
                    log(f"  {ds} {i}/{len(todo)} ok={ok} fail={fail} skip={skip}")
        log(f"[{ds}] done ok={ok} fail={fail} skip={skip}")
    log("ALL_DATASETS_DONE")


if __name__ == "__main__":
    main()
