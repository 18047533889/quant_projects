#!/usr/bin/env python3
"""下载回测窗口的行业 + 市值数据到本地 cos_data（供中性化评估用）。"""
import subprocess, time, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

COS = "/home/sunhaiwei/quantsociety/bin/clean-cos-ro"
BASE = "cos://qs-cold/clean_data/ashare/lqtp_data"
LOCAL = Path("/home/sunhaiwei/cos_data")

DATASETS = ["StockIndustry", "StockValuationDaily"]
START, END = date(2019, 1, 1), date(2025, 12, 31)
KEY_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})\.parquet")


def cos_cli(*args, max_attempts=4, wait=90):
    for attempt in range(1, max_attempts + 1):
        try:
            r = subprocess.run([COS, *args], check=True, text=True, capture_output=True)
            return r.stdout
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or "").lower()
            transient = any(x in msg for x in ("400", "503", "502", "524", "77", "timeout", "internal error", "tempor"))
            if attempt == max_attempts or not transient:
                raise
            time.sleep(wait)


def list_keys(dataset):
    out = cos_cli("ls", f"{BASE}/{dataset}/", "--recursive", "--limit", "100000")
    keys = []
    for line in out.splitlines():
        m = KEY_RE.search(line)
        if m and START <= date.fromisoformat(m.group(1)) <= END:
            keys.append(m.group(1))
    return sorted(keys)


def download_one(dataset, day):
    dst = LOCAL / dataset / f"{day}.parquet"
    if dst.exists() and dst.stat().st_size > 1000:
        return "skip"
    for attempt in range(1, 4):
        try:
            cos_cli("cp", f"{BASE}/{dataset}/{day}.parquet", str(dst))
            return "ok"
        except Exception:
            if attempt == 3:
                return "FAIL"
            time.sleep(20)
    return "FAIL"


def main():
    (LOCAL / "StockIndustry").mkdir(parents=True, exist_ok=True)
    (LOCAL / "StockValuationDaily").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        days = list_keys(ds)
        print(f"[{ds}] {len(days)} 天待下载", flush=True)
        ok = fail = skip = 0
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(download_one, ds, d) for d in days]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                ok += r == "ok"; fail += r == "FAIL"; skip += r == "skip"
                if i % 300 == 0:
                    print(f"  {ds} {i}/{len(days)} ok={ok} fail={fail} skip={skip}", flush=True)
        print(f"[{ds}] done ok={ok} fail={fail} skip={skip}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
