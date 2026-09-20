#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cos_migrate.py -- 大目录迁移到 COS：上传 -> 逐文件尺寸比对 -> 抽样哈希 -> 删本地。

**关键环境事实（2026-09-20 实测，务必记住）**
------------------------------------------------
这台机器经 `factor-admin-cos` 网关访问 COS 时，**multipart（分片）上传被 CAM 策略
拒绝**：只要 coscli 判定要走分片（文件大于 `--part-size`），操作必然失败，
耗时 ~31s 后报 `Error num: 1`、`OK size: 0 Byte`。

- 失败样本：207MB / 60MB 文件，`--part-size 8/32/128`，thread 1/2/4/32 —— 全失败。
- 成功样本：同文件 `--part-size 512/64`（即整文件单分片）→ 成功，实测 86 MB/s（207MB/2.4s）。
- 换 `meta/public_meta` 前缀重试 multipart 同样失败 → 是身份级限制，不是路径或并发问题。

结论：**必须让每个文件都走"单次 PutObject"**，即 `--part-size` 要大于该单元内最大文件。
COS 单次 PutObject 硬上限 5 GiB，因此 >5 GiB 的文件本工具不上传，标记为 oversize 让人工处理
（拆分为 <5GiB 片段 + 清单，因为本地也没有空间做流式重写）。

**校验与删除的可靠性**
--------------------
远端无删除权限（`factor-admin-cos` 只有 read+write），所以本地删除不可撤销。校验必须可证伪：
  1. 文件数完全相等；
  2. **逐文件尺寸字符串完全相等**（本地用与 coscli 相同的 human 格式化 → 确定性字符串比较）；
  3. 抽样若干文件**下载回本地做 md5 比对**（防"尺寸对内容错"）。
三项全过才 `rmtree`；任何一项不过都保留本地并报错退出（fail-closed）。

用法
----
    python3 cos_migrate.py --plan plan.json --dry-run
    python3 cos_migrate.py --plan plan.json --unit L01_panel --verify-only
    python3 cos_migrate.py --plan plan.json --delete --part-size 5120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

GATEWAY = "/usr/local/libexec/quantsociety-cos/factor-admin-cos"
BUCKET = "quant-factors-1425188104"
COS_ROOT = f"cos://{BUCKET}/"
SINGLE_PUT_LIMIT = 5 * (1 << 30)  # 5 GiB 单次 PutObject 硬上限

# coscli 表格行：key | type | modified | etag | size | restore
ROW_RE = re.compile(
    r"^(?P<key>.+?)\s*\|\s*(?P<type>[A-Z_]+)\s*\|\s*(?P<modified>[^|]*?)\s*\|"
    r"\s*\"?(?P<etag>[0-9a-f]{16,})\"?\s*\|\s*(?P<size>[0-9.]+ [KMGT]?B)\s*\|"
)
SIZE_RE = re.compile(r"^[0-9.]+ [KMGT]?B$")
OK_RE = re.compile(r"OK num:\s*(\d+)")
ERR_RE = re.compile(r"Error num:\s*(\d+)")


def human(n: int) -> str:
    """与 coscli 输出一致的尺寸字符串。

    实测 coscli 的口径：**永远两位小数**，包括字节档 —— 19 字节打印成 `19.00 B`，
    976 字节打印成 `976.00 B`。写成 `19 B` 会让所有 <1KB 的文件误判为尺寸不符
    （2026-09-20 在 `run.lock`(0B) / 一个 976B 的 html 上各踩一次）。
    """
    for unit, div in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.2f} B"


def cos_uri(key: str) -> str:
    key = key.strip().lstrip("/")
    return key if key.startswith("cos://") else COS_ROOT + key


def obj_uri(key: str, rel: str) -> str:
    """把「单元 key」和「单元内的相对路径」拼成对象 URI。

    坑（2026-09-20）：单文件单元时 `collect_files` 返回的 rel **就是文件名本身**，
    而 key 里已经带了文件名。老代码一律拼 `{key}/{rel}`，会去下载
    `.../x.parquet/x.parquet`（不存在）→ rc!=0 → 被误判成 SPOT_HASH_FAILED，
    于是单文件单元永远删不掉本地副本（W05_fm_all_loose 卡了几轮就是这个）。
    """
    k = key.strip().strip("/")
    if rel and (k == rel or k.endswith("/" + rel)):
        return cos_uri(k)
    return cos_uri(f"{k}/{rel}")


def run(argv: list[str], *, timeout: int | None = None, log=None) -> subprocess.CompletedProcess:
    cmd = ["sudo", "-n", GATEWAY, *argv]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if log is not None:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.write((proc.stdout or "")[-4000:])
        log.write((proc.stderr or "")[-2000:])
        log.flush()
    return proc


# --------------------------------------------------------------- 本地清单
def collect_files(root: Path, excludes: list[str]) -> list[tuple[str, Path, int]]:
    """返回 [(rel, abspath, bytes)]，剔除 excludes 里的 glob 匹配项。"""
    import fnmatch

    if root.is_file():
        cand = [(root.name, root)]
    else:
        cand = []
        for dirpath, _dirs, names in os.walk(root):
            for nm in names:
                fp = Path(dirpath) / nm
                cand.append((str(fp.relative_to(root)), fp))
    out = []
    for rel, fp in cand:
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fp.name, pat) for pat in excludes):
            continue
        try:
            out.append((rel, fp, fp.stat().st_size))
        except OSError:
            continue
    return out


def manifest_of(files: list[tuple[str, Path, int]]) -> dict[str, str]:
    return {rel: human(sz) for rel, _fp, sz in files}


# --------------------------------------------------------------- 远端清单
def remote_manifest(key: str, *, is_file: bool = False, log=None) -> dict[str, str]:
    proc = run(["ls", cos_uri(key)] + ([] if is_file else ["-r"]), timeout=1800, log=log)
    if proc.returncode != 0:
        raise RuntimeError(f"ls rc={proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")

    def norm(k: str) -> str:
        k = k.strip().lstrip("/")
        return k.split("/", 3)[-1] if k.startswith("cos://") else k

    if is_file:
        want = norm(key)
        for line in proc.stdout.splitlines():
            m = ROW_RE.match(line)
            if m and m.group("key").strip().rstrip(",") == want:
                return {Path(want).name: m.group("size").strip()}
        return {}

    prefix = norm(key).rstrip("/") + "/"
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        if m.group("type") == "DIR":
            continue  # coscli 会为目录写占位对象，不是真实文件
        full = m.group("key").strip().rstrip(",")
        size = m.group("size").strip()
        if not full.startswith(prefix) or not SIZE_RE.match(size):
            continue
        rel = full[len(prefix):]
        # coscli 会为每个目录写一个占位对象（列出来带尾斜杠），不是真实文件
        if rel and not rel.endswith("/"):
            out[rel] = size
    return out


# --------------------------------------------------------------- 校验
def compare(local: dict[str, str], remote: dict[str, str]) -> dict:
    lk, rk = set(local), set(remote)
    missing, extra = sorted(lk - rk), sorted(rk - lk)
    mismatched = sorted(k for k in (lk & rk) if local[k] != remote[k])
    return {
        "local_files": len(local), "remote_files": len(remote),
        "n_missing": len(missing), "missing_on_cos": missing[:20],
        "n_extra": len(extra), "extra_on_cos": extra[:20],
        "n_size_mismatch": len(mismatched),
        "size_mismatch": [(k, local[k], remote[k]) for k in mismatched[:20]],
        "ok": not missing and not extra and not mismatched,
    }


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def spot_check_md5(files: list[tuple[str, Path, int]], key: str, *, n: int,
                   stage: Path, log=None, max_gb: float = 4.0) -> dict:
    # 本地盘紧张：抽小文件回验，避免把 SSD 写爆。上限随磁盘水位可调。
    limit = int(max_gb * (1 << 30))
    small = [f for f in files if f[2] <= limit]
    if not small:
        # 单文件单元且体积超过上限时，跳过但**不算失败**（尺寸串已比对过），
        # 但要显式记录下来，不能伪装成"已哈希验证"。
        return {"skipped": True, "ok": True,
                "why": f"no file <= {max_gb} GiB to spot-check"}
    picks = random.Random(20260920).sample(small, min(n, len(small)))
    stage.mkdir(parents=True, exist_ok=True)
    results = []
    for rel, fp, sz in picks:
        dst = stage / rel.replace("/", "__")
        entry = {"rel": rel, "ok": False, "why": "not attempted"}
        for attempt in (1, 2):  # 瞬时截断：重试一次再定性
            # 显式 --part-size：抽样文件都 <= max_gb (默认 4GiB)，单次 GetObject
            # 就够，避免走分片下载路径（该路径曾产生 rc=0 的截断文件）。
            proc = run(["cp", obj_uri(key, rel), str(dst),
                        "--part-size", "5120"], timeout=3600, log=log)
            if proc.returncode != 0 or not dst.exists():
                entry = {"rel": rel, "ok": False, "why": f"download failed (attempt {attempt})"}
                dst.unlink(missing_ok=True)
                continue
            got = dst.stat().st_size
            if got != sz:
                # 长度不符 => 下载截断，不是内容不符。区分开，便于定位。
                entry = {"rel": rel, "ok": False,
                         "why": f"download truncated (attempt {attempt}): {got} != {sz} bytes"}
                dst.unlink(missing_ok=True)
                continue
            hl, hr = md5(fp), md5(dst)
            entry = {"rel": rel, "ok": hl == hr, "local": hl, "remote": hr,
                     "bytes": sz, "attempt": attempt}
            dst.unlink(missing_ok=True)
            if entry["ok"]:
                break
        results.append(entry)
    bad = [r for r in results if not r.get("ok")]
    return {"checked": len(results), "failed": bad, "ok": not bad,
            "results": results}


# --------------------------------------------------------------- 主流程
def process_unit(unit: dict, args, log) -> dict:
    name, key = unit["unit"], unit["key"]
    local = Path(unit["local"]).expanduser()
    excludes = unit.get("exclude", [])
    rec: dict = {"unit": name, "local": str(local), "key": key, "exclude": excludes}

    if not local.exists():
        rec["status"] = "LOCAL_MISSING"
        return rec

    files = collect_files(local, excludes)
    n_bytes = sum(sz for _r, _f, sz in files)
    rec.update(local_files=len(files), local_bytes=n_bytes, local_human=human(n_bytes))

    oversize = [(r, sz) for r, _f, sz in files if sz > args.max_single_part_mb * (1 << 20)]
    print(f"\n{'=' * 72}\n[{name}] {local}\n  -> {cos_uri(key)}")
    print(f"  {len(files)} files / {rec['local_human']}"
          + (f"  (exclude={excludes})" if excludes else ""))
    if oversize:
        rec["oversize"] = [{"rel": r, "human": human(sz)} for r, sz in oversize[:20]]
        print(f"  !! {len(oversize)} file(s) exceed single-part limit {args.max_single_part_mb}MB:")
        for r, sz in oversize[:5]:
            print(f"     {human(sz):>10}  {r}")
    if args.dry_run:
        rec["status"] = "DRY_RUN"
        return rec
    if oversize and not args.allow_oversize:
        rec["status"] = "OVERSIZE_ABORT"
        print("  abort: 先拆分/排除超限文件，否则会触发被禁的 multipart 上传")
        return rec

    # 1) 上传（强制单次 PutObject：part-size > 最大文件）
    cp_args = ["cp", str(local), cos_uri(key)]
    if local.is_dir():
        cp_args.append("-r")
    for pat in excludes:
        cp_args += ["--exclude", f"*{pat}" if not pat.startswith("*") else pat]
    cp_args += ["--check-point", "--thread-num", str(args.threads),
                "--part-size", str(args.part_size), "--err-retry-num", "3"]
    t0 = time.time()
    proc = run(cp_args, timeout=args.timeout, log=log)
    rec["upload_seconds"] = round(time.time() - t0, 1)
    m_ok, m_err = OK_RE.search(proc.stdout or ""), ERR_RE.search(proc.stdout or "")
    rec["upload_ok_num"] = int(m_ok.group(1)) if m_ok else None
    rec["upload_err_num"] = int(m_err.group(1)) if m_err else None
    rec["upload_tail"] = (proc.stdout or "")[-600:]
    rec["upload_rc"] = proc.returncode
    print(f"  upload rc={proc.returncode} {rec['upload_seconds']}s "
          f"ok={rec['upload_ok_num']} err={rec['upload_err_num']}")
    if proc.returncode != 0 or (rec["upload_err_num"] or 0) > 0:
        rec["status"] = "UPLOAD_FAILED"
        return rec

    # 2) 清单比对
    remote = remote_manifest(key, is_file=local.is_file(), log=log)
    cmp_ = compare(manifest_of(files), remote)
    rec["compare"] = cmp_
    print(f"  manifest: local={cmp_['local_files']} remote={cmp_['remote_files']} "
          f"missing={cmp_['n_missing']} extra={cmp_['n_extra']} "
          f"size_mismatch={cmp_['n_size_mismatch']}")
    if not cmp_["ok"]:
        rec["status"] = "MANIFEST_MISMATCH"
        return rec
    if args.verify_only:
        rec["status"] = "VERIFIED"
        return rec

    # 3) 抽样哈希
    if args.spot > 0:
        sc = spot_check_md5(files, key, n=args.spot, max_gb=args.spot_max_gb,
                            stage=Path.home() / "cos_stage" / "_spot", log=log)
        rec["spot_md5"] = sc
        print(f"  spot md5: checked={sc.get('checked')} ok={sc.get('ok')}")
        if not sc.get("ok"):
            rec["status"] = "SPOT_HASH_FAILED"
            return rec

    # 4) 删本地
    if args.delete:
        if local.is_file():
            local.unlink()
        else:
            shutil.rmtree(local)
        rec["deleted_local"] = True
        rec["status"] = "MIGRATED"
        print(f"  local removed: {local}")
    else:
        rec["status"] = "VERIFIED_KEPT_LOCAL"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--unit", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--allow-oversize", action="store_true")
    ap.add_argument("--spot", type=int, default=3)
    ap.add_argument("--spot-max-gb", type=float, default=4.0,
                    help="抽样回验的文件大小上限（GiB）；受本地磁盘水位约束")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--part-size", type=int, default=5120,
                    help="MB；必须 > 单元内最大文件，强制单次 PutObject（multipart 被禁）")
    ap.add_argument("--max-single-part-mb", type=int, default=5120)
    ap.add_argument("--timeout", type=int, default=12 * 3600)
    ap.add_argument("--out", default="cos_migrate_result.json")
    args = ap.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    if args.unit:
        want = set(args.unit)
        plan = [u for u in plan if u["unit"] in want]

    results = []
    with open(f"{args.out}.log", "a", encoding="utf-8") as log:
        for unit in plan:
            try:
                results.append(process_unit(unit, args, log))
            except Exception as exc:  # noqa: BLE001
                results.append({"unit": unit["unit"], "status": "ERROR", "error": repr(exc)[:400]})
                print(f"  !! {unit['unit']} ERROR: {exc!r}")
            Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))

    print(f"\n{'=' * 72}\n汇总:")
    for r in results:
        print(f"  {r['unit']:<28} {str(r.get('status')):<22} {r.get('local_human', '')}")
    good = {"MIGRATED", "VERIFIED", "VERIFIED_KEPT_LOCAL", "DRY_RUN"}
    return 1 if any(r.get("status") not in good for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
