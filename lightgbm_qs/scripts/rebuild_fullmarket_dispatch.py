# -*- coding: utf-8 -*-
"""全市场 LQTP 因子重建调度器（多进程，日期切分 × 公式等分，内存不重）。

职责：
  1. 把 formula_map 的 1414 个因子名按 procs 等分。
  2. 把日期 [2016-01-01, 2026-08-21] 切成 shreds 段（每段连续，含 1 天重叠，
     保证跨段公式窗口（warmup）内存一致）。切分段的公式窗口长度不能大于
     overlap（默认 260 个交易日，覆盖 1 年 warmup 因子的需求）。
  3. 每个 worker： rebuild_all_factors_from_new_cos.py
       --name  'glob1,glob2,...'  --start <s>  --end <e>  [--force]
     worker 自身读全市场面板、逐因子覆盖写同名 parquet（全宽 5460 列），
     输出互不冲突、mtime 幂等。
  4. 顶层顺序等待（不叠加 worker 进程数）——单面板全市场 5460×2585 约数 GB、
     进程独立内存，避免同机多进程巨面板并发导致内存/IO 踩踏。
"""
import argparse, json, os, subprocess, sys, time


ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
PY = "/home/sunhaiwei/quant_projects/.venv/bin/python"
SCRIPT = os.path.join(ROOT, "scripts", "rebuild_all_factors_from_new_cos.py")
FORMULA_MAP = os.path.join(ROOT, "data", "build", "lqtp_formula_map.json")
LOG = "/tmp/rebuild_fullmarket.log"

T0 = time.time()


def plog(line):
    elapsed = time.time() - T0
    print(f"[{elapsed:7.0f}s] {line}", flush=True)
    with open(LOG, "a") as f:
        f.write(f"[{elapsed:7.0f}s] {line}\n")


def markets_shreds(start, end, shreds, trading_days, overlap=260):
    """把 [start, end] 的交易日序列均分成 shreds 段（相邻段 overlap 天重叠）。

    返回 [(seg_start, seg_end), ...]，seg 都是实际交易日。shreds==1 时返回
    [(start_df, end_df)]（end 取日历边界，data_access 读满即可）。
    """
    if shreds <= 1:
        return [(_fmt_day(start), _fmt_day(end))]
    # 取一段足够覆盖 [start..end] 的实际交易日（未来日期也当 trading day，
    # 反正数据只在确有交易的日期有行）。
    days = trading_days(start, end, count=shreds * 2 + overlap * 2)
    segs = []
    total_days = len(days)
    if total_days < 1:
        return [(str(start), str(end))]
    # 每段大约 表示 总交易日/shreds 天；留出 overlap 尾巴。
    step = max(1, (total_days - overlap) // max(1, shreds))
    for i in range(shreds):
        lo = min(total_days - 1, i * step)
        hi = min(total_days - 1, lo + step + overlap)
        if i == shreds - 1:
            hi = total_days - 1
        if i == 0:
            lo = 0
        segs.append((str(days[lo]), str(days[hi])))
    # 去重相邻相同段
    out = []
    for seg in segs:
        if not out or out[-1] != seg:
            out.append(seg)
    return out


def _fmt_day(d):
    import datetime as _dt

    if isinstance(d, str):
        return d
    if isinstance(d, _dt.date):
        return d.strftime("%Y-%m-%d")
    try:
        import pandas as pd

        if isinstance(d, pd.Timestamp):
            return d.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(d)


def trading_days(start, end, count):
    """basic 交易日序列：从 pandas 日期范围近似（全市场每日非周末即交易日）。
    足够切分任务用；不精确也不影响正确性（段端在无数据日期也只是空段）。"""
    import pandas as pd

    rng = pd.bdate_range(start=start, end=end, freq="B")
    if len(rng) < count:
        # 扩展到更远未来（未来无数据 → 数据范围的最后一天仍会被覆盖到）
        rng = pd.bdate_range(start=start, periods=count, freq="B")
    return [d.strftime("%Y-%m-%d") for d in rng[:count]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8, help="公式等分 worker 数")
    ap.add_argument("--shreds", type=int, default=6, help="日期区间切段数（沿至今最后日期）")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--force", action="store_true", help="强制重算（默认对已全宽的跳过）")
    ap.add_argument("--norm-force", action="store_true",
                    help="把每个已有池文件都当过期（宽窄一律重算；最终清场）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发同时跑的 worker 数（默认 1，串行——避免多进程巨面板内存踩踏）")
    args = ap.parse_args()

    names = list(json.load(open(FORMULA_MAP)).keys())
    plog(f"[dispatch] names={len(names)} procs={args.procs} shreds={args.shreds} "
         f"dates={args.start}..{args.end} workers={args.workers} force={args.force}")

    segs = markets_shreds(args.start, args.end, args.shreds, trading_days,
                          overlap=min(260, max(1, 2585 // args.shreds)))
    plog(f"[dispatch] date segments={len(segs)}")
    for i, (s, e) in enumerate(segs):
        plog(f"[dispatch]   seg{i}: {s} .. {e}")

    level2 = (args.procs > 1 and args.shreds > 1)
    tasks = []
    if args.shreds <= 1 and args.procs > 1:
        # 只有公式切分：单日期区间，worker 各自读全市场整段（内存 2× 冗余但语义正确）
        chunks = [names[i::args.procs] for i in range(args.procs)]
        for i, ch in enumerate(chunks):
            if not ch:
                continue
            tasks.append((ch, args.start, args.end, f"w{i}"))
    else:
        for si, (s, e) in enumerate(segs):
            if args.procs <= 1:
                tasks.append((names, s, e, f"seg{si}"))
            else:
                chunks = [names[i::args.procs] for i in range(args.procs)]
                for i, ch in enumerate(chunks):
                    if not ch:
                        continue
                    tasks.append((ch, s, e, f"seg{si}w{i}"))

    plog(f"[dispatch] tasks={len(tasks)}")

    sem = args.workers
    t0 = time.time()
    running = []
    next_task = 0
    failed = []
    while next_task < len(tasks) or running:
        while len(running) < sem and next_task < len(tasks):
            ch, s, e, tag = tasks[next_task]
            next_task += 1
            cmd = [
                PY, SCRIPT,
                "--name", ",".join(ch),
                "--start", s, "--end", e,
            ]
            if args.force:
                cmd.append("--force")
            if getattr(args, "norm_force", False):
                cmd.append("--norm-force")
            plog(f"[dispatch] spawn {tag} start={s} end={e} names={len(ch)} "
                 f"({'norm-force' if args.norm_force else 'force' if args.force else 'idem'})")
            p = subprocess.Popen(
                cmd, stdout=open("/dev/null", "wb"), stderr=subprocess.STDOUT,
            )
            running.append((p, tag, cmd))
        if not running:
            break
        pp, tag, cmd = running.pop(0)
        rc = pp.wait()
        el = time.time() - t0
        if rc != 0:
            failed.append(tag)
            plog(f"[dispatch] worker {tag} FAILED rc={rc} ({el:.0f}s elapsed)")
            # 重试一次同任务
            ch, s, e = cmd[cmd.index("--name") + 1], cmd[cmd.index("--start") + 1], cmd[cmd.index("--end") + 1]
            retry = [PY, SCRIPT, "--name", ch, "--start", s, "--end", e]
            if "--force" in cmd:
                retry.append("--force")
            if "--norm-force" in cmd:
                retry.append("--norm-force")
            plog(f"[dispatch] retry {tag}")
            rp = subprocess.Popen(retry, stdout=open("/dev/null", "wb"), stderr=subprocess.STDOUT)
            rc2 = rp.wait()
            if rc2 != 0:
                failed.append(f"{tag}:retry")
                plog(f"[dispatch] worker {tag} retry FAILED rc={rc2}")
            else:
                plog(f"[dispatch] worker {tag} retry OK")
        else:
            plog(f"[dispatch] worker {tag} ok ({el:.0f}s elapsed)")

    plog(f"[dispatch] DONE failed={failed} total={time.time()-t0:.0f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()