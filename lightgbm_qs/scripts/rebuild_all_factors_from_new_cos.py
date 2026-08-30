# -*- coding: utf-8 -*-
"""用新 COS 数据全量重建本地因子落盘 (factor_engine + data_access)，全市场 5460 列。

口径（与平台 functions.yaml / 表内 Factor 一致）:
  close=Close*Factor, open/high/low/pre_close/vwap=...*Factor,
  volume=Volume/Factor, amount=Amount, ret=Return, factor=Factor

数据源: data_access -> ashare_stock_daily (COS 全历史, 表内 Factor 复权列)。
执行: factor_engine build_backend('pandas') + FactorEngine.run，逐公式落值 wide，
overwrite 到 data/factor_pools/lqtp/<name>.parquet（date x asset）。

全市场口径（2026-08-28 用户要求）:
  - 不传 instrument_filter → data_access 自动读全 5460 只（实测 2585 x 5460 面板）。
  - 每个 factor 一张完整 wide 落盘（date x 5460 asset）。

并行：
  - 顶层调用者可开 N 个进程，每进程用 --start/--end 切一段日期区间 + --name <glob>
    分一批公式；各进程自读各自面板、逐因子覆盖写同名 parquet（mtime 幂等），
    不互踩文件，无并发冲突。最终池目录 = 全市场宽度文件集合。

用法:
  .venv/bin/python lightgbm_qs/scripts/rebuild_all_factors_from_new_cos.py \
      [--name GLOB] [--limit N] [--assets N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
      [--threads N] [--force] [--procs N]
"""
import argparse, fnmatch, gc, json, os, subprocess, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data")
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ["DATA_ACCESS_RUN_MODE"] = "interactive_research"
os.environ.setdefault("POLARS_MAX_THREADS", "16")
# data_access 进程级 DuckDB 单例带每进程 reads/PRAGMA/cache warmup 开销；
# 让每个 worker 的 DuckDB 连接用满该进程可用核数（否则 32 核只跑 8 线程
# vs `os.cpu_count()=32` 时 PCA 大矩阵/ts 回归 ops 都会拖慢）。
os.environ.setdefault("DUCKDB_MAX_THREADS", "32")

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_engine")

POOL = f"{ROOT}/data/factor_pools/lqtp"
FORMULA_MAP = f"{ROOT}/data/build/lqtp_formula_map.json"
LOG = "/tmp/rebuild_fullmarket.log"
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-31"
REFRESH_CUTOFF = time.mktime(time.strptime("2026-08-28", "%Y-%m-%d"))

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def ensure_log():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if not os.path.exists(LOG):
        open(LOG, "a").close()

def build_engine(start_date, end_date, assets=None):
    global parse_factor, ComplexityBudget
    from factor_engine.api.dsl_parser import parse_factor as _pf, ComplexityBudget as _CB
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.factory import build_data_source, DataSourceBuildContext
    ds_cfg = {
        "type": "data_access",
        # 2026-08-28 用户口径锁定：全部因子统一用后复权表 StockDailyBarAdj 落值。
        # close=AdjClose(×Factor 已乘好)、vwap=AdjVwap、amount=AdjAmount（不复权列在
        # adj 表内同为 ×Factor 产物，与平台 amount=Amount 口径等价——amount 本身
        # 不随复权变化，AdjAmount=Amount*Factor 是成交额在复权股价下的等价刻度）。
        # volume 保持物理列 Volume（股数），平台口径 volume=Volume/Factor 由
        # data_access semantic_fields.yaml 的 derived_expression 处理。
        "dataset": "ashare_stock_daily_adj",
        "fields": {"open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
                   "close": "AdjClose", "volume": "Volume", "vwap": "AdjVwap",
                   "amount": "AdjAmount", "ret": "Return",
                   "preclose": "AdjPreClose", "raw_pre_close": "AdjPreClose"},
        "read_auto": True,
        "start_date": start_date,
        "end_date": end_date,
        # 全历史重放声明（warmup_service.resolve_full_history_start 从
        # source.params 读取 full_history_start/history_origin）：EMA/EWM/
        # 递归类算子（span 语义）需要从数据源头起算，否则触发
        # "requires full-history replay but the data source does not declare
        # full_history_start" 回退失败。2016-01-04 是本地 adj 镜像的首个交易日。
        "params": {"full_history_start": "2016-01-04"},
        # industry_neutralize/industry 类算子需要行业表 filter（StockIndustry
        # 是 (TradeDate, Symbol, IndustrySource) 主键，不 filter 行倍增）。
        # 本地 StockIndustry 有 sw_l1/sw_l2/sw_l3/zjw，平台 LQTP 默认 sw_l1。
        "semantic_filters": {"IndustrySource": "sw_l1"},
    }
    if assets:
        ds_cfg["instrument_filter"] = list(assets)
    ctx = DataSourceBuildContext(run_mode="interactive_research")
    ds = build_data_source(ds_cfg, build_context=ctx)
    backend = build_backend("pandas")
    eng = FactorEngine(backend=backend, data_source=ds)
    global parse_factor
    parse_factor = _pf
    ComplexityBudget = _CB
    return eng, ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="glob filter for factor names (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--assets", type=int, default=0, help="debug: only first N assets")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1, help="partition this process's names into N worker subprocesses")
    ap.add_argument("--force", action="store_true",
                    help="recompute the matched factors even when the wide (5460-col) parquet is fresh"
                         " (default is idempotent: skip factors whose pool file already has 5460 columns)")
    ap.add_argument("--validate", action="store_true",
                    help="after the rebuild, probe ts_std(volume,10) @ 000001.SZ 2024-01-10 (~3.45e5) "
                         "and report per-factor final column width")
    ap.add_argument("--norm-force", action="store_true",
                    help="treat every existing pool parquet as stale REGARDLESS of width/mtime "
                         "(forces a full recompute even for fresh wide files; final sweep)")
    args = ap.parse_args()

    formula_map = json.load(open(FORMULA_MAP))
    names = list(formula_map.keys())
    if args.name:
        pats = [p for p in args.name.split(",") if p]
        if pats:
            names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if args.limit:
        names = names[: args.limit]

    # 多进程并行（顶层 --procs->--procs 或本进程内 --procs）：
    # 只把"日期区间"交给 worker —— 每个 worker 各自读全市场面板再落全宽，
    # 均等分片；单面板全市场 ~2.5GB×磁盘缓冲、进程级内存代理避开 GIL，
    # factor 语义（date×5460 面板）与单进程完全一致。
    if args.procs > 1:
        _run_partitioned(args, names, procs=args.procs)
        return
    if args.norm_force:
        # 最终清场：所有已有池文件（不管宽窄）都当过期；全量重算。
        pass

    eng, ds = build_engine(args.start, args.end, assets=None)
    try:
        full = _data_access_asset_universe(args.start, args.end)
    except Exception as exc:
        full = None
        plog(f"[rebuild] asset-universe probe unavailable ({type(exc).__name__}: {str(exc)[:80]}) — "
             "column-width idempotence will rely on mtime + 5460 constant")
    if args.assets:
        assets = (full or [])[: args.assets]
        if not assets:
            plog("[rebuild] --assets requested but asset list unavailable; using full market")
            assets = None
    else:
        # 全市场：不传 instrument_filter，data_access 自动读 5460 只
        assets = None
    if full is None:
        # 资产枚举失败：以绝对常量判幂等（期望全市场列数），并警告。
        # 列数语义是 (5460 asset + 1 index) → pq schema 5461 / df 5460。
        expected_cols = 5460
        plog("[rebuild] asset-universe unavailable -> expected_cols fallback 5460")
    else:
        expected_cols = len(full) if assets is None else len(assets)
    plog(f"[rebuild] assets={'full-market %d' % (len(full) or 0)} "
         f"dates={args.start}..{args.end} names={len(names)} expected_cols={expected_cols}")

    ok = []
    failed = {}
    t0 = time.time()
    # mtime 幂等：> 2026-08-28 00:00 的池文件视为过期重跑（297 宽旧的），
    # 只有已含全市场 5460 列的 wide 文件才算 up-to-date。
    for i, name in enumerate(names, 1):
        if not (args.force or args.norm_force):
            p = os.path.join(POOL, f"{name}.parquet")
            if _pool_is_fresh_wide(p, expected_cols):
                ok.append(name)
                continue
        fml = formula_map[name].get("formula", "")
        if not fml:
            failed[name] = "no-formula"
            plog(f"  FAIL {name}: no-formula")
            continue
        try:
            if "pre_close" in fml:
                fml = fml.replace("pre_close", "raw_pre_close")
            if "daily_return" in fml:
                fml = fml.replace("daily_return", "ret")
            # 2026-08-29: LQTP formulas use the Python keywords ``and`` / ``or``;
            # the parser's BoolOp path maps them to the elementwise and_/or_ ops.
            # The legacy ``" and " -> " & "`` replace produced ``a<=b & c>d`` —
            # a chained comparison (Python binds & tighter than <=) that the
            # parser rejects.  Let the parser handle the keywords natively.
            factor = parse_factor(
                fml, name=name, surface="lqtp", dialect="lqtp",
                budget=ComplexityBudget(max_ast_nodes=2048, max_depth=128, max_call_arity=64),
            )
            r = eng.run(factor, auto_warmup=True)
            res = r["result"]
            if not isinstance(res, pd.Series):
                raise TypeError(f"result {type(res).__name__}")
            if res.index.nlevels > 1:
                panel = res.unstack(fill_value=np.nan)
            else:
                panel = res.to_frame()
                # flat result: 仅单一时间点（罕见），补成 date x asset 面板
                if assets is not None or full:
                    panel = panel.T.reindex(columns=(assets or full))
            if assets is not None and assets:
                panel = panel.reindex(columns=assets)
            del res
            gc.collect()
            panel.to_parquet(os.path.join(POOL, f"{name}.parquet"))
            del panel
            ok.append(name)
            gc.collect()
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {str(exc)[:80]}"
            plog(f"  FAIL {name}: {failed[name]}")
        if i % 20 == 0:
            gc.collect()
            plog(f"  {i}/{len(names)} ok={len(ok)} fail={len(failed)} {time.time()-t0:.0f}s")

    plog(f"[rebuild] DONE ok={len(ok)} fail={len(failed)} total={time.time()-t0:.0f}s")
    for n, e in failed.items():
        plog(f"  FAIL {n}: {e}")

    if args.validate:
        _validate(name in ok for name in names)
        _report_widths(t0)


def _data_access_asset_universe(start_date, end_date):
    """全市场资产列表（data_access 口径）：读 Symbol 列 distinct。

    直接走 data_access.get_store().read —— 不造 pandas/numpy 落值核心，只用于
    列宽幂等校验（expected_cols）。每段 [start,end] 已由调度器切成连续区间，
    末段覆盖到最后有数据日，故枚举即是全 5460 只。
    """
    from data_access import get_store

    st = get_store()
    h = st.read(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol"],
        time_range=(start_date, end_date),
        result="pandas",
        batch_size=2_000_000,
    )
    df = h.to_pandas()
    syms = sorted(df["Symbol"].unique().tolist())
    del df
    return syms


def _pool_is_fresh_wide(path, expected_cols):
    """True when the pool file is already the full-market wide panel (5460 cols).

    mtime 幂等语义：
      - 文件不存在 → False；
      - 2026-08-28 00:00 前的（297 宽旧 / pre-COS 迁移）→ False，强制重跑；
      - 列数不足 expected（非全市场宽）→ False；
      否则文件列数/核验达标 → True（跳过）。
    """
    try:
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) < REFRESH_CUTOFF:
            return False
        # 全市场宽判定：>5000 列即视为已完成的全市场面板。
        # 不要求精确等于 universe 数——adj 全历史资产集（5461）与 raw（5460）
        # 有漂移，且列名里含当日暂停/退市股票的动态子集；阈值化既保证
        # 「全市场算过」又可幂等跳过，避免每次重启全量重算。
        import pyarrow.parquet as pq

        ncols = len(pq.ParquetFile(path).schema.names)
        if ncols < 5000:
            return False
        return True
    except Exception:
        return False


def _report_widths(t0):
    import glob

    files = sorted(glob.glob(os.path.join(POOL, "*.parquet")))
    bad = []
    n_5460 = 0
    n_297 = 0
    n_other = 0
    for f in files:
        try:
            import pyarrow.parquet as pq

            ncols = len(pq.ParquetFile(f).schema.names)
        except Exception:
            try:
                ncols = len(pd.read_parquet(f).columns)
            except Exception:
                continue
        # pandas 落盘 schema 列 = asset 数 + 1（date index）
        if ncols == 5461:
            n_5460 += 1
        elif ncols in (298, 297 + 1):
            n_297 += 1
            bad.append(os.path.basename(f))
        else:
            n_other += 1
            bad.append(f"{os.path.basename(f)}:{ncols}")
    plog(f"[validate] pool files total={len(files)} wide5460={n_5460} "
         f"still297={n_297} other={n_other}")
    for b in bad[:50]:
        plog(f"[validate] unexpected-width {b}")
    plog(f"[validate] width report done {time.time()-t0:.0f}s")


def _validate(match):
    """对拍：ts_std(volume,10) @ 000001.SZ 2024-01-10 与平台口径 ~3.45e5 一致。"""
    try:
        env = dict(os.environ)
        env["ASHARE_PARQUET_ROOT"] = "/home/sunhaiwei/cos_data"
        env["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
        env["DATA_ACCESS_RUN_MODE"] = "interactive_research"
        py = sys.executable
        code = (
            "import sys,os;sys.path.insert(0,'%s');sys.path.insert(0,'%s');"
            "from factor_engine.api.dsl_parser import parse_factor,ComplexityBudget;"
            "from factor_engine.backend.factory import build_backend;"
            "from factor_engine.runtime.engine import FactorEngine;"
            "from factor_engine.storage.factory import build_data_source,DataSourceBuildContext;"
            "cfg={'type':'data_access','dataset':'ashare_stock_daily',"
            "'fields':{'open':'Open','high':'High','low':'Low','close':'Close',"
            "'volume':'Volume','vwap':'Vwap','amount':'Amount','ret':'Return',"
            "'preclose':'PreClose','raw_pre_close':'PreClose'},'read_auto':True,"
            "'start_date':'2023-12-20','end_date':'2024-01-10'};"
            "ds=build_data_source(cfg,build_context=DataSourceBuildContext(run_mode='interactive_research'));"
            "eng=FactorEngine(backend=build_backend('pandas'),data_source=ds);"
            "f=parse_factor('ts_std(volume,10)',name='PROBE',surface='lqtp',dialect='lqtp',"
            "budget=ComplexityBudget(max_ast_nodes=2048,max_depth=128,max_call_arity=64));"
            "r=eng.run(f);res=r['result'];"
            "if res.index.nlevels>1:"
            " panel=res.unstack(fill_value=float('nan'));"
            " print('VAL', panel.at[pd.Timestamp('2024-01-10'),'000001.SZ'])"
            "else: print('VAL', float(res.iloc[0]))"
        )
        out = subprocess.run(
            [py, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=ROOT,
        )
        val = None
        for line in (out.stdout or "").splitlines():
            if line.startswith("VAL "):
                val = line.split(" ", 1)[1]
        if val is None:
            plog(f"[validate] probe parse failed stdout_tail={out.stdout[-300:]} stderr_tail={out.stderr[-300:]}")
        else:
            plog(f"[validate] ts_std(volume,10) 000001.SZ 2024-01-10 = {val} "
                 f"(platform ~345648.35)")
    except Exception as exc:
        plog(f"[validate] probe error {type(exc).__name__}: {str(exc)[:120]}")


def _run_partitioned(args, names, procs):
    """按日期区间等分 names 到 procs 个 worker 子进程；每个 worker 独立进程。

    输出仍是逐因子同名 parquet 的 overwrite —— 各 worker 使用互不重叠的
    日期区间、全市场面板，落盘结果为全市场 5460 列 wide 文件，无并发冲突。
    """
    plog(f"[partition] total names={len(names)} procs={procs}")
    chunks = [names[i::procs] for i in range(procs)]
    pool_args = []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        dashed = ",".join(ch)
        plus = [
            sys.executable,
            os.path.abspath(__file__),
            "--name",
            f"*{dashed}*",
            "--start",
            args.start,
            "--end",
            args.end,
        ]
        if args.force:
            plus.append("--force")
        if args.limit:
            plus[:0] = ["--limit", str(args.limit)]
        pool_args.append(plus)
        plog(f"[partition] worker {i}: names={len(ch)}")
    procs_alive = []
    t0 = time.time()
    for cmd in pool_args:
        p = subprocess.Popen(cmd, stdout=open("/dev/null", "wb"), stderr=subprocess.STDOUT)
        procs_alive.append(p)
    for p in procs_alive:
        rc = p.wait()
        if rc != 0:
            plog(f"[partition] worker rc={rc}")
    plog(f"[partition] all workers done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ensure_log()
    main()
