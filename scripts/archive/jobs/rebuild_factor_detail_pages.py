#!/usr/bin/env python3
"""
重建本周61个因子的详情页：
1. 从 factor_delivery_converted 提取公式（自动翻转IC<0的）
2. 生成 IC月度热力图 / 十分层 / 多空NAV（matplotlib PNG）
3. IC时序用内嵌SVG（无需CDN）
4. 中性化前后RankIC/RankICIR
5. 落值：用 weekly_backtest_output/factor_values.parquet（如已落好）
"""

from __future__ import annotations
import sys, os, json, re, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import io, base64

# ---- 中文字体（Noto Sans CJK SC）----
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
if os.path.exists(_CN_FONT):
    try:
        import matplotlib.font_manager as _fm
        _fm.fontManager.addfont(_CN_FONT)
        _CN_NAME = _fm.FontProperties(fname=_CN_FONT).get_name()
        plt.rcParams["font.sans-serif"] = [_CN_NAME, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

try:
    import duckdb
    _HAS_DUCKDB = True
except Exception:
    _HAS_DUCKDB = False

# 接上 quant_evaluator 评估库
QE_AVAILABLE = False
try:
    from quant_evaluator.metrics.ic import (
        compute_daily_ic,
        compute_mean_ic,
        compute_ic_std,
        _spearman_rank_correlation,
        _pearson_correlation,
    )
    from quant_evaluator.metrics.portfolio_stats import (
        compute_sharpe_ratio,
        compute_maximum_drawdown,
        compute_win_rate,
        compute_long_short_returns,
    )
    from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_series
    from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    QE_AVAILABLE = True
except Exception as _qe_err:
    _QE_IMPORT_ERR = _qe_err

PROJECT = Path("/home/sunhaiwei/quant_projects")
CONV_DIR = Path("/home/sunhaiwei/factor_delivery_converted/factors_combined")
REPORT_DIR = Path(os.environ.get("FACTOR_REPORT_DIR", str(PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23")))
FACTORS_DIR = REPORT_DIR / "factors"
BACKTEST_OUT = PROJECT / "weekly_backtest_output"
OUT_IMGS = REPORT_DIR  # 生成的图放这里

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

# 字段与算子释义补全字典（jobs/field_op_doc.py）——覆盖基础 OHLCV + 衍生字段 + 全部用到的算子
try:
    from jobs.field_op_doc import apply_field_op_docs, DSL_NOISE
    _HAS_FIELD_DOC = True
except Exception:
    apply_field_op_docs = None
    DSL_NOISE = set()
    _HAS_FIELD_DOC = False

# LQTP 转换结果（61 条：dsl / status / can_use_factor_engine / note / is_flipped）
# FACTOR_SET=all 时加载 456 全量转换；否则 61
_FACTOR_SET = os.environ.get("FACTOR_SET", "61").strip()
_LQTP_PATH_ALL = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
_LQTP_PATH_61 = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json")
_LQTP_RECORDS: dict[str, dict] = {}
try:
    _lqtp_path = _LQTP_PATH_ALL if (_FACTOR_SET == "all" and _LQTP_PATH_ALL.exists()) else _LQTP_PATH_61
    if _lqtp_path.exists():
        import json as _json
        _raw = _json.loads(_lqtp_path.read_text())
        if isinstance(_raw, dict):
            # 新格式：dict keyed by 因子名（aacd 版），value 含 lqtp_formula/fe_formula/custom_dsl/status/flipped
            for _k, _rec in _raw.items():
                _rec = dict(_rec)
                _rec.setdefault("page_name", _k)
                _rec["dsl"] = _rec.get("lqtp_formula") or _rec.get("fe_formula") or _rec.get("custom_dsl") or ""
                _rec["status"] = _rec.get("status", "fallback")
                _rec["can_use_factor_engine"] = (_rec.get("status") == "ok")
                _rec["is_flipped"] = bool(_rec.get("flipped", False))
                _LQTP_RECORDS[_k] = _rec
        elif isinstance(_raw, list):
            for _rec in _raw:
                _rec = dict(_rec)
                _LQTP_RECORDS[_rec.get("page_name", "")] = _rec
except Exception:
    _LQTP_RECORDS = {}

# ============================================================
# 配色
# ============================================================
BG = "#eef2f7"
PANEL = "#fff"
FG = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
PRIMARY = "#1e4d8c"
POS = "#16a34a"
NEG = "#dc2626"
PLOT_BG = "#ffffff"

# ============================================================
# 工具函数
# ============================================================
def load_weekly_factor_names() -> list[str]:
    if _FACTOR_SET == "all":
        # 全量：从 factor_matrices_all/ 的文件名（page_name）
        names = [f.stem for f in _FV_DIR.glob("*.parquet")] if _FV_DIR.exists() else []
        return sorted(names)
    names = []
    for f in FACTORS_DIR.glob("factor_*.html"):
        name = f.stem.replace("factor_", "")
        names.append(name)
    return sorted(names)

def find_converted_file(name: str) -> Path | None:
    """查找转换好的因子文件（处理翻转因子）"""
    # 尝试直接匹配
    for basename in [name, name.replace("_flipped", ""), name + "_flipped"]:
        p = CONV_DIR / f"factor_{basename}.json"
        if p.exists():
            return p
    # 尝试带数字后缀
    base = name.replace("_flipped", "")
    for suf in ["_109", "_110", "_116", "_118", "_190", "_246", "_315", "_363"]:
        p = CONV_DIR / f"factor_{base}{suf}.json"
        if p.exists():
            return p
    return None

try:
    from fe_dsl_formula import derive_fe_dsl as _derive_fe_dsl, display_formula_text as _display_fe_dsl
    _HAS_FE_NORMALIZER = True
except Exception:
    _HAS_FE_NORMALIZER = False

try:
    from factor_manual_61 import FACTOR_MANUAL as _FACTOR_MANUAL
except Exception:
    _FACTOR_MANUAL = {}


def extract_formula_info(name: str) -> dict:
    """提取因子公式信息（含原始 Python code + DSL 转换公式）

    优先使用 factor_manual_61.FACTOR_MANUAL (中文手写公式 + 步骤).
    返回字段:
      formula:     原 lqtp formula 字符串 (兜底展示)
      code:        原 Python code
      dsl:         最干净的 FE DSL 表达式 (sign 前缀已加好)
      dsl_status:  'ok' | 'fallback' | 'empty'
      dsl_note:    fallback 时的解释文字
      title:       中文标题 (manual 才有)
      steps:       中文步骤列表 (manual 才有)
      manual:      是否来自手写 manual
      rationale / required_columns / is_flipped
    """
    path = find_converted_file(name)
    if path is None:
        return {"formula": "（未找到公式文件）", "code": "",
                "dsl": "（公式为空）", "dsl_status": "empty",
                "dsl_note": "", "rationale": "",
                "required_columns": "", "is_flipped": "_flipped" in name,
                "title": "", "steps": [], "manual": False}

    with open(path) as f:
        d = json.load(f)

    raw_formula = (d.get("formula") or "").strip()
    code = d.get("code", "")
    rationale = (d.get("rationale") or "")[:400]
    is_flipped = "_flipped" in name

    # ---- 0. 优先用 LQTP 转换结果 (formula_lqtp.json) ----
    lqtp_rec = _LQTP_RECORDS.get(name)
    if lqtp_rec:
        dsl_text = lqtp_rec.get("dsl", "") or ""
        status = lqtp_rec.get("status", "fallback")
        can_fe = lqtp_rec.get("can_use_factor_engine", False)
        note = lqtp_rec.get("note", "")
        lqtp_steps = lqtp_rec.get("steps") or []
        lqtp_title = lqtp_rec.get("title") or ""
        if status == "ok" and can_fe:
            dsl_note = "✅ 已按 LQTP 转换，factor_engine DSL 可直接执行"
        elif status == "fallback":
            dsl_note = "已转 FactorEngine（DSL 语法近似，落值用真实 Python code）"
        else:
            dsl_note = "⚠️ 自命名 DSL，用不了 factor_engine —— " + note
        return {
            "formula": raw_formula or lqtp_rec.get("lqtp_formula", "") or "（未提供原 formula）",
            "code": code,
            "dsl": dsl_text,
            "dsl_status": status,
            "dsl_note": dsl_note,
            "fe_formula": d.get("fe_formula", "") or lqtp_rec.get("fe_formula", "") or "",
            "lqtp_formula": d.get("lqtp_formula", "") or lqtp_rec.get("lqtp_formula", "") or "",
            "rationale": rationale,
            "required_columns": ", ".join(lqtp_rec.get("required_columns") or d.get("required_columns", []) or []),
            "is_flipped": is_flipped or lqtp_rec.get("is_flipped", False),
            "title": lqtp_title,
            "steps": lqtp_steps,
            "manual_note": "",
            "manual": False,
            "can_use_factor_engine": can_fe,
        }

    # ---- 0b. 手写 manual (中文 DSL + 步骤) 兜底 ----
    manual = _FACTOR_MANUAL.get(name)
    if manual:
        manual_dsl = manual["dsl"]
        already_neg = manual_dsl.startswith("-")
        if is_flipped and not already_neg:
            manual_dsl = "-" + manual_dsl
        return {
            "formula": raw_formula or manual["title"],
            "code": code,
            "dsl": manual_dsl,
            "dsl_status": "ok",
            "dsl_note": "（手写公式 — 字段/算子含义见下方）",
            "fe_formula": d.get("fe_formula", "") or "",
            "lqtp_formula": d.get("lqtp_formula", "") or "",
            "rationale": rationale,
            "required_columns": ", ".join(d.get("required_columns", []) or []),
            "is_flipped": is_flipped,
            "title": manual["title"],
            "steps": manual["steps"],
            "manual_note": manual.get("note", ""),
            "manual": True,
        }

    # ---- 1. 用 fe_dsl_formula 推出最干净的 FE DSL ----
    if _HAS_FE_NORMALIZER:
        result = _derive_fe_dsl(d)
        display = _display_fe_dsl(
            result.get("dsl", ""),
            result.get("status", "fallback"),
            is_flipped,
            result.get("note", ""),
            raw_formula,
            d.get("rationale") or "",
            code,
        )
        dsl_text = display["display"]
        dsl_status = display["status"]
        dsl_note = display["explanation"]
    else:
        dsl_text = ("−" if is_flipped else "") + (raw_formula or "（公式为空）")
        dsl_status = "fallback" if raw_formula else "empty"
        dsl_note = ""

    return {
        "formula": raw_formula or "（未提供原 formula）",
        "code": code,
        "dsl": dsl_text,
        "dsl_status": dsl_status,
        "dsl_note": dsl_note,
        "fe_formula": d.get("fe_formula", "") or "",
        "lqtp_formula": d.get("lqtp_formula", "") or "",
        "rationale": rationale,
        "required_columns": ", ".join(d.get("required_columns", []) or []),
        "is_flipped": is_flipped,
        "title": "",
        "steps": [],
        "manual_note": "",
        "manual": False,
    }

def load_factor_values() -> pd.DataFrame | None:
    """加载已落好的因子值"""
    fv_path = BACKTEST_OUT / "factor_values.parquet"
    if not fv_path.exists():
        return None
    try:
        df = pd.read_parquet(fv_path)
        return df
    except Exception:
        return None

def load_ic_series(factor_name: str) -> list[dict]:
    """从详情页HTML里提取IC时序数据"""
    html_path = FACTORS_DIR / f"factor_{factor_name}.html"
    if not html_path.exists():
        return []
    with open(html_path) as f:
        content = f.read()
    match = re.search(r"const icData\s*=\s*(\[.*?\]);", content, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except Exception:
        return []

def load_metrics_from_html(factor_name: str) -> dict:
    """从现有HTML提取已有指标"""
    html_path = FACTORS_DIR / f"factor_{factor_name}.html"
    if not html_path.exists():
        return {}
    with open(html_path) as f:
        content = f.read()
    metrics = {}

    # 提取 grid-4 和 grid-3 的指标
    val_pattern = re.compile(r'<b[^>]*>([-.\d]+)</b><span>(.*?)</span>', re.DOTALL)
    for m in val_pattern.finditer(content):
        val, label = m.group(1), m.group(2).strip()
        metrics[label] = m.group(0)

    # 翻转标记
    metrics["is_flipped"] = "_flipped" in factor_name
    return metrics

# ============================================================
# 指标计算
# ============================================================
def compute_annualized_sharpe(daily_rets: pd.Series, periods_per_year: int = 252) -> float:
    if daily_rets.std() == 0:
        return 0.0
    return (daily_rets.mean() / daily_rets.std()) * math.sqrt(periods_per_year)

def compute_max_drawdown(nav: pd.Series) -> float:
    if len(nav) == 0:
        return 0.0
    nav = nav.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(dd.min())

_HAS_FV_DF = None  # 进程级缓存 parquet
_FV_DIR = (BACKTEST_OUT / "factor_matrices_all") if (_FACTOR_SET == "all") else (BACKTEST_OUT / "factor_matrices")   # 每因子一个 parquet (date × symbol)

def _load_full_fv() -> pd.DataFrame:
    """进程级缓存：从 factor_matrices/ 逐因子加载（新格式，per-factor 文件）。

    返回 MultiIndex columns (factor, symbol) 的宽表；仅为兼容旧接口，
    实际逐因子调用时直接读单文件更省内存。
    """
    global _HAS_FV_DF
    if _HAS_FV_DF is not None:
        return _HAS_FV_DF
    try:
        files = sorted(_FV_DIR.glob("*.parquet")) if _FV_DIR.exists() else []
        if not files:
            # 回退旧宽表
            _HAS_FV_DF = pd.read_parquet(BACKTEST_OUT / "factor_values.parquet",
                                         engine='pyarrow',
                                         thrift_string_size_limit=2**31-1,
                                         thrift_container_size_limit=2**31-1)
            return _HAS_FV_DF
        # 只加载第一个文件的结构作为日期轴（各文件同日期轴）
        first = pd.read_parquet(files[0])
        idx = first.index
        cols = []
        mats = {}
        for fp in files:
            m = pd.read_parquet(fp)
            fac = fp.stem  # 已是 page_name（含 _flipped）
            for sym in m.columns:
                mats[(fac, sym)] = m[sym].values
        df = pd.DataFrame(mats, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["factor", "symbol"])
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        _HAS_FV_DF = df
    except Exception as _e:
        _HAS_FV_DF = pd.DataFrame()
    return _HAS_FV_DF

def _load_factor_matrix(name: str) -> pd.DataFrame:
    """从 per-factor parquet 加载真因子矩阵 (date × symbol)"""
    try:
        fac = f"factor_{name}"
        for cand in [f"{name}.parquet", f"{fac}.parquet", f"{name.replace('_flipped','')}.parquet"]:
            p = _FV_DIR / cand
            if p.exists():
                m = pd.read_parquet(p)
                if not isinstance(m.index, pd.DatetimeIndex):
                    m.index = pd.to_datetime(m.index)
                return m
        # 兜底: 旧宽表
        df = _load_full_fv()
        if df.empty:
            return pd.DataFrame()
        cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == fac]
        if not cols:
            return pd.DataFrame()
        m = df[cols]
        m.columns = [c[1] for c in m.columns]
        return m
    except Exception:
        return pd.DataFrame()

_HAS_CLOSE = None
def _load_close_matrix() -> pd.DataFrame:
    """加载收盘价矩阵 (date × symbol)"""
    global _HAS_CLOSE
    if _HAS_CLOSE is not None:
        return _HAS_CLOSE
    try:
        LOCAL_DAILY = Path.home() / "cos_data" / "StockDailyBar"
        files = sorted(LOCAL_DAILY.glob("2*.parquet"))
        if not files:
            _HAS_CLOSE = pd.DataFrame()
            return _HAS_CLOSE
        files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, Close as close
            FROM read_parquet({files_str})
            WHERE TradeDate >= DATE '2019-01-02'
        """).df()
        mat = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
        mat.index = pd.to_datetime(mat.index)
        mat = mat.sort_index()
        _HAS_CLOSE = mat
        return mat
    except Exception:
        _HAS_CLOSE = pd.DataFrame()
        return _HAS_CLOSE

# ============================================================
# 批量评估（向量化：用 quant_evaluator 的 compute_daily_ic 一次算所有因子）
# ============================================================
_ALL_METRICS_CACHE = None  # 全 61 因子指标缓存（一次性算）

# 收益口径全局常量：后复权 AdjVwap，t+1 成交 → t+2 卖出（企业级）
_HORIZON_SHIFT = -2
# 交易成本假设：双边 0.1%（单边 0.05%），按换手率 × 费率扣减（可配环境变量 TRADING_COST_BPS）
_COST_BPS = float(os.environ.get("TRADING_COST_BPS", "10"))  # 双边总费率，单位 bp
_TOTAL_COST = _COST_BPS / 1e4

_OPT_META_PATH = BACKTEST_OUT / "optimized_meta.json"
_OPT_META_CACHE = None


def _load_optimized_meta() -> dict:
    """进程级缓存加载 optimized_meta.json（只读；含 is_flipped 标记）"""
    global _OPT_META_CACHE
    if _OPT_META_CACHE is None:
        _OPT_META_CACHE = {}
        try:
            if _OPT_META_PATH.exists():
                _OPT_META_CACHE = json.loads(_OPT_META_PATH.read_text())
        except Exception:
            _OPT_META_CACHE = {}
    return _OPT_META_CACHE


def _is_flipped_by_meta(name: str) -> bool:
    """用 optimized_meta.json 判断因子是否已翻转（is_flipped=True）。

    注意：61 因子带 _flipped 后缀的变体（如 vol_volume_asym_ewma_flipped）在
    456 全量渲染时以基础名（无后缀）重新落值并翻转，详情页也是基础名；
    这里的 name 一律是 factor_matrices_all/ 里的 page_name（无 _flipped 后缀）。
    """
    meta = _load_optimized_meta()
    base = name.replace("_flipped", "")
    rec = meta.get(name) or meta.get(base)
    if rec is None:
        # 兜底：名字带 _flipped 后缀本身即翻转
        return name.endswith("_flipped")
    return bool(rec.get("is_flipped", False))


def _load_vwap_adj() -> pd.DataFrame:
    """加载 后复权 AdjVwap 矩阵 (date × symbol) — 全局收益口径硬性。

    数据源 StockDailyBarAdj/（后复权），禁止未复权 StockDailyBar.Vwap。
    进程级缓存，跨线程共享（主进程算好后 ThreadPool 直接读）。
    """
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    try:
        if not _HAS_DUCKDB:
            _HAS_VWAP = pd.DataFrame()
            return _HAS_VWAP
        ADJ_DIR = Path.home() / "cos_data" / "StockDailyBarAdj"
        files = sorted(ADJ_DIR.glob("*.parquet"))
        if not files:
            _HAS_VWAP = pd.DataFrame()
            return _HAS_VWAP
        fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
            FROM read_parquet({fs})
        """).df()
        mat = df.pivot_table(index='date', columns='symbol', values='vwap', aggfunc='first')
        mat.index = pd.to_datetime(mat.index)
        mat = mat.sort_index()
        _HAS_VWAP = mat
    except Exception:
        _HAS_VWAP = pd.DataFrame()
    return _HAS_VWAP


_HAS_VWAP = None

def _compute_all_factors_metrics() -> dict:
    """一次性向量化算出所有因子的所有指标。

    流程：
      1. 读 parquet 拿到全因子矩阵 (T, N_total)
      2. 读 后复权 AdjVwap 矩阵做交集
      3. 调用 quant_evaluator.compute_daily_ic(factor_batch=(T,N,F), label_bundle=(T,N))
         → (ic_series (T,F), valid_count (T,F))
      4. 用 compute_long_short_returns 算 long/short returns
      5. 用 compute_sharpe_ratio / compute_maximum_drawdown 算业绩指标
      6. 扣双边交易成本（按 Top10% 组换手率 × 费率）
    返回 dict[name] -> {...}，每个 name 含 ic_series / decile_navs / perf / mean_ic ...
    """
    global _ALL_METRICS_CACHE
    if _ALL_METRICS_CACHE is not None:
        return _ALL_METRICS_CACHE
    if not _HAS_DUCKDB:
        return {}

    try:
        # 1. 全量加载因子矩阵（per-factor 文件；日期轴取第一个文件）
        df_full = _load_full_fv()
        if df_full.empty:
            _ALL_METRICS_CACHE = {}
            return _ALL_METRICS_CACHE

        # 拆出 (T, N_total) per factor（只保留对得上日期的）
        factor_names = sorted(set(c[0] for c in df_full.columns))
        common_idx = df_full.index
        mat_dict = {}
        for fn in factor_names:
            cols = [c for c in df_full.columns if c[0] == fn]
            if not cols:
                continue
            m = df_full[cols].copy()
            m.columns = [c[1] for c in m.columns]
            mat_dict[fn] = m

        # 2. 后复权 AdjVwap 矩阵（收益口径 vwap-to-vwap）
        vwap = _load_vwap_adj()
        if vwap.empty:
            _ALL_METRICS_CACHE = {}
            return _ALL_METRICS_CACHE
        vwap.index = pd.to_datetime(vwap.index)
        # 用第一个因子时间轴作 common
        common_idx = mat_dict[factor_names[0]].index.intersection(vwap.index)
        vwap = vwap.loc[common_idx]

        # 对齐每个因子到 vwap columns
        common_cols = vwap.columns
        for fn in factor_names:
            mat_dict[fn] = mat_dict[fn].reindex(index=common_idx, columns=common_cols)

        fwd = vwap.pct_change().shift(_HORIZON_SHIFT)  # 后复权 vwap-to-vwap（t+1成交→t+2卖出，企业级）
        fwd_a = fwd.values.astype(np.float64)

        T, N = vwap.shape
        print(f"[batch] common shape: ({T} days x {N} stocks), {len(factor_names)} factors")

        # 3. 构造 (T, N, F) FactorBatch + (T, N) LabelBundle
        # factor_values: (T, N, F)
        if QE_AVAILABLE:
            values_3d = np.stack([
                mat_dict[fn].values.astype(np.float64) for fn in factor_names
            ], axis=-1)  # (T, N, F)
            fb = FactorBatch(
                factor_ids=tuple(factor_names),
                values=values_3d,
                layout="wide",
                time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
                asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
            )
            import pandas as _pd_t
            _T = _pd_t.DatetimeIndex(common_idx)
            _T_end = _T + _pd_t.Timedelta(days=1)
            lb = LabelBundle(
                target_id="next_ret",
                values=fwd_a,
                horizon=1,
                decision_time=tuple(_T),
                label_start_time=tuple(_T),
                label_end_time=tuple(_T_end),
            )
            ic_arr, valid_arr = compute_daily_ic(
                fb, lb, method="spearman", min_assets=20
            )  # (T, F), (T, F)
            # turn NaN → 0
            ic_arr = np.where(np.isfinite(ic_arr), ic_arr, 0.0)
        else:
            # 兜底：手写每天 for 循环算 Spearman
            ic_arr = np.zeros((T, len(factor_names)))
            for fi, fn in enumerate(factor_names):
                mn = mat_dict[fn].values
                for t in range(T):
                    m = mn[t]; r = fwd_a[t]
                    mask = np.isfinite(m) & np.isfinite(r)
                    if mask.sum() < 20:
                        continue
                    m_v = m[mask]; r_v = r[mask]
                    rm = m_v.argsort().argsort()
                    rr = r_v.argsort().argsort()
                    sm = rm.std(); sr = rr.std()
                    if sm > 1e-9 and sr > 1e-9:
                        ic_arr[t, fi] = float(((rm - rm.mean()) * (rr - rr.mean())).sum() / (len(m_v) * sm * sr))

        # 4. 构造返回值
        out = {}
        n_years = T / 252.0

        for fi, fn in enumerate(factor_names):
            ic_series = pd.Series(ic_arr[:, fi], index=common_idx)
            mean_ic = float(np.mean(ic_arr[:, fi]))
            std_ic = float(np.std(ic_arr[:, fi], ddof=1))
            icir = mean_ic / std_ic if std_ic > 1e-9 else 0.0

            # ===== 十分层 NAV（含双边交易成本，按 Top10% 组换手率 × 费率）=====
            fv = mat_dict[fn].values.astype(np.float64)
            fr = fwd_a
            valid_mask = np.isfinite(fv) & np.isfinite(fr)

            # 每日十分组（G1=bottom 10% → G10=top 10%），并记录 Top10% 组换手率
            mat_ranks = pd.DataFrame(fv).rank(axis=1, method='first', pct=True).values
            group_ids = np.floor(mat_ranks * 10).clip(0, 9).astype(int)
            group_ids[~valid_mask] = -1

            # 预计算每列次日是否同组（用于组换手率）
            prev_gids = np.full(N, -1)
            group_ret = np.zeros((T, 10))          # 已扣费后的各组收益
            top_turnover = np.zeros(T)             # Top10% 组每日换手率

            for t in range(T):
                for k in range(10):
                    mk = (group_ids[t] == k)
                    if not mk.any():
                        continue
                    # 组换手：与昨日同组比例（新进/退出视为换手）
                    if t > 0:
                        same = (prev_gids[mk] == k)
                        to_rate = 1.0 - float(same.mean())
                    else:
                        to_rate = 0.0
                    gr_ret = float(np.nanmean(fr[t, mk]))
                    if k == 9:
                        top_turnover[t] = to_rate
                    group_ret[t, k] = gr_ret - to_rate * _TOTAL_COST
                prev_gids = group_ids[t].copy()

            # 第 0 天无前日，换手按 0 计（成本不扣）
            top_turnover[0] = 0.0
            mean_top_turnover = float(np.nanmean(top_turnover[1:])) if T > 1 else 0.0

            # NAV
            decile_navs = {f"G{k+1}": np.cumprod(1 + group_ret[:, k]) for k in range(10)}
            r_ls = group_ret[:, 9] - group_ret[:, 0]  # 多空收益（G10 已扣费，G1 已扣费）
            decile_navs["LS"] = np.cumprod(1 + r_ls)

            # 业绩指标
            ls_nav = decile_navs["LS"]
            ls_rets = np.diff(ls_nav) / ls_nav[:-1]
            ls_rets_safe = np.concatenate([[0.0], ls_rets])

            if QE_AVAILABLE:
                ls_sharpe = float(compute_sharpe_ratio(ls_rets_safe, periods_per_year=252))
                ls_mdd_tup = compute_maximum_drawdown(ls_rets_safe, missing_return_policy="zero_fill")
                ls_mdd = float(ls_mdd_tup[0]) if isinstance(ls_mdd_tup, tuple) else float(ls_mdd_tup)
                ls_winrate = float(compute_win_rate(ls_rets_safe))
            else:
                ls_sharpe = float(np.mean(ls_rets_safe) / np.std(ls_rets_safe) * np.sqrt(252)) if np.std(ls_rets_safe) > 0 else 0
                ls_mdd = float((pd.Series(ls_nav) / pd.Series(ls_nav).cummax() - 1).min()) if len(ls_nav) > 0 else 0
                ls_winrate = float((ls_rets > 0).sum() / max(len(ls_rets), 1))

            ls_annual = float(ls_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
            g10_ann = float(decile_navs["G10"][-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0
            g1_ann = float(decile_navs["G1"][-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0
            g10_rets = np.diff(decile_navs["G10"]) / decile_navs["G10"][:-1]
            g1_rets = np.diff(decile_navs["G1"]) / decile_navs["G1"][:-1]
            if QE_AVAILABLE:
                g10_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g10_rets), periods_per_year=252))
                g1_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g1_rets), periods_per_year=252))
            else:
                g10_sharpe = float(np.mean(g10_rets) / np.std(g10_rets) * np.sqrt(252)) if np.std(g10_rets) > 0 else 0
                g1_sharpe = float(np.mean(g1_rets) / np.std(g1_rets) * np.sqrt(252)) if np.std(g1_rets) > 0 else 0

            # 写输出
            dates_out = list(common_idx)
            decile_navs_out = {"dates": dates_out}
            for k, v in decile_navs.items():
                decile_navs_out[k] = list(v)

            out[fn] = {
                "ic_series": ic_series,
                "decile_navs": decile_navs_out,
                "dates_out": dates_out,
                "perf": {
                    "ls_sharpe": ls_sharpe,
                    "ls_annual": ls_annual,
                    "ls_mdd": ls_mdd,
                    "ls_winrate": ls_winrate,
                    "g10_annual": g10_ann,
                    "g1_annual": g1_ann,
                    "g10_sharpe": g10_sharpe,
                    "g1_sharpe": g1_sharpe,
                    "turnover": mean_top_turnover,
                    "cost_bps": _COST_BPS,
                    "n_periods": T,
                    "start_date": str(common_idx[0])[:10] if T > 0 else None,
                    "end_date": str(common_idx[-1])[:10] if T > 0 else None,
                },
                "mean_ic": mean_ic,
                "ic_ir": icir,
                "win_rate": ls_winrate,
                "ic_std": std_ic,
                "qe_used": QE_AVAILABLE,
            }

        _ALL_METRICS_CACHE = out
        print(f"[batch] done {len(out)} factors")
        return out

    except Exception as e:
        import traceback
        print(f"[batch ERR] {e}")
        traceback.print_exc()
        _ALL_METRICS_CACHE = {}
        return {}


def _compute_factor_metrics(name: str) -> dict:
    """对单个因子取批量算好的指标（保证和首页一致）"""
    if not _HAS_DUCKDB:
        return {}
    full = _compute_all_factors_metrics()
    # 兼容两种 key 格式: 'xxx' 或 'factor_xxx'
    return full.get(f"factor_{name}", full.get(name, {}))


def compute_ic_stats(ic_series: pd.Series) -> dict:
    """计算IC相关统计（直接用 Series，不再过滤 0）"""
    if ic_series is None or len(ic_series) == 0:
        return {}
    s = ic_series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return {}

    mean_ric = float(s.mean())
    std_ric = float(s.std())
    ric_ir = mean_ric / std_ric if std_ric > 0 else 0
    win_rate = float((s > 0).sum() / len(s))

    # 月度IC (mean)
    monthly = s.resample("ME").mean().dropna()

    return {
        "mean_rankic": mean_ric,
        "std_ric": std_ric,
        "rankic_ir": ric_ir,
        "win_rate": win_rate,
        "n_periods": len(s),
        "monthly_ic": monthly,
        "series": s,
    }

# ============================================================
# 绘图函数（返回 base64 PNG）
# ============================================================
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=PLOT_BG, edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def plot_ic_monthly_heatmap(monthly_ic: pd.Series, factor_name: str, is_flipped: bool = False) -> str:
    """月度IC热力图"""
    if monthly_ic is None or len(monthly_ic) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    monthly_ic = monthly_ic.dropna()
    if len(monthly_ic) < 2:
        plt.close(fig)
        return ""

    # 热力图数据
    months = monthly_ic.index.to_period("M")
    years = monthly_ic.index.year
    unique_years = sorted(set(years))
    unique_months = list(range(1, 13))

    grid = np.full((len(unique_years), 12), np.nan)
    for i, (dt, v) in enumerate(monthly_ic.items()):
        yi = unique_years.index(dt.year)
        mi = dt.month - 1
        grid[yi, mi] = v

    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["1","2","3","4","5","6","7","8","9","10","11","12"])
    ax.set_yticks(range(len(unique_years)))
    ax.set_yticklabels(unique_years)
    ax.set_xlabel("月份")
    flip_suffix = "（已翻正）" if is_flipped else ""
    ax.set_title(f"{factor_name} — 月度 RankIC 热力图{flip_suffix}", fontsize=9, color=FG)
    plt.colorbar(im, ax=ax, label="RankIC", shrink=0.8)

    # 填数值
    for yi in range(len(unique_years)):
        for mi in range(12):
            v = grid[yi, mi]
            if not np.isnan(v):
                ax.text(mi, yi, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.05 else FG)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def plot_decile_nav(decile_data: dict, factor_name: str, is_flipped: bool = False) -> str:
    """十分层净值曲线 (10 条曲线 + 多空)"""
    if not decile_data or "dates" not in decile_data:
        return ""
    dates = pd.to_datetime(decile_data["dates"])
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, 10))
    for k in range(1, 11):
        gkey = f"G{k}"
        if gkey not in decile_data:
            continue
        g = decile_data[gkey]
        if len(g) != len(dates):
            continue
        lw = 0.9
        ls = "-"
        if k == 10:
            lw = 1.3; ls = "-"
        ax.plot(dates, g, color=colors[k-1], linewidth=lw, linestyle=ls,
                label=f"G{k}", alpha=0.85)

    # 多空: 紫色虚线
    if "LS" in decile_data and len(decile_data["LS"]) == len(dates):
        ax.plot(dates, decile_data["LS"], color="#7c3aed",
                linewidth=2.0, label="多空 (G10-G1)", linestyle="-")

    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.7)
    flip_suffix = "（已翻正）" if is_flipped else ""
    cost_suffix = f"（扣双边成本 {_COST_BPS:.0f} bp）" if _COST_BPS else ""
    ax.set_title(f"{factor_name} — 十分层净值曲线 (G1~G10){flip_suffix} {cost_suffix}", fontsize=10, color=FG, fontweight='bold')
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def plot_ic_timeseries(ic_series: pd.Series, factor_name: str, is_flipped: bool = False) -> str:
    """IC时序SVG（内嵌，无需CDN）— 显示所有 IC, 包括 0"""
    if ic_series is None or len(ic_series) < 2:
        return ""
    # 不丢 0 值: 全部画
    s = ic_series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)

    dates = s.index
    vals = s.values
    n = len(dates)

    W, H = 700, 180
    x_pad, y_pad = 40, 20
    plot_w = W - x_pad * 2
    plot_h = H - y_pad * 2

    # 固定 vmin/vmax=-0.15..0.15 让图可读
    v_min, v_max = -0.15, 0.15
    v_range = v_max - v_min
    x_vals = np.arange(n)
    y_norm = np.clip((vals - v_min) / v_range, 0, 1)

    xs = x_pad + (x_vals / max(n - 1, 1)) * plot_w
    ys = y_pad + (1 - y_norm) * plot_h
    zero_y = y_pad + (1 - (0 - v_min) / v_range) * plot_h

    # bars — 只画非 NaN 的 IC
    bars = ""
    for i in range(n):
        v = vals[i]
        if not np.isfinite(v) or abs(v) < 1e-9:
            continue  # 跳过 NaN / 零 IC 位置，保持图表干净
        cx = xs[i]
        cy = ys[i]
        bar_h = abs(cy - zero_y)
        y_top = min(cy, zero_y)
        color = "#16a34a" if v > 0 else "#dc2626"
        bars += '<rect x="{:.1f}" y="{:.1f}" width="2.4" height="{:.1f}" fill="{}" opacity="0.85"/>\n'.format(
            cx - 1.2, y_top, max(bar_h, 0.5), color
        )

    # x labels
    step = max(1, n // 12)
    x_labels = []
    for i in range(0, n, step):
        x_labels.append(
            '<text x="{:.1f}" y="{}" text-anchor="middle" font-size="7" fill="#64748b">{}</text>\n'.format(
                xs[i], H - 4, str(dates[i].strftime("%Y-%m"))[:7]
            )
        )

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + str(W) + ' ' + str(H) + '" '
        'style="width:100%;max-height:200px;font-family:Segoe UI,Microsoft YaHei,system-ui,sans-serif">\n'
        + bars + '\n'
        '<line x1="' + str(x_pad) + '" y1="' + str(round(zero_y, 1)) + '" x2="' + str(W - x_pad) + '" y2="' + str(round(zero_y, 1)) + '" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="3,3"/>\n'
        + (f'<text x="{x_pad}" y="{y_pad - 6}" font-size="7" fill="#b45309">IC 已翻正（×(-1)）</text>\n' if is_flipped else '')
        + ''.join(x_labels) + '\n'
        '</svg>'
    )
    return svg

def plot_long_short_nav(decile_data: dict, factor_name: str, is_flipped: bool = False) -> str:
    """多空净值图"""
    if not decile_data or "dates" not in decile_data:
        return ""
    dates = pd.to_datetime(decile_data["dates"])
    if "LS" not in decile_data or len(decile_data["LS"]) != len(dates):
        return ""

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ls = decile_data["LS"]
    g10 = decile_data.get("G10", [])
    g1 = decile_data.get("G1", [])

    ax.plot(dates, ls, color="#7c3aed", linewidth=1.5, label="多空 (G10-G1)")
    if len(g10) == len(dates):
        ax.plot(dates, g10, color="#16a34a", linewidth=1, label="G10 (多头)", alpha=0.7)
    if len(g1) == len(dates):
        ax.plot(dates, g1, color="#dc2626", linewidth=1, label="G1 (空头)", alpha=0.7)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    flip_suffix = "（已翻正）" if is_flipped else ""
    cost_suffix = f"（扣双边成本 {_COST_BPS:.0f} bp）" if _COST_BPS else ""
    ax.set_title(f"{factor_name} — 多空净值曲线{flip_suffix} {cost_suffix}", fontsize=9, color=FG)
    ax.set_ylabel("净值")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def plot_ic_distribution(ic_series: pd.Series, factor_name: str, is_flipped: bool = False) -> str:
    """IC分布直方图"""
    if ic_series is None or len(ic_series) < 5:
        return ""
    ic_filtered = ic_series[ic_series != 0]
    if len(ic_filtered) < 5:
        return ""

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(ic_filtered.values, bins=30, color=PRIMARY, alpha=0.7, edgecolor="white")
    # 红线 = 均值 (而非 0)；0 线作为参考
    ic_mean = float(np.mean(ic_filtered.values))
    ic_med = float(np.median(ic_filtered.values))
    ax.axvline(ic_mean, color="red", linewidth=2, linestyle="-", label=f"均值 = {ic_mean:+.4f}")
    ax.axvline(ic_med, color="orange", linewidth=1.5, linestyle="--", label=f"中位数 = {ic_med:+.4f}")
    ax.axvline(0, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="upper right")
    flip_suffix = "（已翻正）" if is_flipped else ""
    ax.set_title(f"{factor_name} — RankIC 分布{flip_suffix}", fontsize=8, color=FG)
    ax.set_xlabel("RankIC")
    ax.set_ylabel("频数")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

# ============================================================
# HTML 生成
# ============================================================
def _render_field_op_doc(required_columns: str, dsl: str, FIELD_DOC: dict, OP_DOC: dict) -> str:
    """渲染 '字段与算子释义' 卡片内容, 只列公式/字段中实际出现的字段和算子"""
    import re as _re

    parts = []
    # ---- 字段 ----
    cols = [c.strip() for c in (required_columns or "").split(",") if c.strip()]
    # 同时从 dsl 中抓可能出现的字段 (单标识符, 不在算子表里的)
    KNOWN_OPS = {"ts_mean","ts_std","ts_min","ts_max","ts_sum","ts_median","ts_rank",
                 "ts_delta","ts_corr","ts_cov","ts_decay_linear","ts_decay_exp","ts_skew",
                 "ts_kurt","ts_argmax","ts_argmin","ts_quantile","ts_topk_sum","ts_pct",
                 "ema","ewma","ewm","rank","cs_rank","cs_zscore","scale","winsorize",
                 "where","abs","log","sqrt","sign","pow","tanh","delay","ts_delay",
                 "where","clip","group_mean","group_rank","group_zscore","nan_to_num",
                 "ts_atr","ts_regression","ts_decay","amount_weighted_mean","is_high","is_low",
                 # 算子参数 / Python内置函数（DSL中可能出现但不等于字段）
                 "span","pct_change","min","max","returns","shift","zscore","n","k","p","q",
                 # 中间派生变量名（DSL中可能出现的语义变量）
                 "score","depth_score","overnight_ret","overnight_imp",
                 "sigma_down","sigma_up","downside_var","total_var",
                 "ewm_total_var","ewma_downside_var","ewma_down_vol","ewma_up_vol",
                 "sma_down","sma_up","down_vol_60","up_vol_60",
                 "robust_vol_ratio","style_gate_resvol_high",
                 "delta_depth_10","vol_ratio_60","vol_ratio_40",
                 "amount_ma_diff","mkt_cap_float","atr_20","trading_datetime",
                 "open_price","high_price","low_price"}
    if OP_DOC:
        KNOWN_OPS = KNOWN_OPS | set(OP_DOC.keys())
    used_cols = set()
    if dsl:
        for m in _re.finditer(r"\b([a-zA-Z_][a-zA-Z_0-9]*)\b", dsl):
            name = m.group(1)
            if name in DSL_NOISE:
                continue
            if name not in KNOWN_OPS and not name.replace(".", "").replace("-", "").isdigit():
                used_cols.add(name)

    all_cols = list(dict.fromkeys(cols + sorted(used_cols)))  # 去重保序

    if all_cols:
        rows = []
        for c in all_cols:
            if c in FIELD_DOC:
                nm, desc = FIELD_DOC[c]
                rows.append((c, nm, desc))
            else:
                rows.append((c, "（未登记字段）", "请在 jobs/rebuild_factor_detail_pages.py 的 FIELD_DOC 中补充含义"))
        if rows:
            parts.append('<h3 style="font-size:0.95rem;margin:0 0 8px;color:var(--primary)">输入字段</h3>')
            parts.append('<table class="meta-table"><thead><tr><th>字段</th><th>含义</th><th>说明</th></tr></thead><tbody>')
            for c, nm, desc in rows:
                parts.append(f'<tr><td><code>{c}</code></td><td>{nm}</td><td style="color:var(--muted)">{desc}</td></tr>')
            parts.append('</tbody></table>')

    # ---- 算子 ----
    if dsl:
        used_ops = []
        seen = set()
        for m in _re.finditer(r"\b([a-zA-Z_][a-zA-Z_0-9]*)\s*\(", dsl):
            op = m.group(1)
            if op not in OP_DOC:
                # DSL 语法关键词/内置函数不做算子展示
                if op in DSL_NOISE or op in ("if", "else", "then", "For", "I", "C", "O", "H", "L"):
                    continue
            if op not in seen:
                seen.add(op)
                used_ops.append(op)
        if used_ops:
            parts.append('<h3 style="font-size:0.95rem;margin:18px 0 8px;color:var(--primary)">算子</h3>')
            parts.append('<table class="meta-table"><thead><tr><th>算子</th><th>含义</th></tr></thead><tbody>')
            for op in used_ops:
                doc = OP_DOC.get(op, f"{op}(...) — 自定义算子")
                parts.append(f'<tr><td><code>{op}</code></td><td style="color:var(--muted)">{doc}</td></tr>')
            parts.append('</tbody></table>')

    return "\n".join(parts) if parts else "<div class='zero-notice'>无可展示的字段或算子</div>"


def build_detail_html(
    factor_name: str,
    formula: str,
    code: str,
    ic_stats: dict,
    decile_data: dict,
    svg_timeseries: str,
    monthly_chart: str,
    decile_chart: str,
    ls_chart: str,
    dist_chart: str,
    is_flipped: bool,
    perf: dict = None,
    dsl: str = "",
    rationale: str = "",
    required_columns: str = "",
    dsl_status: str = "ok",
    dsl_note: str = "",
    title: str = "",
    steps: list = None,
    manual_note: str = "",
    cost_bps: float = None,
) -> str:
    steps = steps or []
    """构建单个因子详情页HTML"""

    perf = perf or {}
    if cost_bps is None:
        cost_bps = float(perf.get("cost_bps", _COST_BPS))
    mean_rankic = ic_stats.get("mean_rankic", 0)
    rankic_ir = ic_stats.get("rankic_ir", 0)
    win_rate = ic_stats.get("win_rate", 0)
    n_periods = ic_stats.get("n_periods", 0)

    # 优先用真 perf，没值才走 decile_data 兜底
    if perf and "ls_sharpe" in perf:
        ls_sharpe_gross = perf.get("ls_sharpe", 0)
        ls_annual_return = perf.get("ls_annual", 0) * 100  # → %
        ls_mdd = perf.get("ls_mdd", 0) * 100  # → %
        ls_winrate = perf.get("ls_winrate", 0) * 100
        g10_ann = perf.get("g10_annual", 0) * 100
        g1_ann = perf.get("g1_annual", 0) * 100
    elif decile_data and "LS" in decile_data:
        ls_vals = decile_data["LS"]
        if len(ls_vals) > 2:
            ls_ser = pd.Series(ls_vals)
            ls_rets = ls_ser.pct_change().dropna()
            ls_sharpe_gross = compute_annualized_sharpe(ls_rets)
            ls_nav_end = ls_vals[-1]
            ls_nav_start = ls_vals[0] if ls_vals[0] != 0 else 1
            ls_annual_return = (ls_nav_end / ls_nav_start - 1) * 100 * 252 / max(len(ls_vals), 1)
            ls_mdd = compute_max_drawdown(ls_ser) * 100
            ls_winrate = float((ls_rets > 0).sum() / max(len(ls_rets), 1)) * 100
            g10_ann = g1_ann = 0
        else:
            ls_sharpe_gross = ls_annual_return = ls_mdd = ls_winrate = g10_ann = g1_ann = 0
    else:
        ls_sharpe_gross = ls_annual_return = ls_mdd = ls_winrate = g10_ann = g1_ann = 0

    def pcls(v):
        return "pos" if v >= 0 else "neg"

    def fmt(v, pct=False):
        if pct:
            return f"{v:.2f}%"
        if abs(v) < 1:
            return f"{v:.4f}"
        return f"{v:.3f}"

    turnover_daily = float(perf.get("turnover", 0)) if perf else 0.0
    rankic_win_rate = float(ic_stats.get("win_rate", 0)) if ic_stats else 0.0

    # 公式展示（自动翻转说明）
    # 公式区显示的是 extract_formula_info 已经塞好 sign 前缀 + fallback 包装的 dsl_text
    flip_note = ""
    if is_flipped:
        flip_note = '<span class="badge badge-yellow">⚠ 已翻转（IC&lt;0）</span>'
    if dsl_status == "fallback":
        flip_note += '<span class="badge badge-orange" style="background:#fed7aa;color:#9a3412;margin-left:4px">⚠ FE DSL 暂不支持</span>'
        if not is_flipped:
            flip_note = '<span class="badge badge-orange" style="background:#fed7aa;color:#9a3412">⚠ FE DSL 暂不支持</span>'

    qe_badge = '<span class="qe-info">⚡ quant_evaluator</span>' if QE_AVAILABLE else '<span class="qe-info" style="background:#fee2e2;color:#991b1b">legacy</span>'

    # rationale
    rationale_html = ""
    if rationale:
        rationale_html = f'<div class="rationale"><b>设计意图：</b> {rationale}</div>'

    # ===== 字段释义 (dataaccess 标准列) =====
    FIELD_DOC = {
        "open": ("开盘价", "当日首笔成交价，T-1 close 跳空缺口分析需要"),
        "high": ("最高价", "当日成交最高价；计算日内振幅 / 上影线"),
        "low": ("最低价", "当日成交最低价；计算日内振幅 / 下影线"),
        "close": ("收盘价", "当日收盘价；多数因子都用 close 而非 vwap"),
        "close_price": ("收盘价 (别名)", "等同 close；不同源可能有不同列名"),
        "vwap": ("成交量加权均价", "amount/volume，更能代表真实成交均价"),
        "volume": ("成交量 (股数)", "默认未平减；反映参与度"),
        "amount": ("成交额 (元)", "amount = price × volume；反映资金参与"),
        "pre_close": ("昨收", "T-1 收盘；用于计算跳空 / 当日收益"),
        "adj_factor": ("复权因子", "后复权调整；与 close_price 相乘得复权价"),
        "is_suspend": ("停牌标志", "True 当日停牌；用于过滤无效信号"),
        "high_limit": ("涨停价", "T+1 涨停限制 +10%（创业板/科创板+20%)"),
        "low_limit": ("跌停价", "T+1 跌停限制 -10%（创业板/科创板-20%)"),
        "TradingDay": ("交易日", "YYYY-MM-DD 字符串"),
        "OrderBookId": ("标的代码", "内部证券唯一 ID"),
        "Symbol": ("股票代码", "外部代码 (如 000001.SZ)"),
        "TradeDate": ("交易日", "来自 parquet 的日期字段"),
        "pb_lf": ("PB 市净率 (last_filed)", "最新一次披露的市净率 = price / 每股净资产；估值越低越便宜"),
        "pe_ttm": ("PE 市盈率 TTM", "滚动 12 个月市盈率 = price / EPS_TTM"),
        "ps_ttm": ("PS 市销率 TTM", "滚动 12 个月市销率 = market_cap / revenue_TTM"),
        "eps_ttm": ("每股收益 TTM", "trailing twelve months 每股收益"),
        "roe_ttm": ("ROE 净资产收益率 TTM", "净利润 / 平均股东权益，越高质量越好"),
        "debttoassets": ("资产负债率", "总负债 / 总资产"),
        "free_turn": ("自由换手率", "剔除非流通股本影响后的换手率"),
        "turnover": ("换手率", "成交量 / 流通股本；越高代表交易越活跃"),
        "log_volume": ("log(volume)", "成交量的自然对数；用于压缩极端值"),
        "log_amount": ("log(amount)", "成交额的对数"),
        "log_close": ("log(close)", "收盘价对数"),
        "log_return": ("对数收益", "log(close / ts_delay(close, 1))"),
        "ret": ("日收益", "(close - ts_delay(close, 1)) / ts_delay(close, 1)"),
        "ret_5": ("5 日收益", "(close - ts_delay(close, 5)) / ts_delay(close, 5)"),
        "vol_ratio": ("量比", "volume / ts_mean(volume, N)，>1 放量,<1 缩量"),
        "vol_ma20": ("20 日均量", "ts_mean(volume, 20)"),
        "vol_ma40": ("40 日均量", "ts_mean(volume, 40)"),
        "vol_ma60": ("60 日均量", "ts_mean(volume, 60)"),
        "amount_ma": ("成交额均线", "ts_mean(amount, N)"),
        "vwap_30": ("30 日均 VWAP", "ts_mean(vwap, 30)"),
        "resvol_high": ("高残差波动期标志", "alpha_tools 分类器: 残差 std>阈值时为 True"),
        "resvol_low": ("低残差波动期标志", "alpha_tools 分类器: 残差 std<阈值时为 True"),
        "high_resvol": ("高残差波动期 (别名)", "resvol_high 的别名"),
        "liquidity_high": ("高流动性标志", "alpha_tools 对 amount/turnover 的分类"),
        "is_high_volume": ("高量能日标志", "alpha_tools.classify_volume_regime 输出"),
        "is_low_volume": ("低量能日标志", "同上"),
        "raw": ("中间原始信号", "未平滑/未标准化的因子值"),
        "neg_returns": ("负收益序列", "min(ret, 0)；只取收益 ≤ 0 部分"),
        "pos_returns": ("正收益序列", "max(ret, 0)；只取收益 ≥ 0 部分"),
        "downside_returns": ("下行收益", "neg_returns 的别名"),
        "upside_returns": ("上行收益", "pos_returns 的别名"),
        "free_turn_20": ("20 日自由换手率", "近 20 日自由换手率"),
        "pb_lf_smoothed": ("平滑 PB", "PB 经 ema/rolling 平滑"),
        "normalized": ("归一化值", "标准化到 0~1 或 -1~1"),
        "depth": ("回撤深度", "rolling_max(close, N) - close, 再除以 peak 归一"),
        "duration": ("回撤持续天数", "自从达到 N 日高点的天数"),
        "recovery": ("恢复强度", "回撤后反弹 / 跌幅"),
        "drawdown": ("回撤", "(close - rolling_max(close, N)) / rolling_max(close, N)"),
        "intensity": ("信号强度", "经标准化或归一化的因子值"),
        "smoothed": ("平滑后的值", "经 ema/rolling 平滑"),
        "typical_price": ("典型价", "(high + low + close) / 3"),
        "adx": ("ADX 平均趋向指数", "talib.ADX(high, low, close, N)；趋势强度"),
        "delta_depth": ("回撤深度变化", "depth[t] - depth[t-k]"),
        "leverage_penalty": ("杠杆惩罚系数", "基于 debttoassets"),
        "ewm_downside_var": ("ewm 下行方差", "对 min(ret,0)^2 做 ewm"),
        "amount_weighted_mean": ("amount 加权均值", "按成交额加权的时间维度平均"),
        # ===== 中间派生变量 / DSL语义变量（DSL公式内部生成，不对应原始列） =====
        "span": ("滚动窗口参数", "EMA / 滚动窗口的半衰期 span=n（span=2/(n+1) 近似 α）"),
        "range_pct": ("日内振幅率", "(high - low) / close，即当日的振幅/收盘价比率，反映波动幅度"),
        "score": ("中间信号", "多因子或复合计算的中间结果，在最终输出前可能再经平滑/标准化"),
        "depth_score": ("回撤强度", "综合回撤深度与持续时间计算的信号强度"),
        "overnight_ret": ("隔夜收益", "(open - ts_delay(close,1)) / ts_delay(close,1)；反映竞价阶段涨跌"),
        "overnight_imp": ("隔夜冲击", "基于隔夜收益方向与幅度计算的冲击系数"),
        "sigma_down": ("下行波动率", "仅取 ret<0 部分计算的滚动标准差，衡量下行风险"),
        "sigma_up": ("上行波动率", "仅取 ret>0 部分计算的滚动标准差，衡量上行弹性"),
        "downside_var": ("下行方差", "min(ret,0)^2 的滚动均值（半方差）"),
        "total_var": ("总方差", "ret^2 的滚动均值（总波动率）"),
        "ewm_total_var": ("EWM 总方差", "对 ret^2 做指数加权移动平均"),
        "ewma_downside_var": ("EMA 下行方差", "对 min(ret,0)^2 做指数加权移动平均"),
        "ewma_down_vol": ("EMA 下行波动", "对负收益标准差做 EMA 平滑"),
        "ewma_up_vol": ("EMA 上行波动", "对正收益标准差做 EMA 平滑"),
        "sma_down": ("简单均线（下）", "仅对负收益序列 ret<0 部分做简单滚动均值"),
        "sma_up": ("简单均线（上）", "仅对正收益序列 ret>0 部分做简单滚动均值"),
        "down_vol_60": ("60日下行波动率", "近60日负收益的标准差（下行风险度量）"),
        "up_vol_60": ("60日上行波动率", "近60日正收益的标准差（上行弹性度量）"),
        "robust_vol_ratio": ("稳健波动比", "上行波动 / 下行波动（>1 代表弹性大于风险）"),
        "style_gate_resvol_high": ("高残差波动风格门控", "在 alpha 的 regime 分类中，残差波动处于高状态时触发"),
        "delta_depth_10": ("回撤深度变化 (10日)", "depth[t] - depth[t-10]；连续10日回撤深度变化"),
        "vol_ratio_60": ("量比 (60日)", "volume / ts_mean(volume, 60)；当前成交量相对60日均量的比值"),
        "vol_ratio_40": ("量比 (40日)", "volume / ts_mean(volume, 40)；当前成交量相对40日均量的比值"),
        "amount_ma_diff": ("成交额均线差", "amount - ts_mean(amount, N)；偏离均线程度"),
        "mkt_cap_float": ("流通市值", "收盘价 × 流通股本；反映股票的实际可交易规模"),
        "atr_20": ("ATR 平均真实波幅 (20日)", "talib.ATR(high, low, close, 20)；综合波动率"),
        "trading_datetime": ("交易时间戳", "结合日期与交易时段的完整时间标记"),
        "open_price": ("开盘价（别名）", "等同 open；部分数据源使用 open_price 列名"),
        "high_price": ("最高价（别名）", "等同 high；部分数据源使用 high_price 列名"),
        "low_price": ("最低价（别名）", "等同 low；部分数据源使用 low_price 列名"),
        "zscore": ("Z-score 标准化", "ts_zscore(x, n) = (x - mean) / std；偏离均值的标准化度量"),
        "returns": ("收益率序列", "通常指 ret 或 ret_N（多期收益）；DSL 中 ret 的别名"),
    }

    # ===== 算子释义 (FactorEngine / LQTP 算子) =====
    OP_DOC = {
        # 时间序列算子
        "ts_mean": "ts_mean(x, n) → x 在过去 n 天的滚动均值 (rolling mean)",
        "ts_std": "ts_std(x, n) → x 在过去 n 天的滚动标准差 (rolling std)",
        "ts_min": "ts_min(x, n) → x 在过去 n 天的滚动最小值",
        "ts_max": "ts_max(x, n) → x 在过去 n 天的滚动最大值",
        "ts_sum": "ts_sum(x, n) → x 在过去 n 天的滚动求和",
        "ts_median": "ts_median(x, n) → x 在过去 n 天的滚动中位数",
        "ts_rank": "ts_rank(x, n) → x 当前值在过去 n 天里的百分位 (0~1)",
        "ts_delta": "ts_delta(x, n) → x[t] - x[t-n]（n 期差分）",
        "ts_corr": "ts_corr(x, y, n) → x 与 y 过去 n 天的滚动相关",
        "ts_cov": "ts_cov(x, y, n) → x 与 y 过去 n 天的滚动协方差",
        "delay": "delay(x, n) → x[t-n]（n 期滞后）",
        # 截面算子
        "rank": "rank(x) → x 在当日截面上的分位数 (0~1，pct=True)",
        "scale": "scale(x) → x 在当日截面上的和=1 (归一化)",
        # 极值裁剪
        "sign": "sign(x) → x 的符号 (1, 0, -1)",
        "abs": "abs(x) → 绝对值",
        # 数学算子
        "log": "log(x) → 自然对数 ln(x)",
        "sqrt": "sqrt(x) → 平方根",
        "exp": "exp(x) → e^x",
        "pow": "pow(x, n) → x 的 n 次方",
        "max": "max(x, n) → 滚动最大值 / 或元素级 max",
        "min": "min(x, n) → 滚动最小值 / 或元素级 min",
        "mean": "mean(x, n) → 滚动均值",
        "sum": "sum(x, n) → 滚动求和",
        "std": "std(x, n) → 滚动标准差",
        "prod": "prod(x, n) → 滚动乘积",
        # 截面统计
        "sigma_down": "sigma_down(close, n) → 滚动下行波动率 (只算 ret<0 部分的 std)",
        "sigma_up": "sigma_up(close, n) → 滚动上行波动率 (只算 ret>0 部分的 std)",
        "MA_volume": "MA_volume(volume, n) → volume 在过去 n 天的滚动均线",
        "MA_close": "MA_close(close, n) → close 在过去 n 天的滚动均线",
        "MA_high": "MA_high(high, n) → high 在过去 n 天的滚动均线",
        "MA_low": "MA_low(low, n) → low 在过去 n 天的滚动均线",
        "MA_open": "MA_open(open, n) → open 在过去 n 天的滚动均线",
        "MA": "MA(x, n) → x 在过去 n 天的滚动均线 (通用简写)",
        # EWMA
        "ewma": "ewma(x, span=n) → 指数加权移动平均 (span=α近似窗口)",
        "EWMA": "EWMA(x, span=n) → 同上（简写）",
        "ewm": "ewm(...) → pandas ewm 系列方法",
        # LQTP
        "ts_decay_linear": "ts_decay_linear(x, n) → 线性衰减加权",
        # alpha_tools
        "classify_volume_regime": "alpha_tools.classify_volume_regime(vol, n) → (is_high, is_low, vol_ratio)；量能状态分类",
        "decompose_overnight_intraday": "alpha_tools.decompose_overnight_intraday(close, open) → (overnight_ret, intraday_ret)；隔夜+日内拆解",
        "get_up_space": "minute_tools.get_up_space(vol, ...) → (is_high, z)；上空间扫描",
        "ATR": "talib.ATR(high, low, close, n) → 平均真实波幅",
        "EMA": "talib.EMA(x, n) → 指数均线",
        "SMA": "talib.SMA(x, n) → 简单均线",
        "STDDEV": "talib.STDDEV(x, n) → 滚动标准差",
        # pandas
        "shift": "shift(n) → pandas 滞后/超前；用于 shift(1) 防未来",
        "rolling": "rolling(n) → pandas 滚动窗口",
        "ewm": "ewm(...) → pandas 指数加权",
        "rank": "DataFrame.rank(method='first') → 排名 (含 ties)",
        "pct_change": "pct_change(n) → x[t]/x[t-n] - 1 (收益率)",
        "where": "where(cond, x) → 当 cond 为真取 x，否则 NaN",
        "clip": "clip(lo, hi) → 上下限裁剪",
        "replace": "replace(to_replace, value) → 替换值",
    }

    # ===== 补全字段/算子释义（jobs/field_op_doc.py）=====
    if apply_field_op_docs is not None:
        apply_field_op_docs(FIELD_DOC, OP_DOC)

    def render_steps(formula_str: str) -> str:
        """把 formula 文本分解为一步步，每步标注算子含义"""
        if not formula_str or formula_str.startswith("（"):
            return ""
        # 简单分词
        import re as _re2
        steps = []
        # 找所有的算子调用 (字母开头跟 (...))
        # 扩展算子表：ts_*, delay, rank, scale, sign, abs, log, sqrt, exp, ewm*,
        # ATR, EMA, SMA, STDDEV, sigma_*, MA_*, rolling, shift, pct_change, where,
        # clip, replace, classify_*, decompose_*, get_up_space, max, min, mean, sum
        op_pattern = (
            r'(ts_\w+|delay|rank|scale|sign|abs|log|sqrt|exp|pow|sin|cos|tan|'
            r'ewma|EWMA|ewm|ewmstd|rolling|shift|pct_change|where|clip|replace|'
            r'classify_volume_regime|decompose_overnight_intraday|get_up_space|'
            r'ATR|EMA|SMA|STDDEV|sigma_\w+|MA_\w+|MA\w*|'
            r'\bmax\b|\bmin\b|\bmean\b|\bsum\b|\bprod\b|\bstd\b)'
        )
        for m in _re2.finditer(op_pattern + r'\s*\(([^()]*)\)', formula_str):
            op = m.group(1)
            args = m.group(2)
            doc = OP_DOC.get(op, f"{op}({args}) — 用户自定义算子")
            steps.append((op, args, doc))
        if not steps:
            return ""
        html = '<ol style="padding-left:18px;margin:8px 0">'
        for op, args, doc in steps:
            html += f'<li><code style="color:#7c3aed">{op}</code>(<code style="color:#0e7490">{args}</code>) — <span style="color:#475569">{doc}</span></li>'
        html += '</ol>'
        return html

    # 字段表 HTML
    if required_columns:
        fields = [c.strip() for c in required_columns.split(",") if c.strip()]
        rows = ""
        for f in fields:
            label, desc = FIELD_DOC.get(f, (f, "（未在 dataaccess 标准字段中 — 可能为自定义）"))
            color = "#0e7490" if f in FIELD_DOC else "#94a3b8"
            rows += f'<tr><td><code style="color:{color}">{f}</code></td><td><b>{label}</b></td><td style="color:#475569">{desc}</td></tr>'
        fields_html = (
            '<div class="card">\n'
            '<h2>📊 输入字段释义 (dataaccess 标准列)</h2>\n'
            '<table class="meta-table"><thead><tr><th>列名</th><th>含义</th><th>说明</th></tr></thead><tbody>'
            + rows +
            '</tbody></table></div>\n'
        )
    else:
        fields_html = ""

    # 算子逐步解释
    formula_steps = render_steps(formula)
    if formula_steps:
        formula_steps_html = (
            '<div class="card">\n'
            '<h2>🔍 公式逐行拆解（算子释义）</h2>\n'
            + formula_steps +
            '</div>\n'
        )
    else:
        formula_steps_html = ""

    # DSL block (FE / LQTP 表达式) — 这里显示的 dsl 来自 extract_formula_info,
    # 已含 sign 前缀 / fallback 包装 / 不可表达的标记
    dsl_block = ""
    if dsl and dsl != "（公式为空）":
        dsl_esc = dsl.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        note_html = ""
        if dsl_note and dsl_status == "fallback":
            note_esc = dsl_note.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            note_html = (
                '<div style="margin-top:8px;padding:8px 12px;background:#fff7ed;'
                'border-left:3px solid #f97316;border-radius:6px;'
                'font-size:0.78rem;color:#7c2d12">' + note_esc + '</div>\n'
            )
        elif dsl_note and dsl_status == "custom":
            note_esc = dsl_note.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            note_html = (
                '<div style="margin-top:8px;padding:8px 12px;background:#fee2e2;'
                'border-left:3px solid #dc2626;border-radius:6px;'
                'font-size:0.78rem;color:#991b1b">' + note_esc + '</div>\n'
            )
        elif dsl_note and (is_flipped or dsl_status == "ok"):
            note_esc = dsl_note.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            note_html = (
                '<div style="margin-top:8px;font-size:0.78rem;color:var(--muted)">'
                + note_esc + '</div>\n'
            )
        dsl_block = (
            '<div class="card">\n'
            '<h2>📐 因子表达式 DSL (FactorEngine)</h2>\n'
            '<div class="formula-wrap" style="font-size:0.92rem;color:#1e4d8c;font-weight:500">'
            + dsl_esc + '</div>\n'
            + note_html +
            '</div>\n'
        )

    # Code block (原始 Python)
    if code:
        code_esc = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # 简单语法高亮: 关键字
        import re as _re
        for kw in ["def", "return", "import", "from", "if", "else", "elif", "for", "while",
                  "in", "and", "or", "not", "True", "False", "None", "class"]:
            code_esc = _re.sub(rf'\b({kw})\b', rf'<span class="kw">\1</span>', code_esc)
        code_esc = _re.sub(r'#[^\n]*', lambda m: f'<span class="com">{m.group(0)}</span>', code_esc)
        code_esc = _re.sub(r"'(.*?)'", lambda m: f"<span class=\"str\">'{m.group(1)}'</span>", code_esc)
        code_esc = _re.sub(r'"(.*?)"', lambda m: f'<span class="str">"{m.group(1)}"</span>', code_esc)
        code_block = (
            '<div class="card">\n'
            '<h2>🐍 原始 Python 代码 (factor_delivery)</h2>\n'
            '<div class="code-block">' + code_esc + '</div>\n'
            '</div>\n'
        )
    else:
        code_block = ""

    # IC月度图
    monthly_img = f'<img src="data:image/png;base64,{monthly_chart}" style="width:100%;border-radius:8px;"/>' if monthly_chart else '<div class="zero-notice">月度IC数据不足</div>'

    # 十分层图
    decile_img = f'<img src="data:image/png;base64,{decile_chart}" style="width:100%;border-radius:8px;"/>' if decile_chart else ""

    # 多空净值图
    ls_img = f'<img src="data:image/png;base64,{ls_chart}" style="width:100%;border-radius:8px;"/>' if ls_chart else ""

    # IC分布图
    dist_img = f'<img src="data:image/png;base64,{dist_chart}" style="width:100%;border-radius:8px;"/>' if dist_chart else ""

    # IC时序（SVG）
    ic_ts_section = f'''
<div class="card">
<h2>RankIC 时序</h2>
<div class="chart-wrap">{svg_timeseries}</div>
</div>''' if svg_timeseries else ""

    # 格式化公式（代码风格）
    formula_display = formula.replace("<", "&lt;").replace(">", "&gt;")

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="zh-CN">\n'
        '<head>\n'
        '<meta charset="utf-8"/>\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>\n'
        '<title>' + factor_name + '</title>\n'
        '<style>\n'
        ':root{'
        '--bg:' + BG + ';'
        '--panel:' + PANEL + ';'
        '--fg:' + FG + ';'
        '--muted:' + MUTED + ';'
        '--line:' + LINE + ';'
        '--primary:' + PRIMARY + ';'
        '--pos:' + POS + ';'
        '--neg:' + NEG + '}\n'
        '* { box-sizing:border-box }\n'
        'body { margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }\n'
        'header { background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:28px 48px 22px }\n'
        'header h1 { margin:0 0 6px; font-size:1.5rem; word-break:break-all }\n'
        'header .meta { opacity:0.85; font-size:0.85rem; margin-top:4px }\n'
        'main { max-width:1100px; margin:0 auto; padding:24px }\n'
        '.back { display:inline-block; margin-bottom:16px; color:#93c5fd; font-weight:500; text-decoration:none; font-size:0.88rem }\n'
        '.back:hover { text-decoration:underline }\n'
        '.grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px }\n'
        '.grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:12px }\n'
        '.metric { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; text-align:center; box-shadow:0 4px 24px rgba(15,23,42,0.06) }\n'
        '.metric b { display:block; font-size:1.35rem }\n'
        '.metric span { color:var(--muted); font-size:0.73rem }\n'
        '.pos { color:var(--pos) }\n'
        '.neg { color:var(--neg) }\n'
        '.cost-note { background:#f0fdf4; border:1px solid #bbf7d0; border-left:3px solid var(--pos); padding:8px 14px; border-radius:8px; font-size:0.78rem; color:#166534; margin-bottom:12px }\n'
        '.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 4px 24px rgba(15,23,42,0.06); margin-bottom:16px }\n'
        'h2 { font-size:0.95rem; color:var(--primary); margin:0 0 12px; border-bottom:1px solid var(--line); padding-bottom:8px }\n'
        '.formula-wrap { background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:16px; font-family:"Courier New",monospace; font-size:0.85rem; word-break:break-all; line-height:1.8; white-space:pre-wrap }\n'
        '.code-block { background:#0f172a; color:#e2e8f0; border-radius:8px; padding:16px 18px; font-family:"JetBrains Mono","Fira Code","Courier New",monospace; font-size:0.82rem; line-height:1.55; overflow-x:auto; white-space:pre; max-height:480px; border:1px solid #1e293b; box-shadow:0 4px 16px rgba(15,23,42,0.1) }\n'
        '.code-block .kw { color:#c084fc }\n'
        '.code-block .str { color:#86efac }\n'
        '.code-block .com { color:#64748b; font-style:italic }\n'
        '.code-block .num { color:#fcd34d }\n'
        '.code-tabs { display:flex; gap:8px; margin-bottom:10px }\n'
        '.code-tab { padding:4px 14px; border-radius:14px; font-size:0.78rem; font-weight:600; cursor:pointer; border:1px solid var(--line); background:#f1f5f9; color:var(--muted) }\n'
        '.code-tab.active { background:var(--primary); color:#fff; border-color:var(--primary) }\n'
        '.meta-table { width:100%; border-collapse:collapse; font-size:0.84rem }\n'
        '.meta-table th { background:#f1f5f9; color:var(--primary); padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid var(--primary) }\n'
        '.meta-table td { padding:7px 10px; border-bottom:1px solid var(--line) }\n'
        '.meta-table td:first-child { color:var(--muted); width:160px; font-weight:500 }\n'
        '.meta-table tr:hover td { background:#f8fafc }\n'
        '.chart-wrap { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px }\n'
        '.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.73rem; margin-left:6px }\n'
        '.badge-green { background:#dcfce7; color:#166534 }\n'
        '.badge-blue { background:#dbeafe; color:#1e40af }\n'
        '.badge-yellow { background:#fef3c7; color:#92400e }\n'
        '.badge-red { background:#fee2e2; color:#991b1b }\n'
        '.zero-notice { background:#fef9c3; border:1px solid #fde047; padding:12px 16px; border-radius:8px; font-size:0.85rem; color:#854d0e; margin-bottom:16px }\n'
        '.rationale { background:linear-gradient(90deg,#eff6ff,#f0fdfa); border-left:4px solid var(--primary); padding:12px 16px; border-radius:6px; margin-bottom:16px; font-size:0.88rem; color:#334155 }\n'
        'img { border-radius:8px }\n'
        '.qe-info { display:inline-block; background:linear-gradient(90deg,#ede9fe,#dbeafe); color:#5b21b6; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-left:8px }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<header>\n'
        '<a class="back" href="../index.html">&#8592; 返回汇总</a>\n'
        '<h1><code>' + factor_name + '</code>' + flip_note + qe_badge + '</h1>\n'
        '<div class="meta">本周新挖 · 回测区间 ' + (perf.get('start_date','—') if perf else '—') + ' ~ ' + (perf.get('end_date','—') if perf else '—') + ' · 评估: ' + ('quant_evaluator' if QE_AVAILABLE else 'legacy') + ' · 生成 ' + datetime.now().strftime('%Y-%m-%d %H:%M') + '</div>\n'
        '</header>\n'
        '<main>\n'
        '\n<!-- 核心指标 -->\n'
        '<div class="grid-4">\n'
        '<div class="metric"><b class="' + pcls(mean_rankic) + '">' + fmt(mean_rankic) + '</b><span>Mean RankIC</span></div>\n'
        '<div class="metric"><b class="' + pcls(rankic_ir) + '">' + fmt(rankic_ir) + '</b><span>RankIC IR</span></div>\n'
        '<div class="metric"><b class="' + pcls(ls_sharpe_gross) + '">' + fmt(ls_sharpe_gross) + '</b><span>LS Sharpe</span></div>\n'
        '<div class="metric"><b class="' + pcls(ls_annual_return) + '">' + fmt(ls_annual_return, pct=True) + '</b><span>LS 年化</span></div>\n'
        '</div>\n'
        '<div class="grid-4">\n'
        '<div class="metric"><b class="' + pcls(ls_annual_return * 2) + '">' + fmt(ls_annual_return * 2, pct=True) + '</b><span>LS 累计收益</span></div>\n'
        '<div class="metric"><b class="' + pcls(ls_mdd) + '">' + fmt(ls_mdd, pct=True) + '</b><span>LS 最大回撤</span></div>\n'
        '<div class="metric"><b class="' + pcls(ls_winrate) + '">' + fmt(ls_winrate, pct=True) + '</b><span>LS 日胜率</span></div>\n'
        '<div class="metric"><b class="' + pcls(turnover_daily) + '">' + fmt(turnover_daily, pct=True) + '</b><span>Top10% 换手率 (日)</span></div>\n'
        '</div>\n'
        '<div class="grid-3">\n'
        '<div class="metric"><b class="' + pcls(g10_ann) + '">' + fmt(g10_ann, pct=True) + '</b><span>G10 (多头) 年化</span></div>\n'
        '<div class="metric"><b class="' + pcls(g1_ann) + '">' + fmt(g1_ann, pct=True) + '</b><span>G1 (空头) 年化</span></div>\n'
        '<div class="metric"><b class="' + pcls(rankic_win_rate) + '">' + fmt(rankic_win_rate, pct=True) + '</b><span>RankIC 胜率</span></div>\n'
        '</div>\n'
        '<div class="cost-note">净值已扣双边交易成本 ' + f"{cost_bps:.1f} bp" + '（换手率 × 费率）；多空/十分层均含停牌股剔除（收益为 NaN 不参与）。</div>\n'
        '\n<!-- 设计意图 -->\n'
        + (rationale_html if rationale else '') +
        '\n<!-- 公式 -->\n'
        '<div class="card">\n'
        '<h2>因子路由 / 公式 ' + flip_note + (f' &nbsp; <span style="font-weight:400;color:var(--primary);font-size:0.92rem">{title}</span>' if title else '') + '</h2>\n'
        '<div class="formula-wrap">' + (dsl if dsl else formula.replace("<","&lt;").replace(">","&gt;")) + '</div>\n'
        + (f'<div style="margin-top:10px;font-size:0.82rem;color:var(--muted)">依赖列: <code>{required_columns}</code></div>' if required_columns else '') +
        '</div>\n'
        + ((
            '\n<!-- 计算步骤 -->\n'
            '<div class="card">\n'
            '<h2>📝 计算步骤 (一步一步解释)</h2>\n'
            '<ol style="line-height:1.9;padding-left:24px;font-size:0.92rem">' +
            ''.join(f'<li>{s.replace("<","&lt;").replace(">","&gt;")}</li>' for s in steps) +
            '</ol>\n'
            + (f'<div style="margin-top:12px;padding:10px 14px;background:#f0f9ff;border-left:3px solid #0ea5e9;border-radius:6px;font-size:0.85rem;color:#075985"><b>说明：</b>{manual_note.replace("<","&lt;").replace(">","&gt;")}</div>' if manual_note else '') +
            '</div>\n'
        ) if steps else '')
        + ((
            '\n<!-- 字段算子释义 -->\n'
            '<div class="card">\n'
            '<h2>📚 字段与算子释义</h2>\n'
            + _render_field_op_doc(required_columns, dsl, FIELD_DOC, OP_DOC) +
            '</div>\n'
        ) if (required_columns or dsl) else '')
        +
        '\n<!-- DSL 转换 -->\n'
        + dsl_block +
        '\n<!-- 输入字段释义 -->\n'
        + fields_html +
        '\n<!-- 算子释义 -->\n'
        + formula_steps_html +
        '\n<!-- 原始 Python Code -->\n'
        + code_block +
        ic_ts_section +
        '\n<!-- 月度IC热力图 -->\n'
        '<div class="card">\n'
        '<h2>月度 RankIC 热力图</h2>\n'
        + monthly_img + '\n'
        '</div>\n'
        '\n<!-- 十分层 -->\n'
        '<div class="card">\n'
        '<h2>十分层净值曲线 (G1~G10 + 多空)</h2>\n'
        + decile_img + '\n'
        '</div>\n'
        '\n<!-- 多空净值 -->\n'
        '<div class="card">\n'
        '<h2>多空净值曲线 (G10 - G1)</h2>\n'
        + ls_img + '\n'
        '</div>\n'
        '\n<!-- IC分布 -->\n'
        '<div class="card">\n'
        '<h2>RankIC 分布 (红线 = 均值)</h2>\n'
        + dist_img + '\n'
        '</div>\n'
        '\n<!-- 统计摘要 -->\n'
        '<div class="card">\n'
        '<h2>因子统计摘要</h2>\n'
        '<table class="meta-table">\n'
        '<thead><tr><th>指标</th><th>值</th></tr></thead>\n'
        '<tbody>\n'
        '<tr><td>因子名称</td><td><code>' + factor_name + '</code></td></tr>\n'
        '<tr><td>回测区间</td><td>' + (perf.get('start_date','—') if perf else '—') + ' ~ ' + (perf.get('end_date','—') if perf else '—') + '</td></tr>\n'
        '<tr><td>回测交易日</td><td>' + str(n_periods) + ' 天</td></tr>\n'
        '<tr><td>已翻转</td><td>' + ('是 <span class="badge badge-yellow">IC&lt;0 时取负调正（已翻正）</span>' if is_flipped else '否') + '</td></tr>\n'
        '<tr><td>评估库</td><td>' + ('<code>quant_evaluator</code>' if QE_AVAILABLE else 'legacy') + '</td></tr>\n'
        '<tr><td>Mean RankIC</td><td class="' + pcls(mean_rankic) + '">' + fmt(mean_rankic) + '</td></tr>\n'
        '<tr><td>RankIC 标准差</td><td>' + fmt(ic_stats.get("std_ric", 0)) + '</td></tr>\n'
        '<tr><td>RankIC IR</td><td class="' + pcls(rankic_ir) + '">' + fmt(rankic_ir) + '</td></tr>\n'
        '<tr><td>RankIC 胜率</td><td class="' + pcls(win_rate) + '">' + fmt(win_rate, pct=True) + '</td></tr>\n'
        '<tr><td>LS Sharpe</td><td class="' + pcls(ls_sharpe_gross) + '">' + fmt(ls_sharpe_gross) + '</td></tr>\n'
        '<tr><td>LS 年化收益</td><td class="' + pcls(ls_annual_return) + '">' + fmt(ls_annual_return, pct=True) + '</td></tr>\n'
        '<tr><td>LS 累计收益</td><td class="' + pcls(ls_annual_return * 2) + '">' + fmt(ls_annual_return * 2, pct=True) + '</td></tr>\n'
        '<tr><td>LS 最大回撤</td><td class="' + pcls(ls_mdd) + '">' + fmt(ls_mdd, pct=True) + '</td></tr>\n'
        '<tr><td>LS 日胜率</td><td class="' + pcls(ls_winrate) + '">' + fmt(ls_winrate, pct=True) + '</td></tr>\n'
        '<tr><td>Top10% 换手率 (日)</td><td>' + (fmt(perf.get('turnover', 0), pct=True) if perf else '—') + '</td></tr>\n'
        '<tr><td>双边费率假设</td><td>' + f"{cost_bps:.1f} bp（按换手率×费率扣减）" + '</td></tr>\n'
        '<tr><td>G10 (多头) 年化</td><td class="' + pcls(g10_ann) + '">' + fmt(g10_ann, pct=True) + '</td></tr>\n'
        '<tr><td>G1 (空头) 年化</td><td class="' + pcls(g1_ann) + '">' + fmt(g1_ann, pct=True) + '</td></tr>\n'
        '</tbody></table>\n'
        '</div>\n'
        '\n</main>\n'
        '</body>\n'
        '</html>\n'
    )
    return html


# ============================================================
# 主流程
# ============================================================
def process_factor(name: str) -> tuple[str, bool, dict]:
    """处理单个因子，返回 (name, success, stats)
    stats包含: ic, mean_ic, ic_ir, sharpe, annual, mdd, win_rate, g10_ann, g1_ann"""
    try:
        # 1. 提取公式
        fi = extract_formula_info(name)
        formula = fi["formula"]
        code = fi["code"]
        is_flipped = fi["is_flipped"] or _is_flipped_by_meta(name)

        # 2. 用真因子值算所有指标
        fm = _compute_factor_metrics(name)
        ic_series = fm.get("ic_series", pd.Series(dtype=float))
        ic_stats = compute_ic_stats(ic_series)

        # decile_data: 新的 key 命名 ('G1'..'G10', 'LS', 'dates')
        decile_raw = fm.get("decile_navs", {})
        dates_out = fm.get("dates_out", [])
        decile_dates = decile_raw.get("dates") or dates_out
        decile_data = {"dates": decile_dates}
        for k, v in decile_raw.items():
            if k == "LS":
                decile_data["LS"] = v
            elif k != "dates":
                decile_data[k] = v

        perf = fm.get("perf", {})

        # 3. 生成图表
        monthly_chart = plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name, is_flipped) if ic_stats else ""
        decile_chart = plot_decile_nav(decile_data, name, is_flipped) if decile_data else ""
        ls_chart = plot_long_short_nav(decile_data, name, is_flipped) if decile_data else ""
        dist_chart = plot_ic_distribution(ic_series, name, is_flipped) if len(ic_series) > 0 else ""
        svg_ts = plot_ic_timeseries(ic_series, name, is_flipped) if len(ic_series) > 0 else ""

        # 4. 构建HTML (传 perf + mean_ic 等)
        html = build_detail_html(
            factor_name=name,
            formula=fi["formula"],
            code=fi["code"],
            ic_stats=ic_stats,
            decile_data=decile_data,
            svg_timeseries=svg_ts,
            monthly_chart=monthly_chart,
            decile_chart=decile_chart,
            ls_chart=ls_chart,
            dist_chart=dist_chart,
            is_flipped=is_flipped,
            perf=perf,
            dsl=fi.get("dsl", ""),
            rationale=fi.get("rationale", ""),
            required_columns=fi.get("required_columns", ""),
            dsl_status=fi.get("dsl_status", "ok"),
            dsl_note=fi.get("dsl_note", ""),
            title=fi.get("title", ""),
            steps=fi.get("steps", []),
            manual_note=fi.get("manual_note", ""),
        )
        out_path = FACTORS_DIR / f"factor_{name}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        return (name, True, {
            "name": name,
            "mean_ic": fm.get("mean_ic", 0),
            "ic_ir": fm.get("ic_ir", 0),
            "ls_sharpe": perf.get("ls_sharpe", 0),
            "ls_annual": perf.get("ls_annual", 0),
            "ls_mdd": perf.get("ls_mdd", 0),
            "win_rate": fm.get("win_rate", 0),
            "g10_annual": perf.get("g10_annual", 0),
            "g1_annual": perf.get("g1_annual", 0),
        })
    except Exception as e:
        return (name, False, {"error": str(e)[:200]})


def process_factor_thread(name: str, batch_metrics: dict) -> tuple:
    """线程版 process_factor - 直接用主进程已算好的 batch_metrics"""
    try:
        fi = extract_formula_info(name)
        is_flipped = fi["is_flipped"] or _is_flipped_by_meta(name)
        # batch_metrics 的 key = per-factor parquet 文件名（page_name，不带 factor_ 前缀）
        fm = batch_metrics.get(name, {})
        if not fm:
            fm = batch_metrics.get(f"factor_{name}", {})
        if not fm:
            fm = batch_metrics.get(f"factor_{name}_flipped", {})

        ic_series = fm.get("ic_series", pd.Series(dtype=float)).copy()
        ic_series = ic_series.astype(float).replace([np.inf, -np.inf], np.nan)

        # 兜底: 部分 _flipped 因子因 parquet 无数据, 拿到非 flipped 数据是负的, 强制取反
        if is_flipped and len(ic_series) > 0:
            if ic_series.mean() < 0:
                ic_series = -ic_series
                fm = dict(fm)
                fm["perf"] = dict(fm.get("perf", {}))
                # 镜像翻转：LS 收益取负 → 胜率 = 1 - 原胜率（不是 -胜率）
                for k in ["ls_sharpe", "ls_annual", "g10_annual", "g1_annual", "g10_sharpe", "g1_sharpe"]:
                    if k in fm["perf"]:
                        fm["perf"][k] = -fm["perf"][k]
                if "ls_winrate" in fm["perf"]:
                    fm["perf"]["ls_winrate"] = 1.0 - fm["perf"].get("ls_winrate", 0)
                if "win_rate" in fm:
                    fm["win_rate"] = 1.0 - fm.get("win_rate", 0)
                fm["perf"]["ls_mdd"] = abs(fm["perf"].get("ls_mdd", 0))
                # mirror decile NAVs (G_k ↔ G_{11-k})
                decile_raw = dict(fm.get("decile_navs", {}))
                gs = []
                for k in range(1, 11):
                    key = f"G{k}"
                    if key in decile_raw and len(decile_raw[key]) > 0:
                        gs.append((k, decile_raw[key]))
                if len(gs) == 10:
                    new_dec = {"dates": decile_raw.get("dates", [])}
                    for new_k, (_, v) in zip(range(1, 11), reversed(gs)):
                        new_dec[f"G{new_k}"] = v
                    ls_orig = decile_raw.get("LS", [])
                    if ls_orig and len(ls_orig) > 0:
                        new_dec["LS"] = [1.0 / x if x not in (0, None) else 0.0 for x in ls_orig]
                    fm["decile_navs"] = new_dec
                fm["mean_ic"] = -fm.get("mean_ic", 0)
                fm["ic_ir"] = -fm.get("ic_ir", 0)

        decile_raw = fm.get("decile_navs", {})
        dates_out = fm.get("dates_out", [])
        # 2026-08-29 fix: mirror 翻转时 new_dec 重建了 decile_navs（含 'dates'），
        # 而 fm['dates_out'] 可能为空 → 用 decile_raw 自带的 dates 兜底，避免多空图因 dates 为空而缺失。
        decile_dates = decile_raw.get("dates") or dates_out
        decile_data = {"dates": decile_dates}
        for k, v in decile_raw.items():
            if k == "dates":
                continue
            decile_data[k] = v
        perf = fm.get("perf", {})

        ic_stats = compute_ic_stats(ic_series)

        monthly_chart = plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name, is_flipped) if ic_stats else ""
        decile_chart = plot_decile_nav(decile_data, name, is_flipped) if decile_data else ""
        ls_chart = plot_long_short_nav(decile_data, name, is_flipped) if decile_data else ""
        dist_chart = plot_ic_distribution(ic_series, name, is_flipped) if len(ic_series) > 0 else ""
        svg_ts = plot_ic_timeseries(ic_series, name, is_flipped) if len(ic_series) > 0 else ""

        html = build_detail_html(
            factor_name=name,
            formula=fi["formula"],
            code=fi["code"],
            ic_stats=ic_stats,
            decile_data=decile_data,
            svg_timeseries=svg_ts,
            monthly_chart=monthly_chart,
            decile_chart=decile_chart,
            ls_chart=ls_chart,
            dist_chart=dist_chart,
            is_flipped=fi["is_flipped"],
            perf=perf,
            dsl=fi.get("dsl", ""),
            rationale=fi.get("rationale", ""),
            required_columns=fi.get("required_columns", ""),
            dsl_status=fi.get("dsl_status", "ok"),
            dsl_note=fi.get("dsl_note", ""),
            title=fi.get("title", ""),
            steps=fi.get("steps", []),
            manual_note=fi.get("manual_note", ""),
        )
        out_path = FACTORS_DIR / f"factor_{name}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return (name, True, {
            "name": name,
            "mean_ic": fm.get("mean_ic", 0),
            "ic_ir": fm.get("ic_ir", 0),
            "ls_sharpe": perf.get("ls_sharpe", 0),
            "ls_annual": perf.get("ls_annual", 0),
            "ls_mdd": perf.get("ls_mdd", 0),
            "win_rate": fm.get("win_rate", 0),
            "g10_annual": perf.get("g10_annual", 0),
            "g1_annual": perf.get("g1_annual", 0),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (name, False, {"error": str(e)[:200]})


def main():
    print("=" * 60)
    print("重建因子详情页 — 含真实公式 + 全维度评估图 (quant_evaluator)")
    print("=" * 60)

    names = load_weekly_factor_names()
    print(f"本周因子数: {len(names)}")

    # 预览前3个
    print("\n前3个因子公式预览:")
    for name in names[:3]:
        fi = extract_formula_info(name)
        print(f"  {name}: {fi['formula'][:80]}")

    # ============ 一次性算全 61 因子指标（向量化）============
    print("\n[主进程] 一次性算所有因子指标 (quant_evaluator.compute_daily_ic + compute_long_short_returns) ...")
    t0 = time.time()
    batch_metrics = _compute_all_factors_metrics()
    print(f"[主进程] 指标算完: {len(batch_metrics)} 因子, 用时 {time.time()-t0:.1f}s")

    # ============ ThreadPool 渲染 HTML（matplotlib 在 thread 安全用 Agg backend）============
    print(f"\n开始并行渲染 {len(names)} 个详情页 (ThreadPool)...")
    n_workers = min(8, max(2, (os.cpu_count() or 4) // 4))
    results = []
    all_stats = []
    errors = []
    done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_factor_thread, n, batch_metrics): n for n in names}
        for future in as_completed(futures):
            name, ok, stats = future.result()
            done += 1
            if ok:
                results.append(name)
                stats["name"] = name
                stats["is_flipped"] = "_flipped" in name
                all_stats.append(stats)
            else:
                errors.append(name)
            if done % 10 == 0 or done == len(names):
                print(f"  进度 {done}/{len(names)}, 成功 {len(results)}, 失败 {len(errors)}")

    print(f"\n完成！成功 {len(results)}, 失败 {len(errors)}")
    if errors:
        print(f"失败因子: {errors[:10]}")

    # 输出汇总：按 Sharpe 排序
    if all_stats:
        print("\n按 LS Sharpe 排序 (TOP15 + BOTTOM10):")
        sorted_stats = sorted(all_stats, key=lambda s: s.get("ls_sharpe", 0), reverse=True)
        print(f"{'因子':<55} {'Sharpe':>7} {'年化':>7} {'IC':>8} {'ICIR':>7} {'回撤':>7}")
        for s in sorted_stats[:15]:
            n = s['name'][:53]
            print(f"{n:<55} {s.get('ls_sharpe',0):>+7.2f} {s.get('ls_annual',0)*100:>+6.1f}% {s.get('mean_ic',0):>+8.4f} {s.get('ic_ir',0):>+7.2f} {s.get('ls_mdd',0)*100:>+6.1f}%")
        print("\n... BOTTOM10:")
        for s in sorted_stats[-10:]:
            n = s['name'][:53]
            print(f"{n:<55} {s.get('ls_sharpe',0):>+7.2f} {s.get('ls_annual',0)*100:>+6.1f}% {s.get('mean_ic',0):>+8.4f} {s.get('ic_ir',0):>+7.2f} {s.get('ls_mdd',0)*100:>+6.1f}%")

        with open(REPORT_DIR / "summary_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"\n汇总已存到 {REPORT_DIR / 'summary_stats.json'}")

    update_index_links(names)
    print("\n全部完成！刷新浏览器即可查看。")


def update_index_links(names: list[str]):
    """确保 index.html 链接正确"""
    idx_path = REPORT_DIR / "index.html"
    if not idx_path.exists():
        print("index.html 不存在，跳过链接更新")
        return

    with open(idx_path) as f:
        content = f.read()

    # 检查是否已有因子链接
    existing = set(re.findall(r'href="factors/factor_([^"]+)\.html"', content))
    missing = [n for n in names if n not in existing]

    if missing:
        print(f"index.html 缺少 {len(missing)} 个因子链接（可手动添加 factors/factor_{{name}}.html）")
    else:
        print("index.html 链接完整")


if __name__ == "__main__":
    main()
