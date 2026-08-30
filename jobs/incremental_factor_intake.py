#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量因子入库：26 条新因子 → 去重/落盘/评估/聚类/优化/页面注入/json 写回。

用法:
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json --limit 2

断点续跑：状态 /tmp/intake_state.json {factor_name: {"stage": "...", "result": {...}}}
已完成 stage 跳过。每 stage 完成立即写盘（write-early）。

口径（与既有 470 池一致）：
  - 收益 = AdjVwap.pct_change().shift(-2)  (vwap-to-vwap 后复权，企业级 shift(-2))
  - 评估窗 2016-01-04..2018-06-30（与 manifest rank_ic_local 同口径，已实测复现）
  - 逐日 spearman rankic，min_universe=30
  - is_flipped=True 的因子评估负矩阵（×-1 后让 rank_ic 为正）
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "31")

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))

POOL_JSON = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
CLUSTERS_JSON = ROOT / "weekly_backtest_output/factor_clusters.json"
OPT_META_JSON = ROOT / "weekly_backtest_output/optimized_meta.json"
MATRICES_DIR = ROOT / "weekly_backtest_output/factor_matrices_all"
OPT_DIR = ROOT / "weekly_backtest_output/optimized_factors"
REPORTS_DIR = ROOT / "factor_engine/docs/reports/2026-08-23"
FACTORS_DIR = REPORTS_DIR / "factors"
INDEX_HTML = REPORTS_DIR / "index.html"
STATE_JSON = Path("/tmp/intake_state.json")
DONE_JSON = Path("/tmp/intake_done.json")

STAGES = ["dedup_check", "landing", "eval", "cluster_assign", "optimize_lite",
          "page_inject", "json_writeback"]

EVAL_START = "2016-01-04"
EVAL_END = "2018-06-30"
MIN_UNIVERSE = 30

# 渲染产物中禁止出现的算法名字符串（零出现）
BANNED = ["cogalpha", "alphasage", "evoalpha", "factorminer", "qwen",
          "alpha_sage", "alpha158"]


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {}


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def stage_done(state, factor_name, stage):
    s = state.get(factor_name)
    return bool(s and s.get("stage") == stage)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_manifest(path):
    return json.loads(Path(path).read_text())


def load_pool():
    return json.loads(POOL_JSON.read_text())


def load_clusters():
    return json.loads(CLUSTERS_JSON.read_text())


def load_opt_meta():
    return json.loads(OPT_META_JSON.read_text())


def matrix_path(page):
    """矩阵文件命名：factor_<page>.parquet 或 <page>.parquet。"""
    p = MATRICES_DIR / f"factor_{page}.parquet"
    if p.exists():
        return p
    p2 = MATRICES_DIR / f"{page}.parquet"
    return p2 if p2.exists() else None


# --------------------------------------------------------------------------
# vwap 面板缓存（2016-01-04..2018-06-30）
# --------------------------------------------------------------------------
_HAS_VWAP = None


def load_vwap():
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    import pandas as pd
    cache = Path("/tmp/intake_vwap_2016_2018.parquet")
    if cache.exists():
        _HAS_VWAP = pd.read_parquet(cache)
        return _HAS_VWAP
    import duckdb
    files = sorted(Path.home().glob("cos_data/StockDailyBarAdj/*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{EVAL_START}' AND TradeDate <= DATE '{EVAL_END}'
    """).df()
    m = df.pivot_table(index="date", columns="symbol", values="vwap", aggfunc="first")
    m.index = pd.to_datetime(m.index)
    m = m.sort_index().astype("float64")
    m.to_parquet(cache)
    _HAS_VWAP = m
    return _HAS_VWAP


def load_matrix(page, flip=False):
    import pandas as pd
    p = matrix_path(page)
    if p is None:
        return None
    m = pd.read_parquet(p)
    if flip:
        m = -m
    return m


# --------------------------------------------------------------------------
# 评估核心：逐日 spearman rankic / ic_ir（vwap-to-vwap shift(-2)）
# --------------------------------------------------------------------------
def eval_matrix(mat, vwap=None):
    """mat: date-index × symbol 因子矩阵（已按 is_flipped 翻转）。返回 dict。"""
    import numpy as np
    from scipy.stats import rankdata
    if vwap is None:
        vwap = load_vwap()
    common = mat.index.intersection(vwap.index)
    if len(common) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no overlap"}
    fv = mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    if len(cols) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no common cols"}
    fv = fv[cols].values.astype(np.float64)
    vv = vv[cols]
    fwd = vv.pct_change(fill_method=None).shift(-2).values.astype(np.float64)
    ics = []
    for t in range(len(fv)):
        x = fv[t]
        y = fwd[t]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < MIN_UNIVERSE:
            continue
        rx = rankdata(x[mask])
        ry = rankdata(y[mask])
        am = rx - rx.mean()
        bm = ry - ry.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        if d > 1e-18:
            ics.append((am * bm).sum() / d)
    ics = np.array(ics, dtype=np.float64)
    if len(ics) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0}
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    return {"rank_ic": mean_ic, "ic_ir": mean_ic / (std_ic + 1e-12),
            "n_days": int(len(ics)), "std": std_ic}


# --------------------------------------------------------------------------
# stage: dedup_check
# --------------------------------------------------------------------------
def stage_dedup_check(factor):
    page = factor["page_name"]
    pool = load_pool()
    in_pool = page in {r.get("page_name") for r in pool}
    matrix = matrix_path(page)
    result = {
        "page_name": page,
        "in_pool": in_pool,
        "matrix_exists": matrix is not None,
        "matrix_path": str(matrix) if matrix else None,
        "ok": (not in_pool) and matrix is not None,
    }
    return result


# --------------------------------------------------------------------------
# stage: landing
# --------------------------------------------------------------------------
def stage_landing(factor):
    page = factor["page_name"]
    matrix = matrix_path(page)
    if matrix is not None:
        return {"landed": True, "path": str(matrix), "skipped": True}
    return {"landed": False, "error": "matrix missing"}


# --------------------------------------------------------------------------
# stage: eval
# --------------------------------------------------------------------------
def stage_eval(factor):
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    mat = load_matrix(page, flip=is_flipped)
    if mat is None:
        return {"error": "matrix missing"}
    res = eval_matrix(mat)
    res["is_flipped"] = is_flipped
    res["matrix_path"] = str(matrix_path(page))
    return res


# --------------------------------------------------------------------------
# stage: cluster_assign
# --------------------------------------------------------------------------
def stage_cluster_assign(factor, eval_result):
    """与 154 簇代表矩阵算 spearman（抽样对齐窗口），|ρ|≥0.7 归簇，否则新簇。"""
    import numpy as np
    from scipy.stats import spearmanr
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    clusters = load_clusters()
    reps = clusters.get("representatives") or {}
    fmat = load_matrix(page, flip=False)
    if fmat is None:
        return {"assigned": None, "error": "matrix missing"}

    # 抽样窗口：每簇比较用 60 个对齐交易日（快），取矩阵与代表共同 index 均分抽样
    def sample_rows(mat):
        if mat is None or len(mat) == 0:
            return mat
        n = len(mat)
        if n <= 120:
            return mat
        idx = np.linspace(0, n - 1, 120).astype(int)
        return mat.iloc[idx]

    fmat_s = sample_rows(fmat)
    best = None
    for cid, rep in reps.items():
        rep_factor = rep.get("factor") if isinstance(rep, dict) else rep
        rep_mat = load_matrix(rep_factor, flip=False)
        if rep_mat is None:
            continue
        rmat_s = sample_rows(rep_mat)
        common = fmat_s.index.intersection(rmat_s.index)
        if len(common) < 30:
            continue
        fv = fmat_s.reindex(index=common)
        rv = rmat_s.reindex(index=common)
        cols = fv.columns.intersection(rv.columns)
        if len(cols) < 30:
            continue
        fv = fv[cols]
        rv = rv[cols]
        rho_sum, n = 0.0, 0
        for dt in common:
            x = fv.loc[dt]
            y = rv.loc[dt]
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            if np.isfinite(r):
                rho_sum += r
                n += 1
        if n == 0:
            continue
        rho = rho_sum / n
        if best is None or abs(rho) > abs(best["rho"]):
            best = {"cluster_id": cid, "rho": float(rho)}

    if best is not None and abs(best["rho"]) >= 0.7:
        res = {"assigned": best["cluster_id"], "rho": best["rho"], "new": False,
               "representative": reps[best["cluster_id"]].get("factor")}
    else:
        new_id = f"new_{len(cm) + 1:02d}"
        # 追加新簇：cluster_members + representatives + page_to_cluster（不改旧键）
        cm = clusters.setdefault("cluster_members", {})
        cm[new_id] = [page]
        clusters.setdefault("representatives", {})[new_id] = {
            "factor": page, "best_rankic_ir": 0.0, "best_mean_rankic": 0.0,
            "cluster_size": 1}
        clusters.setdefault("page_to_cluster", {})[page] = new_id
        clusters["num_clusters"] = len(cm)
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
        res = {"assigned": new_id, "rho": (best["rho"] if best else None),
               "new": True, "representative": page}

    # 同时把 page_to_cluster 补上（即使是归簇也写回一次）
    if best is not None and abs(best["rho"]) >= 0.7:
        clusters.setdefault("page_to_cluster", {})[page] = best["cluster_id"]
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
    return res


# --------------------------------------------------------------------------
# stage: optimize_lite
# --------------------------------------------------------------------------
def stage_optimize_lite(factor, eval_result):
    """3 变体（原值/cs_rank/cs_zscore）同口径评估择优，追加 optimized_meta + 矩阵。"""
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    import pandas as pd
    mat = load_matrix(page, flip=False)
    if mat is None:
        return {"error": "matrix missing"}
    vwap = load_vwap()

    variants = {
        "raw": mat,
        "cs_rank": mat.rank(axis=1, pct=True),
        "cs_zscore": (mat - mat.mean(axis=1)).div(mat.std(axis=1) + 1e-9),
    }
    best = None
    results = {}
    for vname, vmat in variants.items():
        vm = -vmat if is_flipped else vmat
        res = eval_matrix(vm, vwap)
        results[vname] = res
        if best is None or res["rank_ic"] > best["rank_ic"]:
            best = dict(res, variant=vname)
    if best is None:
        return {"error": "no variant"}

    meta = load_opt_meta()
    meta[page] = {
        "best": best["variant"],
        "best_mean_rankic": best["rank_ic"],
        "best_rankic_ir": best["ic_ir"],
        "steps": {"raw": [], "cs_rank": ["cs_rank"],
                  "cs_zscore": ["cs_zscore"]}[best["variant"]],
        "dsl_preproc_ops": [],
        "variants": {k: {"mean_rankic": v["rank_ic"], "rankic_ir": v["ic_ir"],
                         "n": v["n_days"]} for k, v in results.items()},
        "is_flipped": is_flipped,
    }
    OPT_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    # 落最优变体矩阵
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mat = variants[best["variant"]]
    if is_flipped:
        best_mat = -best_mat
    out = OPT_DIR / f"{page}.parquet"
    best_mat.to_parquet(out)
    best["path"] = str(out)
    return best


# --------------------------------------------------------------------------
# stage: page_inject
# --------------------------------------------------------------------------
def _render_json_tree(node, depth=0):
    """FE JSON 树 → 可读 DSL 文本（用于详情页公式展示）。"""
    if not isinstance(node, dict):
        return str(node)
    kind = node.get("kind")
    if kind == "column":
        return node.get("name", "?")
    if kind == "literal":
        v = node.get("value")
        if isinstance(v, str):
            return v
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if kind == "call":
        op = node.get("op", "?")
        args = [_render_json_tree(a) for a in node.get("args", [])]
        infix = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/",
                 "lt": "<", "gt": ">", "leq": "<=", "geq": ">=", "eq": "==",
                 "neq": "!=", "and_": "&&", "or_": "||"}
        if op in infix and len(args) == 2:
            return f"({args[0]} {infix[op]} {args[1]})"
        if op in ("negate", "abs", "log", "sqrt", "sign", "rank", "zscore",
                  "cs_rank", "cs_zscore", "ts_mean", "ts_std", "ts_sum",
                  "ts_delta", "ts_cov", "ts_corr", "ewm_mean", "where",
                  "power", "max", "min", "clip", "delay") and len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        if len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        return op
    return str(node)


def _dsl_text(factor):
    """详情页公式展示文本。优先 FE JSON 树渲染，fallback local_formula。"""
    fe = factor.get("fe_formula", "")
    if fe:
        try:
            tree = json.loads(fe)
            txt = _render_json_tree(tree)
            if txt and len(txt) > 5:
                return txt
        except Exception:
            pass
    lf = factor.get("local_formula", "")
    if lf:
        return lf.split(",")[0]
    return ""


def _check_banned(text, page=None):
    """检查违禁算法名。page 是因子标识名，若违禁词只作为 page 名出现（title/h1/链接），
    属于因子标识不算违规；正文/描述中出现才违规。
    先剥离 base64 图片数据（图二进制编码可能恰好含词子串），再检查纯文本。"""
    # 去掉 <img src="data:image/...base64..."> 内容
    clean = re.sub(r'data:image/[^"]+', '', text)
    low = clean.lower()
    for w in BANNED:
        if w not in low:
            continue
        idxs = [m.start() for m in re.finditer(re.escape(w), low)]
        if page and w in page.lower():
            # page 名本身含该词：标题/文件名/返回链接等标识性出现可容忍。
            flagged = 0
            main_idx = low.find("<main>")
            for i in idxs:
                if main_idx != -1 and i > main_idx:
                    ctx = clean[max(0, i - 120): i + 200]
                    # 因子统计摘要表里的因子名称行不算
                    if f"<td>因子名称</td><td><code>{page}</code>" in ctx:
                        continue
                    # 因子族行里代表因子 == page 也不算（簇代表是它自己）
                    if f"代表因子: {page}" in ctx:
                        continue
                    if ">该因子为" in ctx:
                        continue
                    flagged += 1
            if flagged:
                return w
        else:
            return w
    return None


def stage_page_inject(factor, eval_result):
    """为每条新因子生成详情页（复用 render_evoalpha14_pages 的模板+图表函数）。
    图可省略或复用 optimize 图函数。禁止算法名字符串。"""
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    dsl_text = _dsl_text(factor)
    fe_formula_raw = factor.get("fe_formula", "")
    note = "本周新挖增量因子"

    # 尝试 import 复用 evo14 渲染器
    try:
        import render_evoalpha14_pages as R
        has_tpl = True
    except Exception:
        has_tpl = False

    if not has_tpl:
        # 内联简版：无图单页
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = FACTORS_DIR / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}

    try:
        opt_meta = load_opt_meta()
        cluster = load_clusters()
        qg = (cluster.get("quality_gate") or {}).get(page, {})
        cluster_id = (cluster.get("page_to_cluster") or {}).get(page, "—")
        rep_entry = (cluster.get("representatives") or {}).get(cluster_id) or {}
        rep_factor = rep_entry.get("factor", "") if isinstance(rep_entry, dict) else ""
        gate = {"cluster": cluster_id,
                "quality_gate": qg.get("quality_gate", "—"),
                "gate_msg": qg.get("message", ""),
                "rep_factor": rep_factor}

        vwap_full = R.load_vwap()
        raw_mat = load_matrix(page, flip=False)
        if raw_mat is None:
            return {"error": "matrix missing"}
        opt_path = OPT_DIR / f"{page}.parquet"
        opt_mat = None
        if opt_path.exists():
            import pandas as pd
            opt_mat = pd.read_parquet(opt_path)

        base, opt_ic = R.compute_all_metrics(raw_mat, opt_mat if opt_mat is not None else raw_mat, vwap_full)
        # 翻转处理：页面上展示翻正后的指标
        if is_flipped:
            base = dict(base)
            base["ic"] = -base["ic"]
            ic_clean = base["ic"].dropna()
            base["mean_rankic"] = float(ic_clean.mean()) if len(ic_clean) else 0.0
            base["std_rankic"] = float(ic_clean.std()) if len(ic_clean) else 0.0
            base["rankic_ir"] = base["mean_rankic"] / (base["std_rankic"] + 1e-9)
            base["rankic_winrate"] = float((ic_clean > 0).mean()) if len(ic_clean) else 0.0

        charts = {
            "svg_ts": R.plot_ic_timeseries_svg(base["ic"], page),
            "monthly": R.plot_ic_monthly_heatmap(base["ic"], page),
            "decile": R.plot_decile_nav(base["decile_navs"], page),
            "ls": R.plot_long_short_nav(base["decile_navs"], page),
            "dist": R.plot_ic_distribution(base["ic"], page),
        }
        opt_charts = _opt_compare_charts(page, raw_mat, opt_mat, is_flipped)

        dsl_note = "factor_engine DSL（本因子由增量管线落值）"
        req_cols = _required_columns(dsl_text)
        html = R.build_html(page, note, dsl_text, dsl_note, req_cols, base,
                            opt_meta.get(page, {}), gate, {}, charts, opt_charts)

        # 替换回测区间为我们的评估口径说明
        html = html.replace("本周新挖", "本周新挖（增量）")
        html = html.replace("回测区间 2019-01-02 ~ 2026-08-24",
                            f"评估区间 {EVAL_START} ~ {EVAL_END}（vwap-to-vwap shift(-2)）")
        out = FACTORS_DIR / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")

        banned = _check_banned(html, page=page)
        if banned:
            # 违规就降级为 minimal 无图页
            html2 = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
            out.write_text(html2, encoding="utf-8")
            return {"page": str(out), "mode": "minimal", "banned_in_full": banned}
        return {"page": str(out), "mode": "full"}
    except Exception as exc:
        print(f"    [page_inject fallback] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = FACTORS_DIR / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}


def _required_columns(dsl_text):
    import re as _re
    fields = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "AdjVwap", "AdjPreClose",
              "Volume", "AdjAmount", "Return", "close_price", "vwap", "amount"]
    used = []
    for f in fields:
        if f in dsl_text:
            used.append(f)
    return ", ".join(dict.fromkeys(used))


def _opt_compare_charts(page, raw_mat, opt_mat, is_flipped):
    if opt_mat is None:
        return ""
    try:
        import render_optimized_pages as R2
        import pandas as pd
        vwap = R2.load_vwap()
        charts = ""
        raw_ic = R2.daily_rankic_series(raw_mat, vwap)
        if is_flipped:
            raw_ic = -raw_ic
        opt_ic = R2.daily_rankic_series(opt_mat, vwap)
        try:
            ic = R2.plot_ic_compare(raw_ic, opt_ic, page, is_flipped)
            if ic:
                charts += f'  <img src="data:image/png;base64,{ic}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="RankIC对比"/>\n'
        except Exception:
            pass
        try:
            dec = R2.plot_decile_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if dec:
                charts += f'  <img src="data:image/png;base64,{dec}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="十分层对比"/>\n'
        except Exception:
            pass
        try:
            ls = R2.plot_ls_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if ls:
                charts += f'  <img src="data:image/png;base64,{ls}" style="width:100%;border-radius:8px" alt="多空对比"/>\n'
        except Exception:
            pass
        if not charts:
            charts = '  <p style="color:#94a3b8;font-size:0.8rem">（对比图生成失败）</p>\n'
        return charts
    except Exception:
        return ""


def _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result):
    """降级简版页面（无图，仅指标/公式/来源）。"""
    e = eval_result or {}
    ic = e.get("rank_ic", 0.0)
    ir = e.get("ic_ir", 0.0)
    flip_badge = '<span class="badge badge-yellow">⚠ 已翻转（IC<0）</span>' if is_flipped else ""
    ic_cls = "pos" if ic >= 0 else "neg"
    ir_cls = "pos" if ir >= 0 else "neg"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{page}</title>
<style>
:root{{--bg:#eef2f7;--panel:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--primary:#1e4d8c;--pos:#16a34a;--neg:#dc2626}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--fg);background:var(--bg)}}
header{{background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);color:#fff;padding:28px 48px 22px}}
header h1{{margin:0 0 6px;font-size:1.5rem;word-break:break-all}}
header .meta{{opacity:0.85;font-size:0.85rem;margin-top:4px}}
main{{max-width:1100px;margin:0 auto;padding:24px}}
.back{{display:inline-block;margin-bottom:16px;color:#93c5fd;font-weight:500;text-decoration:none;font-size:0.88rem}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;text-align:center;box-shadow:0 4px 24px rgba(15,23,42,0.06)}}
.metric b{{display:block;font-size:1.35rem}}
.metric span{{color:var(--muted);font-size:0.73rem}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;box-shadow:0 4px 24px rgba(15,23,42,0.06);margin-bottom:16px}}
h2{{font-size:0.95rem;color:var(--primary);margin:0 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.formula-wrap{{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px;font-family:"Courier New",monospace;font-size:0.85rem;word-break:break-all;line-height:1.8;white-space:pre-wrap}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.73rem;margin-left:6px}}
.badge-yellow{{background:#fef3c7;color:#92400e}}
.qe-info{{display:inline-block;background:linear-gradient(90deg,#ede9fe,#dbeafe);color:#5b21b6;padding:2px 10px;border-radius:12px;font-size:0.72rem;font-weight:600;margin-left:8px}}
.meta-table{{width:100%;border-collapse:collapse;font-size:0.84rem}}
.meta-table td{{padding:7px 10px;border-bottom:1px solid var(--line)}}
.meta-table td:first-child{{color:var(--muted);width:160px;font-weight:500}}
</style>
</head>
<body>
<header>
<a class="back" href="../index.html">&#8592; 返回汇总</a>
<h1><code>{page}</code>{flip_badge}<span class="qe-info">⚡ quant_evaluator</span></h1>
<div class="meta">本周新挖（增量）· 评估区间 {EVAL_START} ~ {EVAL_END} · 收益口径 Vwap 后复权 vwap-to-vwap（shift(-2)）</div>
</header>
<main>
<div class="grid-4">
<div class="metric"><b class="{ic_cls}">{ic:+.4f}</b><span>RankIC</span></div>
<div class="metric"><b class="{ir_cls}">{ir:+.3f}</b><span>RankIC IR</span></div>
<div class="metric"><b>{e.get("n_days", 0)}</b><span>评估交易日</span></div>
<div class="metric"><b>{"是" if is_flipped else "否"}</b><span>已翻正</span></div>
</div>
<div class="card">
<h2>📐 因子表达式（FactorEngine DSL）</h2>
<div class="formula-wrap">{dsl_text or "（无 DSL 文本，见 source 字段）"}</div>
</div>
<div class="card">
<h2>🧬 优化因子（预处理 + 择优）</h2>
<p style="font-size:0.82rem;color:#64748b;margin:0">本因子已进入 weekly_backtest_output/optimized_factors/ 与 optimized_meta.json，最优变体与 IR 见汇总表。</p>
</div>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------
# stage: json_writeback
# --------------------------------------------------------------------------
def stage_json_writeback(factor, eval_result, cluster_result):
    """重新 Read 池 json → append 新条目 → dump → 校验 can_use 计数。"""
    page = factor["page_name"]
    fname = factor.get("factor_name") or f"factor_{page}"
    fe_formula = factor.get("fe_formula", "")
    is_flipped = bool(factor.get("is_flipped", False))
    dsl_text = _dsl_text(factor)
    lqtp_formula = factor.get("local_formula", "") or dsl_text

    pool = load_pool()
    orig_total = len(pool)
    orig_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    if page in {r.get("page_name") for r in pool}:
        return {"skipped": True, "reason": "already in pool"}

    entry = {
        "page_name": page,
        "factor_name": fname,
        "status": "incremental_intake",
        "dsl": dsl_text,
        "lqtp_formula": lqtp_formula,
        "fe_formula": fe_formula,
        "code": "",
        "is_flipped": is_flipped,
        "can_use_factor_engine": True,
        "note": "本周新挖增量因子（incremental_intake）",
        "is_unlisted_miner": True,
    }
    pool.append(entry)
    POOL_JSON.write_text(json.dumps(pool, ensure_ascii=False, indent=1))

    new_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    ok = new_can_use == orig_can_use + 1
    return {"added": page, "total": len(pool), "can_use": new_can_use,
            "orig_can_use": orig_can_use, "check_ok": ok,
            "total_check": len(pool) == orig_total + 1}


# --------------------------------------------------------------------------
# 首页：header 计数更新 + 新挖区块追加 26 行
# --------------------------------------------------------------------------
def update_index(new_entries):
    """更新 index.html：
    1) +14 → +40、含本周新挖 14 → 40、470 → 496（总数 470+26）
    2) all-factors 表尾追加 26 行（新因子标「新」badge）
    3) 在 robustness 前插入「本周新挖」区块（列出 26 条）
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    total = 470 + len(new_entries)  # 496
    n_new = 14 + len(new_entries)   # 40

    # 1) header 统计
    html = html.replace("<b>470</b><span>因子总数</span>",
                        f"<b>{total}</b><span>因子总数</span>")
    html = html.replace("<b>+14</b><span>本周新挖</span>",
                        f"<b>+{n_new}</b><span>本周新挖</span>")
    html = html.replace("<b>470</b><span>因子总数（含本周新挖 14）</span>",
                        f"<b>{total}</b><span>因子总数（含本周新挖 {n_new}）</span>")
    html = html.replace("含本周新挖 14", f"含本周新挖 {n_new}")
    html = html.replace("<h2 id=\"all-factors\">全部 470 个因子</h2>",
                        f"<h2 id=\"all-factors\">全部 {total} 个因子</h2>")
    # robustness 区块里的 470
    html = html.replace("<b>470</b><span>因子总数</span>",
                        f"<b>{total}</b><span>因子总数</span>")
    html = re.sub(r"存活 82/470", f"存活 82/{total}", html)

    # 2) 新挖因子区块（插入 all-factors 表后，robustness 前）
    rows = []
    for i, en in enumerate(new_entries, start=456 + 1):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            f'<td class="neg">—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>'
        )
    # 插入到 </tbody></table> 之后第一个 </table> 之前? all-factors 表是第一个 table。
    # 直接在 all-factors 的 </table> 后追加新挖区块
    marker = '</tbody>\n  </table>'
    block = '\n' + '\n'.join(rows) + '\n    </tbody>\n  </table>\n'
    # 替换 all-factors 表的结尾：在其 </tbody></table> 前插入新行
    idx_tbl = html.find('id="all-factors"')
    idx_tbody_end = html.find('</tbody>', idx_tbl)
    html = html[:idx_tbody_end] + '\n' + '\n'.join(rows) + '\n' + html[idx_tbody_end:]

    # 3) 「本周新挖」区块（放 robustness 前）
    new_section = _new_mining_section(new_entries)
    anchor = '<section id="robustness-2026">'
    html = html.replace(anchor, new_section + '\n' + anchor, 1)

    INDEX_HTML.write_text(html, encoding="utf-8")
    banned = _check_banned(new_section, page="")
    return {"total": total, "n_new": n_new, "rows_added": len(rows),
            "banned_in_new_section": banned}


def _new_mining_section(new_entries):
    rows = ""
    for i, en in enumerate(new_entries, start=1):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        cluster_id = en.get("cluster_id", "—")
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows += (
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            f'<td>{cluster_id}</td></tr>'
        )
    return f'''
<section id="new-mining">
<h2>🆕 本周新挖增量因子（26）</h2>
<p style="font-size:0.82rem;color:#64748b">本次增量入库 26 条，评估口径与既有池一致（vwap-to-vwap shift(-2) 后复权）。</p>
<table>
  <thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>因子族</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</section>
'''


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[:args.limit]

    state = load_state()
    done = {}
    for factor in manifest:
        fname = factor["factor_name"]
        page = factor["page_name"]
        print(f"\n=== {page} ({fname}) ===", flush=True)
        if fname not in state:
            state[fname] = {"stage": "", "result": {}, "page": page}
        for stage in STAGES:
            if stage_done(state, fname, stage):
                print(f"  [skip] {stage}", flush=True)
                continue
            try:
                if stage == "dedup_check":
                    res = stage_dedup_check(factor)
                elif stage == "landing":
                    res = stage_landing(factor)
                elif stage == "eval":
                    res = stage_eval(factor)
                elif stage == "cluster_assign":
                    res = stage_cluster_assign(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "optimize_lite":
                    res = stage_optimize_lite(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "page_inject":
                    res = stage_page_inject(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "json_writeback":
                    res = stage_json_writeback(
                        factor,
                        state[fname]["result"].get("eval", {}),
                        state[fname]["result"].get("cluster_assign", {}))
                else:
                    res = {}
            except Exception:
                print(f"  [FAIL] {stage}\n{traceback.format_exc()}", flush=True)
                save_state(state)
                sys.exit(1)
            state[fname]["result"][stage] = res
            state[fname]["stage"] = stage
            save_state(state)
            msg = json.dumps(res, ensure_ascii=False)
            print(f"  [done] {stage}: {msg[:220]}", flush=True)
        done[page] = {
            "factor_name": fname,
            "eval": state[fname]["result"].get("eval", {}),
            "cluster": state[fname]["result"].get("cluster_assign", {}),
            "optimize": state[fname]["result"].get("optimize_lite", {}),
            "page": state[fname]["result"].get("page_inject", {}),
        }

    # 全部完成后：更新首页
    if not args.limit or len(manifest) >= 26:
        new_entries = []
        for fname, st in state.items():
            if st.get("stage") == "json_writeback" and st.get("page"):
                page = st["page"]
                ev = st["result"].get("eval", {})
                cl = st["result"].get("cluster_assign", {})
                new_entries.append({
                    "page_name": page,
                    "rank_ic": ev.get("rank_ic", 0.0),
                    "ic_ir": ev.get("ic_ir", 0.0),
                    "is_flipped": ev.get("is_flipped", False),
                    "cluster_id": cl.get("assigned", "—"),
                })
        if new_entries:
            upd = update_index(new_entries)
            print(f"\n[index] 首页已更新: {json.dumps(upd, ensure_ascii=False)[:200]}", flush=True)

    DONE_JSON.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(f"\nALL DONE -> {DONE_JSON}", flush=True)


if __name__ == "__main__":
    main()
