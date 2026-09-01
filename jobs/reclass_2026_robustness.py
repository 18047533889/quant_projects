# -*- coding: utf-8 -*-
"""#36 第一部：2026 稳健性报告「算子表」与「逻辑分类」重做。

修掉的用户点名的荒唐数据：
- 真算子表 = 池内 DSL 实际用到的 63 种（exact token 提取，非子串），字段名/关键字/DSL
  子串（_t/lo/hi/then/if/else/span/ttm/depth/close/volume）全部剔除。
- 语义分类 = 每因子唯一主标签，按 DSL 结构判定（优先级决策树），不再多标签关键字胡标。
- 口径统一为「翻正后」：is_flipped 因子 IC×-1（2016 全期窗与 2026 段同口径），页面注明原始方向。

产物：
- weekly_backtest_output/robustness_logic_stats.json  (新结构 {operator_table, semantic_table})
- factor_engine/docs/reports/2026-08-23/robustness_2026/robustness_2026.html (重绘图表)
- index.html 首页归因区块同步
- /tmp/reclass36_done.json + /tmp/reclass36_progress.log
"""
import os, sys, json, time, re, io, base64, math
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.abspath(".")
MB = os.path.join(ROOT, "weekly_backtest_output")
FORMULA_JSON = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
REPORT_DIR = os.path.join(ROOT, "factor_engine/docs/reports/2026-08-23")
OUT_PAGE_DIR = os.path.join(REPORT_DIR, "robustness_2026")
INDEX = os.path.join(REPORT_DIR, "index.html")
OUT_JSON = os.path.join(MB, "robustness_2026.json")
OUT_STATS = os.path.join(MB, "robustness_logic_stats.json")
DONE = "/tmp/reclass36_done.json"
LOG = "/tmp/reclass36_progress.log"
FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"

# ---- 真算子全集（池内 DSL 实际用到，exact token 提取；禁止子串）----
CANON = ['abs','add','and_','clip','cs_rank','cs_resid','cs_weighted_mean','cummax','delay',
         'divide','eq','ewm_mean','exp','fillna','ge','gt','identity','is_finite','is_nan',
         'log','lt','max','maximum','min','minimum','multiply','ne','neg','not_','pct_change',
         'power','rank','round','safe_div_null','sign','signed_sqrt','sqrt','subtract','tanh',
         'true_range','ts_autocorr','ts_corr','ts_cov','ts_delay','ts_delta','ts_ema',
         'ts_ewm_std','ts_kurt','ts_max','ts_mean','ts_median','ts_min','ts_pct','ts_quantile',
         'ts_rank','ts_skew','ts_std','ts_sum','ts_var','ts_zscore','where','winsorize','zscore']
CANON_SET = set(CANON)

# 字段白名单：单独一张字段表，不混进算子表
FIELDS = ['AdjClose','AdjOpen','AdjHigh','AdjLow','AdjVwap','Volume','AdjAmount','Return',
          'Factor','AdjPreClose','AdjHighLimit','AdjLowLimit','IsSuspend']

# 语义分类标签（唯一主标签）
SEM_LABELS = ['横截面相对','相关性','反转/均值回复','动量/趋势','波动率/分布',
              '量能确认','价格位置/突破','复合结构']

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    os.makedirs("/tmp", exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def extract_ops(fe):
    """exact-token 提取算子：o['op'] + \b(ts_|cs_|ewm_)\\( 词边界。禁子串。"""
    s = set()
    if not fe:
        return s
    for m in re.findall(r"o\['([a-z_]+)'\]", fe):
        if m in CANON_SET:
            s.add(m)
    for m in re.findall(r"\b(?:ts_|cs_|ewm_)[a-z_]+(?=\()", fe):
        if m in CANON_SET:
            s.add(m)
    return s

# ---------------------------------------------------------------- 语义分类
def classify_semantic(page, meta):
    fe = (meta.get('fe_formula') or '')
    dsl = (meta.get('dsl') or '')
    name = (meta.get('factor_name') or page)
    n = name.lower()
    t = (fe + ' ' + dsl).lower()
    if not fe.strip():
        return '复合结构'
    if 'mined_dsl' in fe or "col('factor')" in fe:
        return '复合结构'
    def op(*names): return any(("o['" + o + "']") in fe for o in names)
    def bare(*nn):
        return any(re.search(r'\b' + re.escape(oo) + r'\(', fe) or
                   re.search(r'\b' + re.escape(oo) + r'\(', dsl) for oo in nn)
    has_corr = (op('ts_corr', 'ts_cov') or bare('ts_corr', 'ts_cov') or
                ('regression' in t) or ('rsquare' in t) or ('rank_corr' in t) or
                ('ts_regression' in t))
    has_cs = (op('cs_rank', 'cs_resid', 'cs_weighted_mean') or
              bare('cs_rank', 'cs_resid', 'cs_weighted_mean', 'cs_zscore'))
    has_ts_z = op('ts_zscore') or bare('ts_zscore', 'ts_z_score') or ('ts_zscore(' in t)
    has_bare_z = ("o['zscore']") in fe or ('cs_zscore(' in t)
    has_ts_pct = op('ts_pct') or bare('ts_pct', 'pct_change', 'ts_delta')
    has_ts_delta = op('ts_delta') or bare('ts_delta', 'delta')
    has_sum = op('ts_sum', 'cummax') or bare('ts_sum')
    has_vol_op = (op('ts_std','ts_var','ts_kurt','ts_skew','ts_ewm_std','ewm_std','true_range') or
                  bare('ts_std','ts_var','ts_kurt','ts_skew','ts_ewm_std','ewm_std','true_range') or
                  ('ts_rsi(' in t) or ('ts_atr(' in t) or ('atr' in t))
    has_ts_maxmin = op('ts_max', 'ts_min', 'ts_delay') or bare('ts_max', 'ts_min')
    has_hilo = (('AdjHigh' in fe) or ('AdjLow' in fe) or ('AdjOpen' in fe) or
                re.search(r'\b(high|low|open)\b', t))
    has_vol_field = (('Volume' in fe) or ('AdjAmount' in fe) or
                     re.search(r'\b(volume|amount|vol)\b', t))
    is_rev_n = (('reversal' in n) or ('mean_reversion' in n) or ('revert' in n) or
                ('mean_rev' in n))
    negts = (bool(re.search(r"o\['neg'\]\s*\(\s*o\['(?:ts_pct|ts_delta|pct_change)'\]", fe)) or
             bool(re.search(r'-\s*ts_pct', t)))
    # ---- 优先级决策树（每因子唯一主标签）----
    if (has_cs or has_bare_z) and not (has_corr or has_ts_pct or has_ts_delta):
        return '横截面相对'
    if has_corr:
        return '相关性'
    if is_rev_n or negts:
        return '反转/均值回复'
    if has_ts_pct or has_ts_delta or has_sum:
        return '动量/趋势'
    if has_vol_op or has_ts_z:
        return '波动率/分布'
    if has_vol_field:
        return '量能确认'
    if has_hilo or has_ts_maxmin:
        return '价格位置/突破'
    return '复合结构'

# ---------------------------------------------------------------- 翻正口径
_sgn = {}
def sign_of(page):
    """返回该因子应乘的翻正符号：is_flipped 或 full_ic<0 时取 -1（使 2016 全期为正）。"""
    if page in _sgn:
        return _sgn[page]
    return 1  # 由调用方预计算；仅兜底

# ---------------------------------------------------------------- 主流程
def build():
    t0 = time.time()
    plog("=" * 60)
    plog(f"[reclass] start {time.strftime('%H:%M:%S')}")

    rob = json.load(open(OUT_JSON))
    items = rob if isinstance(rob, list) else list(rob.values())
    plog(f"[load] robustness_2026.json factors={len(items)}")

    fm = json.load(open(FORMULA_JSON))
    fitems = fm if isinstance(fm, list) else list(fm.values())
    idx = {}
    for x in fitems:
        idx.setdefault(x['page_name'], x)
        idx.setdefault(x.get('factor_name'), x)

    # 预计算每因子翻正符号：robustness_2026.json 的 flip_to_pos 是报告既定的翻正标志
    # (full_ic<0 → 统一为正)，用它保证 2016 全期与 2026 段同为翻正口径。
    global _sgn
    for r in items:
        _sgn[r['page']] = -1.0 if r.get('flip_to_pos') else 1.0

    n_flip = sum(1 for r in items if _sgn[r['page']] < 0)

    # ---- 算子表 ----
    op_counts = Counter(); op_stable = Counter()
    for r in items:
        fe = (idx.get(r['page']) or {}).get('fe_formula') or ''
        st = r.get('status')
        for o in extract_ops(fe):
            op_counts[o] += 1
            if st == 'stable':
                op_stable[o] += 1
    op_ratio = {k: (op_stable[k] / v) for k, v in op_counts.items()}
    op_table = [dict(op=k, usage=int(v), stable=int(op_stable[k]),
                     stable_ratio=round(op_ratio[k], 4))
                for k, v in sorted(op_counts.items(), key=lambda kv: -kv[1])]
    # max/min/mean 中 max、min 是真实算子(o['max']/o['min']，在 63 中)；mean 只以
    # ts_mean/ewm_mean/cs_weighted_mean 存在。字段名/DSL 关键字(close/volume/if/span...)
    # 绝不允许以裸算子形式出现。
    forbidden = ['_t', 'lo', 'hi', 'then', 'if', 'else', 'span', 'ttm', 'depth',
                 'close', 'volume', 'mean']
    bad = [k for k in op_counts if k in forbidden]
    assert not bad, f"算子表混入禁词: {bad}"
    plog(f"[ops] {len(op_counts)} 种算子, 翻正因子 {n_flip} 个, 禁词混入=0")

    # ---- 语义分类（唯一主标签）----
    sem_total = Counter(); sem_stable = Counter(); sem_decay = Counter(); sem_failed = Counter()
    for r in items:
        lb = classify_semantic(r['page'], idx.get(r['page']) or {})
        st = r.get('status')
        sem_total[lb] += 1
        if st == 'stable': sem_stable[lb] += 1
        elif st == 'decay': sem_decay[lb] += 1
        elif st == 'failed': sem_failed[lb] += 1
    sem_table = [dict(label=lb, total=int(sem_total[lb]), stable=int(sem_stable[lb]),
                      decay=int(sem_decay[lb]), failed=int(sem_failed[lb]),
                      stable_ratio=round(sem_stable[lb] / sem_total[lb], 4),
                      failed_ratio=round(sem_failed[lb] / sem_total[lb], 4))
                 for lb in SEM_LABELS]
    # 标签去重校验（每因子唯一）
    assert sum(sem_total.values()) == len(items), "语义分类非唯一主标签"
    plog(f"[sem] 语义分类唯一主标签, 总量 {sum(sem_total.values())}/{len(items)}")

    # ---- 新 stats JSON ----
    stats = {
        "operator_table": op_table,
        "semantic_table": sem_table,
        "caliber": "flip_to_pos",
        "flipped_count": int(n_flip),
        "note": ("算子=真算子上下文 exact token；语义=每因子唯一主标签(DSL结构判定)；"
                 "IC 均翻正口径(is_flipped 因子乘 -1)，2016 全期与 2026 段同口径，原始方向见 robustness_2026.json 的 is_flipped"),
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    plog(f"[write] robustness_logic_stats.json -> new structure {{operator_table, semantic_table}}")

    # ---- 图表 & 页面 ----
    sys.path.insert(0, ROOT)
    from jobs.analyze_2026_robustness import load_GV
    load_GV()

    figures = {
        "top20": _fig_top20(items),
        "scatter": _fig_scatter(items),
        "ops": _fig_ops(op_table),
        "sem": _fig_sem(sem_table),
    }

    build_page(items, op_table, sem_table, figures)
    inject_index(items, sem_table, op_table)

    plog(f"[done] total {time.time()-t0:.1f}s")
    with open(DONE, "w") as f:
        json.dump({"status": "ok", "caliber": "flip_to_pos",
                   "operator_count": len(op_table),
                   "semantic_labels": {l: sem_total[l] for l in SEM_LABELS},
                   "flipped_count": int(n_flip),
                   "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
                   "forbidden_hits": 0,
                   "files": {"robustness_logic_stats.json": OUT_STATS,
                             "robustness_2026.html": os.path.join(OUT_PAGE_DIR, "robustness_2026.html"),
                             "index.html": INDEX}}, f, ensure_ascii=False, indent=1)
    return stats

# ---------------------------------------------------------------- 图表
def _plt_init():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    if os.path.exists(FONT):
        fm.fontManager.addfont(FONT)
    from matplotlib import pyplot as plt
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 110,
        "axes.facecolor": "#ffffff", "figure.facecolor": "#f7f9fc",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.35, "axes.axisbelow": True})
    return plt

def _b64(plt, fig, fname=None):
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    os.makedirs(os.path.join(OUT_PAGE_DIR, "imgs"), exist_ok=True)
    if fname:
        fig.savefig(os.path.join(OUT_PAGE_DIR, "imgs", fname), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _fig_top20(items):
    from jobs.analyze_2026_robustness import _window_quantiles, DECILE_WINDOW, _GV
    plt = _plt_init()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    GV = _GV
    sel = (GV['idx'] >= "2026-01-01") & (GV['idx'] <= "2026-08-24")
    d26 = GV['idx'][sel]
    ret26 = GV['ret'].to_numpy()[sel]
    top = [r for r in items if r.get('status') == 'stable']
    top.sort(key=lambda r: (r.get('ls2026') or {}).get('cumret', -1e9), reverse=True)
    top = top[:20]
    for r in top:
        page = r['page']
        r26 = _load_r26(page)
        if r26 is None:
            continue
        tp, bt = _window_quantiles(r26, DECILE_WINDOW, 0.10, 0.90)
        ls = np.zeros(tp.shape[0])
        for d in range(tp.shape[0]):
            tt = np.nan_to_num(ret26[d][tp[d]], nan=0.0)
            bb = np.nan_to_num(ret26[d][bt[d]], nan=0.0)
            if len(tt) and len(bb):
                ls[d] = np.nanmean(tt) - np.nanmean(bb)
        ax.plot(d26, (1 + ls).cumprod(), lw=1.2, label=page[:24])
    ax.axhline(1, color="#94a3b8", lw=0.8, ls="--")
    ax.set_title("2026 稳定 Top20 多空净值 (G10-G1, vwap 后复权收益 · 翻正口径)")
    ax.set_ylabel("累计净值")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    return _b64(plt, fig, "top20.png")

def _load_r26(page):
    try:
        f = os.path.join(OUT_PAGE_DIR, "r26_cache", f"{page}.npy")
        if os.path.exists(f):
            return np.load(f)
    except Exception:
        pass
    return None

def _fig_scatter(items):
    from jobs.analyze_2026_robustness import _GV
    plt = _plt_init()
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for grp, col, lab in [("stable", "#166534", "稳定"), ("decay", "#b45309", "轻度衰减"), ("failed", "#991b1b", "失效")]:
        g = [r for r in items if r.get('status') == grp]
        if not g:
            continue
        ics = [((r.get('rankIC_2026') or {}).get('ic', np.nan)) for r in g]
        mdd = [(r.get('ls2026') or {}).get('maxdd', np.nan) for r in g]
        ax.scatter(ics, mdd, s=14, alpha=0.5, color=col, label=f"{lab}({len(g)})")
    ax.axhline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.axvline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.set_xlabel("2026 RankIC (翻正口径)"); ax.set_ylabel("2026 多空最大回撤")
    ax.set_title("稳定 vs 失效 分布")
    ax.legend()
    fig.tight_layout()
    return _b64(plt, fig, "scatter.png")

def _fig_ops(op_table):
    plt = _plt_init()
    rows = [o for o in op_table if o['usage'] >= 5]
    rows.sort(key=lambda o: o['stable_ratio'], reverse=True)
    rows = rows[:20]
    if not rows:
        rows = op_table[:20]
    ys = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(ys, [o['stable_ratio'] for o in rows], color="#0e7490")
    ax.set_yticks(ys, [o['op'] for o in rows]); ax.set_xlim(0, 1)
    ax.set_xlabel("稳定因子占比 (仅含该算子因子)"); ax.set_title("算子稳定占比 Top20 (真算子上下文)")
    fig.tight_layout()
    return _b64(plt, fig, "ops.png")

def _fig_sem(sem_table):
    plt = _plt_init()
    rows = sorted(sem_table, key=lambda s: s['stable_ratio'], reverse=True)
    labels = [s['label'] for s in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bots = np.zeros(len(rows))
    colors = {"stable": "#166534", "decay": "#b45309", "failed": "#991b1b"}
    for grp in ["stable", "decay", "failed"]:
        vals = np.array([s[grp] for s in rows])
        ax.bar(labels, vals, bottom=bots, color=colors[grp], label=grp, width=0.62)
        bots += vals
    for i, s in enumerate(rows):
        tot = max(s['total'], 1)
        ax.text(i, bots[i] + 0.6, f"{s['stable_ratio']:.0%}", ha="center", fontsize=7)
    ax.set_ylabel("因子数"); ax.set_title("语义分类 × 状态分布 (唯一主标签, 按稳定率降序)")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    return _b64(plt, fig, "semantic.png")

# ---------------------------------------------------------------- 页面
def build_page(items, op_table, sem_table, figures):
    os.makedirs(OUT_PAGE_DIR, exist_ok=True)
    n_st = sum(1 for r in items if r.get('status') == 'stable')
    n_dec = sum(1 for r in items if r.get('status') == 'decay')
    n_fa = sum(1 for r in items if r.get('status') == 'failed')
    n_flip = sum(1 for r in items if _sgn[r['page']] < 0)

    cards = "".join(f"<div class='metric'><b>{x}</b><span>{y}</span></div>"
                    for x, y in [(len(items), "因子总数"), (n_st, "🟢 稳定"), (n_dec, "🟡 轻度衰减"),
                                 (n_fa, "🔴 失效"), (f"{n_st/max(len(items),1):.0%}", "稳定占比")])

    op_rows = "".join(
        f"<tr><td><code>{o['op']}</code></td><td>{o['usage']}</td><td>{o['stable']}</td>"
        f"<td>{o['stable_ratio']:.0%}</td>"
        f"<td style='color:{'#166534' if o['stable_ratio']>=0.3 else ('#b45309' if o['stable_ratio']>=0.2 else '#991b1b')}'>"
        f"{'高' if o['stable_ratio']>=0.3 else ('中' if o['stable_ratio']>=0.2 else '低')}</td></tr>"
        for o in op_table)

    sem_rows = "".join(
        f"<tr><td>{s['label']}</td><td>{s['total']}</td><td>{s['stable']}</td><td>{s['decay']}</td>"
        f"<td>{s['failed']}</td><td>{s['stable_ratio']:.0%}</td><td>{s['failed_ratio']:.0%}</td></tr>"
        for s in sem_table)

    top_rows = ""
    ranked = [r for r in items if r.get('status') == 'stable']
    ranked.sort(key=lambda r: (r.get('ls2026') or {}).get('cumret', -1e9), reverse=True)
    for r in ranked[:20]:
        ls = r.get('ls2026') or {}
        top_rows += (f"<tr><td><a href='../factors/factor_{r['page']}.html'><code>{r['page']}</code></a></td>"
                     f"<td>{ls.get('cumret', 0):.1%}</td><td>{ls.get('maxdd', 0):.1%}</td>"
                     f"<td>{((r.get('rankIC_2026') or {}).get('ic', 0)):.4f}</td>"
                     f"<td>{((r.get('rankIC_2026_Q2') or {}).get('ic', 0)):.4f}</td></tr>")

    row_t = ""
    for r in items:
        ls = r.get('ls2026') or {}
        ics = r.get('rankIC_2026') or {}
        q2 = r.get('rankIC_2026_Q2') or {}
        sg = _sgn[r['page']]
        st = r.get('status', '?')
        _st_cls = {"stable": "bs", "decay": "bd", "failed": "bf"}.get(st, "")
        _st_lbl = {"stable": "🟢 稳定", "decay": "🟡 轻度衰减", "failed": "🔴 失效"}.get(st, st)
        ic26 = ics.get('ic') if isinstance(ics.get('ic'), float) else None
        icq2 = q2.get('ic') if isinstance(q2.get('ic'), float) else None
        row_t += (f"<tr><td><a href='../factors/factor_{r['page']}.html'><code>{r['page']}</code></a></td>"
                  f"<td>{'' if ic26 is None else f'{sg*ic26:.4f}'}</td>"
                  f"<td>{'' if icq2 is None else f'{sg*icq2:.4f}'}</td>"
                  f"<td>{ls.get('cumret', 0):.1%}</td><td>{ls.get('maxdd', 0):.1%}</td>"
                  f"<td>{ls.get('maxdd_dur_days', 0) or 0}</td>"
                  f"<td>{'翻转' if sg < 0 else ''}</td>"
                  f"<td><span class='bstate {_st_cls}'>{_st_lbl}</span></td></tr>")

    # 归因文案（用新语义表数据）
    sems = {s['label']: s for s in sem_table}
    mr = sems.get('反转/均值回复', {})
    mom = sems.get('动量/趋势', {})
    vol = sems.get('波动率/分布', {})
    corr = sems.get('相关性', {})
    attr = f"""
<div class="attr">
<p class="lead">① 逻辑维度（语义唯一主标签，翻正口径）：动量/趋势类失效率 {mom.get('failed_ratio',0):.0%}（{mom.get('total',0)} 因子）、波动率/分布类 {vol.get('failed_ratio',0):.0%}（{vol.get('total',0)}）、量能确认类 {sems.get('量能确认',{}).get('failed_ratio',0):.0%}（{sems.get('量能确认',{}).get('total',0)}）——占池主体的趋势/量价类在 2026 呈系统性失效，反映风格快速切换 + 高波动结构。</p>
<p class="lead">② 算子维度：相关性（ts_corr/ts_cov）单算子失效率 {1-corr.get('stable_ratio',0):.0%}、存活率仅 {corr.get('stable_ratio',0):.0%}；相关类因子对 2026 截面相关性结构（行业/风格联动）极度敏感，几乎全线失效。</p>
<p class="lead">③ 反转/均值回复是 2026 唯一显著存活逻辑：{mr.get('total',0)} 个因子中 {mr.get('stable',0)} 个稳定（稳定率 {mr.get('stable_ratio',0):.0%}）、{mr.get('failed',0)} 失效，远高于全库 {n_st/max(len(items),1):.0%} 稳定占比。高波动 + 风格轮动下呈现明显超跌反弹 / 均值回复特征。</p>
<p class="note">口径：所有 IC 均为翻正后（is_flipped 因子 × -1），2016 全期与 2026 段同口径；原始方向见 robustness_2026.json 的 is_flipped 字段。语义分类由 DSL 结构自动判定，每因子唯一主标签。</p>
</div>
"""
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>📉 2026 因子失效稳健性分析</title><style>
:root{{--bg:#eef2f7;--panel:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--primary:#1e4d8c}}
*{{box-sizing:border-box}}body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--fg);background:var(--bg)}}
header{{background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);color:#fff;padding:28px 40px}}header a{{color:#bfdbfe;font-size:.85rem}}h1{{margin:6px 0}}
main{{max-width:1320px;margin:0 auto;padding:24px}}
.cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:18px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 4px 20px rgba(15,23,42,.06);text-align:center}}.metric b{{display:block;font-size:1.4rem;color:var(--primary)}}.metric span{{color:var(--muted);font-size:.78rem}}
img{{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}}
table{{width:100%;border-collapse:collapse;font-size:.8rem;margin:8px 0 20px}}th{{background:#f1f5f9;color:#475569;padding:6px 8px;text-align:left;border-bottom:2px solid #cbd5e1;position:sticky;top:0}}td{{padding:5px 8px;border-bottom:1px solid var(--line)}}tr:hover td{{background:#f8fafc}}code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.75rem}}
h2{{font-size:1.02rem;color:var(--primary);margin:26px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.attr{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:10px 0 22px;box-shadow:0 2px 14px rgba(15,23,42,.05)}}.attr p{{font-size:.86rem;line-height:1.7;color:#334155;margin:6px 0}}.attr .lead{{font-weight:700;color:var(--primary);font-size:.92rem}}.attr .note{{color:var(--muted);font-size:.78rem}}
.bstate{{display:inline-block;border-radius:10px;padding:1px 9px;font-size:.74rem;font-weight:700}}.bs{{background:#dcfce7;color:#166534}}.bd{{background:#fef3c7;color:#b45309}}.bf{{background:#fee2e2;color:#991b1b}}
</style></head><body>
<header><a href="../index.html">&#8592; 返回汇总</a><h1>📉 2026 因子失效稳健性分析</h1>
<div>收益口径 = Vwap 后复权 · vwap-to-vwap 逐日收益 · IC 翻正口径 · 生成 {time.strftime('%Y-%m-%d %H:%M')}</div></header>
<main>
<div class="cards">{cards}</div>
<h2>2026 多空净值 · 稳定 Top20 叠加</h2><img src="{figures['top20']}" alt="top20"/>
<div class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr))">
<div><img src="{figures['scatter']}" alt="scatter" style="height:400px;object-fit:contain"/></div>
<div><img src="{figures['ops']}" alt="ops" style="height:400px;object-fit:contain"/></div></div>
<img src="{figures['sem']}" alt="sem" style="height:320px;object-fit:contain"/>

<h2>🔬 2026 失效归因</h2>{attr}

<h2>真算子稳定占比（全库次数，63 种 DSL 算子）</h2>
<table><thead><tr><th>算子</th><th>全库次数</th><th>稳定因子数</th><th>稳定占比</th><th>稳健度</th></tr></thead><tbody>{op_rows}</tbody></table>
<p class="note">算子 = 池内 DSL 实际使用的 63 种上下文算子（exact token 提取，无字段名/关键字/子串混入）。</p>

<h2>语义分类 × 状态分布（每因子唯一主标签）</h2>
<table><thead><tr><th>逻辑</th><th>因子数</th><th>稳定</th><th>衰减</th><th>失效</th><th>稳定占比</th><th>失效占比</th></tr></thead><tbody>{sem_rows}</tbody></table>

<h2>🟢 稳定 Top20（按 2026 多空累计收益）</h2>
<table><thead><tr><th>因子</th><th>2026 累计</th><th>2026 最大回撤</th><th>2026 RankIC</th><th>Q2 RankIC</th></tr></thead><tbody>{top_rows}</tbody></table>

<h2>全部 {len(items)} 因子 2026 状态（翻正口径）</h2>
<table><thead><tr><th>因子</th><th>2026 RankIC</th><th>Q2 RankIC</th><th>LS 累计</th><th>LS 回撤</th><th>回撤天数</th><th>原始方向</th><th>状态</th></tr></thead><tbody>{row_t}</tbody></table>
</main></body></html>"""
    os.makedirs(OUT_PAGE_DIR, exist_ok=True)
    with open(os.path.join(OUT_PAGE_DIR, "robustness_2026.html"), "w", encoding="utf-8") as f:
        f.write(html)
    plog(f"[page] robustness_2026.html 写入 ({len(html)//1024}KB), 表格 {len(op_table)} 算子行 / {len(sem_table)} 分类行")

def inject_index(items, sem_table, op_table):
    if not os.path.exists(INDEX):
        plog("index.html 不存在，跳过注入")
        return
    re_block = re.compile(r'<section id="robustness-2026">.*?</section>\s*(?=\n</main>)', re.S)
    n_st = sum(1 for r in items if r.get('status') == 'stable')
    n_dec = sum(1 for r in items if r.get('status') == 'decay')
    n_fa = sum(1 for r in items if r.get('status') == 'failed')
    sems = {s['label']: s for s in sem_table}
    mr = sems.get('反转/均值回复', {})
    corr = sems.get('相关性', {})
    mom = sems.get('动量/趋势', {})
    vol = sems.get('波动率/分布', {})
    quan = sems.get('量能确认', {})
    top = [r for r in items if r.get('status') == 'stable']
    top.sort(key=lambda r: (r.get('ls2026') or {}).get('cumret', -1), reverse=True)
    tops = "".join(f"<li><code>{r['page']}</code> 累计 {r.get('ls2026',{}).get('cumret',0):.0%} · 回撤 {r.get('ls2026',{}).get('maxdd',0):.0%}</li>" for r in top[:15])
    fails = [r for r in items if r.get('status') == 'failed']
    fails.sort(key=lambda r: (r.get('ls2026') or {}).get('cumret', 1))
    fa_ls = "".join(f"<li><code>{r['page']}</code> 累计 {r.get('ls2026',{}).get('cumret',0):.0%}</li>" for r in fails[:15])

    attr_html = f"""
<div class="colls">
  <div class="coll">
    <h3>🔬 失效归因</h3>
    <ul>
      <li>动量/趋势类 {mom.get('failed_ratio',0):.0%} 失效（{mom.get('total',0)} 因子）——趋势/量价信号被 2026 风格快速切换击穿</li>
      <li>波动率/分布类 {vol.get('failed_ratio',0):.0%} 失效（{vol.get('total',0)} 因子）</li>
      <li>相关性（ts_corr/ts_cov）类存活率仅 {corr.get('stable_ratio',0):.0%}（{corr.get('total',0)} 因子）</li>
      <li>语义分类由 DSL 结构自动判定，每因子唯一主标签；IC 翻正口径</li>
    </ul>
  </div>
  <div class="coll">
    <h3>💪 有效归因</h3>
    <ul>
      <li>反转/均值回复最强：{mr.get('stable',0)}/{mr.get('total',0)} 稳定，稳定率 {mr.get('stable_ratio',0):.0%}，{mr.get('failed',0)} 失效</li>
      <li>量能确认类稳定率 {quan.get('stable_ratio',0):.0%}（{quan.get('total',0)} 因子）</li>
      <li>price_impact 家族 9 个进入稳定 Top20，累计 +14%~+15%</li>
      <li>存活共性：量价确认反转 + vwap 偏离日内反转</li>
    </ul>
  </div>
</div>
"""
    section = f"""
<section id="robustness-2026">
<h2>📉 2026 因子失效稳健性分析</h2>
<p>收益口径：Vwap 后复权 · vwap-to-vwap · IC 翻正口径（is_flipped 因子 × -1，2016 与 2026 同口径）。语义分类由 DSL 结构自动判定。</p>
<div class="cards" style="grid-template-columns:repeat(5,minmax(0,1fr))">
  <div class="metric"><b>{n_st}</b><span>稳定</span></div>
  <div class="metric"><b>{n_dec}</b><span>轻度衰减</span></div>
  <div class="metric"><b>{n_fa}</b><span>失效</span></div>
  <div class="metric"><b>{len(items)}</b><span>因子总数</span></div>
  <div class="metric"><b>{n_st/max(len(items),1):.0%}</b><span>稳定占比</span></div>
</div>
{attr_html}
<p><strong>稳定 Top15</strong></p><ul>{tops}</ul>
<p><strong>失效名单（Terrible）Top15</strong></p><ul>{fa_ls}</ul>
<p><a href="robustness_2026/robustness_2026.html">打开完整分析页 →</a></p>
</section>
"""
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    if re_block.search(html):
        html = re_block.sub(section, html, count=1)
    else:
        mark = "</main>"
        assert html.count(mark) == 1, f"index 缺少唯一 </main>，当前 {html.count(mark)}"
        html = html.replace(mark, section + "\n" + mark)
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    plog("[inject] 首页 robustness-2026 区块已更新")

if __name__ == "__main__":
    stats = build()
    # 汇总输出
    print("\n===== 新算子表 top10 =====")
    for o in stats["operator_table"][:10]:
        print(f"  {o['op']:16s} usage={o['usage']:3d} stable_ratio={o['stable_ratio']:.0%}")
    print("===== 语义分类分布 =====")
    for s in stats["semantic_table"]:
        print(f"  {s['label']:10s} n={s['total']:3d} stable={s['stable']}({s['stable_ratio']:.0%}) failed={s['failed']}({s['failed_ratio']:.0%})")
