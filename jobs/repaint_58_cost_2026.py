#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #39: 58 新挖页五图重画 — 补成本扣减 + 2026 分段线 + 摘要表换手率/费率行。

口径（与 canonical 渲染器 rebuild_factor_detail_pages.py 完全一致）：
  - 收益：Vwap_adj.pct_change().shift(-2)（vwap-to-vwap）
  - 换手成本：to_rate = 1.0 - same.mean()（组内与昨日同组比例），group_ret[t,k] = gr_ret - to_rate*0.001
    top_turnover[0] = 0.0（第 0 天不扣）
  - 十分组：rank pct → floor*10 clip 0-9；无效(因子或收益 NaN) 记 -1 不参与
  - 翻正：is_flipped 因子矩阵 ×(-1)，图中标注「已翻正」
  - 窗口：r1 26 页 = 2016 窗截断(2016-01-04~2018-06-30)；r2 32 页 = 全窗
    （例外 vol_momentum_divergence_stable 在 r1 组但矩阵全窗 2588 天，页面声明的也是全窗 2016-01-04~2026-08-27，
      该页按全窗画 + 2026 分段线，指标按全窗算——页面自身自洽即可）
  - r2/全窗页：净值类图(十分层/多空) + RankIC 时序图加 2026-01-01 黄虚线分段
  - 全部 58 页：metric 卡「Top10% 换手率」值补真；摘要表在「LS 日胜率」行后插入
    Top10% 换手率 (日) + 双边费率假设 两行；卡区后补 cost-note div
  - 图题加「扣双边成本 10 bp」标注

写回协议：逐节重扫 h2 偏移；RankIC 时序节先删 svg/img/notice 再插新 img；
mod4 校验 (len-22)%4；PNG 解码验收；div 平衡验收。
"""
import os, json, re, io, base64, warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont('/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf')
plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

PROJ = '/home/sunhaiwei/quant_projects'
FAC = PROJ + '/factor_engine/docs/reports/2026-08-23/factors'
MATDIRS = ['factor_matrices_all_2026r2', 'factor_matrices_all_2026daily', 'factor_matrices_all']
COST_BPS = float(os.environ.get("TRADING_COST_BPS", "10"))
TOTAL_COST = COST_BPS / 1e4          # 双边费率 10 bp
W0, W1 = '2016-01-04', '2018-06-30'  # r1 2016 窗截断
Y26 = pd.Timestamp('2026-01-01')     # 分段线

# ---- 清单 ----
names = [l.strip() for l in open('/tmp/newfact_names.txt') if l.strip()]
PAGE = {}
FLIP = {}
for jf in ['/tmp/new50_selected.json', '/tmp/new50_round2_final.json']:
    for x in json.load(open(jf)):
        PAGE[x['factor_name']] = x['page_name']
        FLIP[x['page_name']] = bool(x.get('is_flipped'))
pages = [PAGE.get(n, n.replace('factor_', '', 1)) for n in names]
win = json.load(open('/tmp/window_groups.json'))
R1 = set(win['2016窗'])     # 26
R2 = set(win['全窗'])       # 32
# 例外页：vol_momentum_divergence_stable 在 r1 组但页面声明全窗（矩阵 2588 天），按全窗画
FULL_EXCEPT = {'vol_momentum_divergence_stable'}

vw = pd.read_parquet(PROJ + '/lightgbm_qs/data/build/ohlcv_adj_wide/Vwap_adj.parquet')
FWD = vw.pct_change().shift(-2)


def mpath(pn):
    for d in MATDIRS:
        q = f'{PROJ}/weekly_backtest_output/{d}/{pn}.parquet'
        if os.path.exists(q):
            return q
    raise FileNotFoundError(pn)


def load_mat(pn, window):
    """载入矩阵并按窗口截断；返回 (values (T,N), dates(DatetimeIndex), fwd((T,N)))。
    window: '2016' 截断 2016-01-04~2018-06-30；'full' 全窗。"""
    m = pd.read_parquet(mpath(pn))
    if window == '2016':
        m = m[(m.index >= W0) & (m.index <= W1)]
    dates = m.index
    f2 = m.reindex(index=FWD.index, columns=FWD.columns).dropna(how='all')
    ret = FWD.reindex(index=f2.index, columns=f2.columns)
    dates2 = f2.index
    return f2.values.astype(np.float64), dates2, ret.values.astype(np.float64)


def compute(fv, fwd, dates, sign):
    """十分层 + 换手成本扣减 + 指标。返回 (ic_series, group_ret(T,10), top_turnover(T), ls, dates)。"""
    a = fv * sign
    b = fwd
    T, N = a.shape
    valid = np.isfinite(a) & np.isfinite(b)
    # 每日 IC
    rk = pd.DataFrame(a).rank(axis=1, pct=True).values
    ics = np.full(T, np.nan)
    for t in range(T):
        x = rk[t]; y = b[t]
        nz = np.isfinite(x) & np.isfinite(y)
        if nz.sum() < 15:
            continue
        xs, ys = x[nz], y[nz]
        ra = pd.Series(xs).rank().values; rb = pd.Series(ys).rank().values
        am = ra - ra.mean(); bm = rb - rb.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        if d > 1e-18:
            ics[t] = (am * bm).sum() / d
    # 十分组 + 换手成本（canonical）
    mat_ranks = pd.DataFrame(a).rank(axis=1, method='first', pct=True).values
    gids = np.floor(mat_ranks * 10).clip(0, 9).astype(int)
    gids[~valid] = -1
    group_ret = np.zeros((T, 10))
    top_turnover = np.zeros(T)
    prev_gids = np.full(N, -1)
    for t in range(T):
        for k in range(10):
            mk = (gids[t] == k)
            if not mk.any():
                continue
            if t > 0:
                same = (prev_gids[mk] == k)
                to_rate = 1.0 - float(same.mean())
            else:
                to_rate = 0.0
            gr_ret = float(np.nanmean(b[t, mk]))
            if k == 9:
                top_turnover[t] = to_rate
            group_ret[t, k] = gr_ret - to_rate * TOTAL_COST
        prev_gids = gids[t].copy()
    top_turnover[0] = 0.0
    ic = pd.Series(ics, index=dates).dropna()
    return ic, group_ret, top_turnover


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def add_2026(ax, common):
    if common.min() < Y26 <= common.max():
        ax.axvline(Y26, color='#f59e0b', lw=1.0, ls='--', alpha=0.7)
        try:
            ax.text(Y26, ax.get_ylim()[1] * 0.95, ' 2026→', fontsize=7, color='#b45309')
        except Exception:
            pass


COST_SUF = f'（扣双边成本 {COST_BPS:.0f} bp）'


def plot_ts(ic, window):
    ma = ic.rolling(20).mean()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(ic.index, ic, lw=0.4, color='#cbd5e1')
    ax.plot(ma.index, ma, lw=1.4, color='#7c3aed')
    ax.axhline(0, color='#94a3b8', lw=0.6)
    wlab = '2016 窗 2016-01~2018-06' if window == '2016' else '全窗 2016-01~2026-08'
    ax.set_title(f'RankIC 时序（细线=日IC，粗线=20日均线，{wlab}，翻正口径）', fontsize=9)
    ax.grid(alpha=0.25)
    if window == 'full':
        add_2026(ax, ic.index)
    return fig_b64(fig)


def plot_heatmap(ic, window):
    if len(ic) < 60:
        return None
    monthly = ic.resample('ME').mean().dropna()
    if len(monthly) < 6:
        return None
    years = sorted(set(monthly.index.year))
    grid = np.full((len(years), 12), np.nan)
    for dt, v in monthly.items():
        grid[years.index(dt.year), dt.month - 1] = v
    vmax = max(0.03, float(np.nanpercentile(np.abs(grid[np.isfinite(grid)]), 95)))
    fig, ax = plt.subplots(figsize=(10, max(2.2, 0.42 * len(years) + 0.8)))
    im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12)); ax.set_xticklabels([f'{m}月' for m in range(1, 13)], fontsize=8)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years, fontsize=8)
    for yi in range(len(years)):
        for mi in range(12):
            v = grid[yi, mi]
            if np.isfinite(v):
                ax.text(mi, yi, f'{v:.3f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(v) > vmax * 0.55 else '#0f172a')
    wlab = '2016 窗' if window == '2016' else '全窗'
    ax.set_title(f'月度 RankIC 热力图（{wlab}，翻正口径）', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return fig_b64(fig)


def plot_decile(group_ret, dates, window):
    valid = np.isfinite(group_ret).any(axis=1)
    if valid.sum() < 30:
        return None
    d = dates[valid]; g = group_ret[valid]
    nav = np.cumprod(1 + g, axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, 10))
    for k in range(10):
        ax.plot(d, nav[:, k], lw=1.0, color=colors[k], label=f'G{k+1}')
    ax.legend(fontsize=6, ncol=10, loc='upper left')
    wlab = '2016 窗' if window == '2016' else '全窗'
    ax.set_title(f'十分层净值（{wlab}，G10=多头，净额已扣双边成本 {COST_BPS:.0f} bp）', fontsize=9)
    ax.grid(alpha=0.25)
    if window == 'full':
        add_2026(ax, d)
    return fig_b64(fig)


def plot_ls(group_ret, dates, window, flip):
    ok = np.isfinite(group_ret[:, [0, 9]]).all(axis=1)
    if ok.sum() < 30:
        return None
    d = dates[ok]
    ls = np.cumprod(1 + group_ret[ok, 9] - group_ret[ok, 0])
    s = pd.Series(ls); dd = (s / s.cummax() - 1).min()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(d, ls, lw=1.4, color='#7c3aed')
    ax.axhline(1, color='#94a3b8', lw=0.7, ls='--')
    ann = f'终值 {ls[-1]:.2f} · 年化 {ls[-1]**(252/max(len(ls),1))-1:+.1%} · 最大回撤 {dd:.1%}'
    wlab = '2016 窗' if window == '2016' else '全窗'
    fl = '，已翻正' if flip else ''
    ax.set_title(f'多空净值 G10−G1（{wlab}，净额已扣双边成本 {COST_BPS:.0f} bp{fl}）   {ann}', fontsize=9)
    ax.grid(alpha=0.25)
    if window == 'full':
        add_2026(ax, d)
    return fig_b64(fig)


def plot_dist(ic, window):
    if len(ic) < 10:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(ic.values, bins=40, color='#1e4d8c', alpha=0.75, edgecolor='white')
    m = float(ic.mean())
    ax.axvline(m, color='red', lw=2, label=f'均值 {m:+.4f}')
    ax.axvline(0, color='gray', lw=1, ls=':')
    ax.legend(fontsize=7)
    wlab = '2016 窗' if window == '2016' else '全窗'
    ax.set_title(f'RankIC 分布（{wlab}，红线=均值）', fontsize=9)
    return fig_b64(fig)


IMG_RE = re.compile(r'data:image/png;base64,[A-Za-z0-9+/=]+')
NOTICE_RE = re.compile(r'<div class="zero-notice">[^<]*</div>')
SECTIONS = [('RankIC 时序', 'ts'), ('月度 RankIC 热力图', 'hm'),
            ('十分层净值曲线', 'dec'), ('多空净值曲线', 'ls'), ('RankIC 分布', 'dist')]


def replace_section(t, sec, newimg):
    """逐节重扫 h2 偏移，替换该节第一个 img/svg/notice 为新图。"""
    h2pos = [(m.start(), m.group(1)) for m in re.finditer(r'<h2>(.*?)</h2>', t)]
    st = [q for q, txt in h2pos if txt.startswith(sec)]
    if not st:
        return None
    s = st[0]
    nxt = min([q for q, _ in h2pos if q > s], default=len(t))
    span = t[s:nxt]
    newimg_tag = f'<img src="data:image/png;base64,{newimg}" style="width:100%;border-radius:8px;"/>'
    if sec == 'RankIC 时序':
        span2 = re.sub(r'<svg.*?</svg>', '', span, flags=re.S)
        span2 = IMG_RE.sub('', span2)
        span2 = NOTICE_RE.sub('', span2)
        span2 = re.sub(r'(</h2>\s*)', r'\1' + newimg_tag.replace('\\', '\\\\'), span2, count=1)
    else:
        span2 = IMG_RE.sub('__NEWIMG__', span)
        span2 = NOTICE_RE.sub('__NEWIMG__', span2)
        span2 = span2.replace('__NEWIMG__', newimg_tag)
    # 逐节重扫
    h2pos2 = [(m.start(), m.group(1)) for m in re.finditer(r'<h2>(.*?)</h2>', t)]
    st2 = [q for q, txt in h2pos2 if txt.startswith(sec)]
    if not st2:
        return None
    s2 = st2[0]
    nxt2 = min([q for q, _ in h2pos2 if q > s2], default=len(t))
    return t[:s2] + span2 + t[nxt2:]


def build_cost_note():
    return ('<div class="cost-note">净值已扣双边交易成本 10.0 bp（换手率 × 费率）；'
            '多空/十分层均含停牌股剔除（收益为 NaN 不参与）。</div>')


def main():
    ok = 0
    fails = []
    for pn in pages:
        try:
            window = 'full' if (pn in R2 or pn in FULL_EXCEPT) else '2016'
            flip = FLIP.get(pn, False)
            fv, dates, fwd = load_mat(pn, window)
            ic, gr, top_turn = compute(fv, fwd, dates, -1.0 if flip else 1.0)
            charts = {'ts': plot_ts(ic, window), 'hm': plot_heatmap(ic, window),
                      'dec': plot_decile(gr, dates, window), 'ls': plot_ls(gr, dates, window, flip),
                      'dist': plot_dist(ic, window)}
            p = f'{FAC}/factor_{pn}.html'
            t = open(p, encoding='utf-8').read()
            # 1) 五图替换
            for sec, key in SECTIONS:
                if not charts[key]:
                    continue
                nt = replace_section(t, sec, charts[key])
                if nt is None:
                    raise RuntimeError(f'{pn}: section {sec} not found')
                t = nt
            # 2) metric 卡 Top10% 换手率 值
            mean_turn = float(np.nanmean(top_turn[1:])) if len(top_turn) > 1 else 0.0
            pat = re.compile(r'<div class="metric"><b[^>]*>[^<]*</b><span>Top10% 换手率[^<]*</span></div>')
            m2 = pat.search(t)
            if not m2:
                raise RuntimeError(f'{pn}: no turnover metric card')
            t = t[:m2.start()] + f'<div class="metric"><b class="pos">{mean_turn*100:.2f}%</b><span>Top10% 换手率 (日)</span></div>' + t[m2.end():]
            # 3) 摘要表插入两行（在 LS 日胜率 行后）
            anchor = r'<tr><td>LS 日胜率</td><td[^>]*>[^<]*</td></tr>'
            am = re.search(anchor, t)
            if not am:
                raise RuntimeError(f'{pn}: summary LS 日胜率 row not found')
            two_rows = (f'\n<tr><td>Top10% 换手率 (日)</td><td>{mean_turn*100:.2f}%</td></tr>\n'
                        f'<tr><td>双边费率假设</td><td>{COST_BPS:.1f} bp（按换手率×费率扣减）</td></tr>')
            t = t[:am.end()] + two_rows + t[am.end():]
            # 4) cost-note div（卡区 grid-3 后）
            cn = build_cost_note()
            if 'class="cost-note"' not in t:
                gm = re.search(r'<div class="grid-3">\s*<div class="metric">.*?</div>\s*</div>', t, re.S)
                if not gm:
                    raise RuntimeError(f'{pn}: no grid-3 metric block')
                t = t[:gm.end()] + '\n' + cn + t[gm.end():]
            # 5) 校验
            bad = [m for m in re.finditer(r'data:image/png;base64,([A-Za-z0-9+/=]+)"', t) if len(m.group(1)) % 4]
            if bad:
                raise RuntimeError(f'{pn}: mod4 {len(bad)}')
            for m in re.finditer(r'data:image/png;base64,([A-Za-z0-9+/=]+)"', t):
                img = Image.open(io.BytesIO(base64.b64decode(m.group(1))))
                img.verify()
            open(p, 'w', encoding='utf-8').write(t)
            ok += 1
            print(f'OK {pn}: win={window} flip={flip} IC={ic.mean():.4f} IR={ic.mean()/ic.std():.3f} '
                  f'turn={mean_turn*100:.2f}% n_days={len(ic)}')
        except Exception as e:
            import traceback; traceback.print_exc()
            fails.append((pn, str(e)[:120]))
    print(f'\nrepainted {ok}/{len(pages)}, fails={fails}')


if __name__ == '__main__':
    main()
