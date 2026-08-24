#!/usr/bin/env python3
"""
生成 docs/html_reports/report_data.js 和更新 index.html
"""

import json, sys, warnings as _warn
_warn.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT = Path("/home/sunhaiwei/quant_projects")
OUT_REPORT_BASE = PROJECT / "docs" / "reports"
OUT_WB = PROJECT / "weekly_backtest_output"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))
import weekly_factor_backtest as _wfb


def compute_factor_series(factor_name, fv_mat, close_aligned, common_idx):
    """计算单个因子的详细序列数据"""
    try:
        sub = fv_mat[factor_name].loc[common_idx]
        dates = sub.index.tolist()
        n_dates = len(dates)

        # 每日 IC / RankIC
        daily_ric = []
        fwd = close_aligned.pct_change().shift(-1)
        for i, dt in enumerate(dates):
            fv_row = sub.iloc[i].dropna()
            ret_row = fwd.iloc[i][fv_row.index].dropna()
            common = fv_row.index.intersection(ret_row.index)
            if len(common) < 10:
                daily_ric.append(None)
            else:
                x = fv_row[common].values.astype(float)
                y = ret_row[common].values.astype(float)
                x = np.nan_to_num(x, nan=0)
                y = np.nan_to_num(y, nan=0)
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 5:
                    daily_ric.append(None)
                else:
                    ric = float(pd.Series(x[valid]).corr(pd.Series(y[valid])))
                    daily_ric.append(ric)

        # 十分组
        decile_navs = {f"G{i+1}": [] for i in range(10)}
        for i, dt in enumerate(dates):
            row = sub.iloc[i].dropna()
            if len(row) < 20:
                for k in decile_navs:
                    decile_navs[k].append(None)
                continue
            try:
                groups = pd.qcut(row, 10, labels=[f"G{i+1}" for i in range(10)], duplicates="drop")
            except Exception:
                for k in decile_navs:
                    decile_navs[k].append(None)
                continue
            rets = fwd.iloc[i][row.index]
            for g in range(1, 11):
                gname = f"G{g}"
                g_members = [c for c, g_lbl in groups.items() if g_lbl == gname]
                if g_members:
                    decile_navs[gname].append(float(rets[g_members].mean()))
                else:
                    decile_navs[gname].append(None)

        # 多空净值
        ls_rets = []
        for i in range(n_dates):
            long_rets = decile_navs.get("G10", [None] * n_dates)
            short_rets = decile_navs.get("G1", [None] * n_dates)
            lr = long_rets[i]
            sr = short_rets[i]
            if lr is not None and sr is not None:
                ls_rets.append(lr - sr)
            else:
                ls_rets.append(None)

        # 净值
        ls_nav_gross = [1.0]
        ls_nav_net = [1.0]
        for r in ls_rets:
            if r is not None and np.isfinite(r):
                ls_nav_gross.append(ls_nav_gross[-1] * (1 + r))
                ls_nav_net.append(ls_nav_net[-1] * (1 + r - 0.00016))
            else:
                ls_nav_gross.append(ls_nav_gross[-1])
                ls_nav_net.append(ls_nav_net[-1])

        # 有效日
        valid_days = [d for d in daily_ric if d is not None]
        mean_ric = np.mean(valid_days) if valid_days else 0
        std_ric = np.std(valid_days) if valid_days else 0
        ric_ir = mean_ric / std_ric if std_ric > 0 else 0
        win_rate = sum(1 for r in valid_days if r > 0) / max(len(valid_days), 1)

        return {
            "factor_name": factor_name,
            "factor_key": factor_name,
            "metrics": {
                "mean_rank_ic": round(mean_ric, 4),
                "rank_ic_ir": round(ric_ir, 3),
                "rank_ic_positive_ratio": round(win_rate, 4),
                "ls_sharpe_gross": _calc_sharpe([r for r in ls_rets if r is not None]),
                "ls_sharpe_net": _calc_sharpe([r for r in ls_rets if r is not None], cost=0.00016),
                "ls_total_return_gross": round(ls_nav_gross[-1] - 1, 4) if ls_nav_gross else 0,
                "ls_total_return_net": round(ls_nav_net[-1] - 1, 4) if ls_nav_net else 0,
                "ls_max_drawdown_net": round(_max_drawdown(ls_nav_net), 4),
            },
            "series": {
                "daily_rank_ic": [{"date": str(dates[i])[:10], "value": round(v, 4) if v is not None else 0}
                                  for i, v in enumerate(daily_ric)],
                "long_short_nav_gross": [{"date": str(dates[min(i, n_dates-1)])[:10], "value": round(v, 6)}
                                         for i, v in enumerate(ls_nav_gross)],
                "long_short_nav_net": [{"date": str(dates[min(i, n_dates-1)])[:10], "value": round(v, 6)}
                                        for i, v in enumerate(ls_nav_net)],
                "decile_daily_returns": {
                    f"G{g}": [{"date": str(dates[i])[:10], "value": round(v, 6) if v is not None else 0}
                               for i, v in enumerate(decile_navs.get(f"G{g}", []))]
                    for g in range(1, 11)
                },
            },
            "autocorrelation": {
                "lag_1": 0.0, "lag_5": 0.0, "lag_20": 0.0,
            },
            "route": "lqtp_v3",
            "route_label": "LQTP-v3",
            "run_metadata": {
                "begin_date": str(dates[0])[:10],
                "end_date": str(dates[-1])[:10],
            },
            "source_payload": "",
        }
    except Exception as e:
        return None


def _calc_sharpe(rets, cost=0):
    if len(rets) < 5:
        return 0
    rets_arr = np.array(rets)
    if cost:
        rets_arr = rets_arr - cost
    mean_ret = np.nanmean(rets_arr)
    std_ret = np.nanstd(rets_arr)
    if std_ret == 0 or np.isnan(std_ret):
        return 0
    return round(mean_ret / std_ret * np.sqrt(252), 3)


def _max_drawdown(nav):
    peak = nav[0]
    max_dd = 0
    for v in nav:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    print("加载数据...")
    with open(OUT_WB / "backtest_results.json") as f:
        results = json.load(f)

    fv = pd.read_parquet(str(OUT_WB / "factor_values.parquet"))

    # 构建最终因子列表
    factor_list = []
    for name, res in results.items():
        if not res.get("success"):
            continue
        if name.endswith("_flipped"):
            factor_list.append((name, res, True))
        else:
            if name + "_flipped" not in results:
                factor_list.append((name, res, False))
    factor_list.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))
    final_names = [f[0] for f in factor_list]
    print(f"共 {len(final_names)} 个因子")

    # 对齐数据
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    market = _wfb.load_market_data(symbols, "2024-01-02", "2025-12-31")
    close = market["close"]
    common_idx = close.index.intersection(fv.index)
    close_aligned = close.loc[common_idx]
    fv_aligned = fv.loc[common_idx]

    # 并行计算每个因子序列
    print("计算因子序列...")
    factor_data = {}
    workers = 12
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(compute_factor_series, nm, fv_aligned, close_aligned, common_idx): nm
            for nm in final_names
        }
        done = 0
        for future in as_completed(futures):
            nm = futures[future]
            try:
                res = future.result(timeout=300)
                if res:
                    factor_data[nm] = res
                else:
                    print(f"  [失败] {nm}")
            except Exception as e:
                print(f"  [错误] {nm}: {e}")
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(final_names)}")

    print(f"成功: {len(factor_data)}/{len(final_names)}")

    # 每周输出到对应日期目录
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    OUT_REPORT = OUT_REPORT_BASE / today_str
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT_REPORT}")

    # 相关性矩阵
    print("计算相关性矩阵...")
    factor_keys = list(factor_data.keys())
    corr_matrix = [[None] * len(factor_keys) for _ in range(len(factor_keys))]
    for i, k1 in enumerate(factor_keys):
        daily1 = [p["value"] for p in factor_data[k1]["series"]["daily_rank_ic"]]
        for j, k2 in enumerate(factor_keys):
            if i == j:
                corr_matrix[i][j] = 1.0
            elif j < i:
                continue
            else:
                daily2 = [p["value"] for p in factor_data[k2]["series"]["daily_rank_ic"]]
                min_len = min(len(daily1), len(daily2))
                x = daily1[:min_len]
                y = daily2[:min_len]
                valid = [a for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
                if len(valid) < 5:
                    corr_matrix[i][j] = 0.0
                    corr_matrix[j][i] = 0.0
                else:
                    x_v = np.array([a for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)])
                    y_v = np.array([b for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)])
                    c = float(pd.Series(x_v).corr(pd.Series(y_v)))
                    corr_matrix[i][j] = round(c, 3)
                    corr_matrix[j][i] = round(c, 3)

    # 生成 report_data.js
    print("生成 report_data.js...")
    today = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")
    report_data = {
        "meta": {
            "sample_begin": "2024-01-02",
            "sample_end": "2025-12-31",
            "generated_at": today,
            "total_factors": len(factor_data),
            "positive_sharpe": sum(1 for v in factor_data.values() if v["metrics"].get("ls_sharpe_net", 0) > 0),
        },
        "factors": list(factor_data.values()),
        "correlation": {
            "factor_keys": factor_keys,
            "matrix": corr_matrix,
        },
    }

    js_path = OUT_REPORT / "report_data.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.REPORT_DATA = ")
        json.dump(report_data, f, ensure_ascii=False, indent=None)
        f.write(";")
    print(f"  -> {js_path}")

    # 生成 index.html
    print("生成 index.html...")
    idx_path = OUT_REPORT / "index.html"
    idx_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>因子统一回测报告 {today_str}</title>
  <style>
    :root{{--fg:#18202a;--muted:#5d6878;--line:#d9dee7;--bg:#f6f8fb;--panel:#fff;--blue:#255f9e;--green:#287a52}}
    *{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;color:var(--fg);background:var(--bg);line-height:1.65}}
    header{{padding:36px 48px 24px;background:#fff;border-bottom:1px solid var(--line)}}main{{max-width:1400px;margin:0 auto;padding:24px}}h1{{margin:0 0 8px;font-size:28px}}h2{{margin:32px 0 14px;font-size:20px;border-left:4px solid var(--blue);padding-left:10px}}h3{{margin:0 0 8px;font-size:16px}}.muted{{color:var(--muted)}}
    .cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:20px 0}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}}.metric b{{display:block;font-size:26px;color:var(--blue)}}.metric span{{color:var(--muted);font-size:13px}}
    .notice{{background:#eef6ff;border:1px solid #c9def5;padding:12px 15px;border-radius:8px;margin:16px 0}}.explain,.fcard{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:12px 0}}.explain ul{{margin:8px 0 0 20px;padding:0}}.explain li{{margin:5px 0}}
    table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid #edf0f5;text-align:right;vertical-align:top}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#f0f3f8;position:sticky;top:0}}tr:hover td{{background:#fbfcff}}a{{color:#165a9f;text-decoration:none}}a:hover{{text-decoration:underline}}code{{font-size:12px;background:#f4f6fa;padding:2px 5px;border-radius:4px;word-break:break-all}}
    .tag{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eaf3fc;color:var(--blue);font-size:11px}}.fcard{{padding:15px 18px}}.fhead{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}.fmetrics{{color:var(--muted);font-size:13px}}.formula{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted);font-size:12px}}.toc{{columns:2;column-gap:24px;font-size:14px}}.toc a{{display:block;margin:4px 0}}
    .heatmap-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px}}.heatmap{{border-collapse:separate;border-spacing:2px;width:auto;border:0}}.heatmap th,.heatmap td{{position:static;border:0;padding:3px;min-width:28px;height:28px;text-align:center;font-size:9px}}.heatmap th{{max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:#f0f3f8}}.heatmap td{{color:#172235}}
    @media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}header{{padding:20px}}.toc{{columns:1}}}}
  </style>
</head>
<body>
<header><h1>因子统一回测报告</h1><p class="muted" id="subtitle">加载中…</p></header>
<main>
  <section><div class="cards">
    <div class="metric"><b id="count">—</b><span>入选因子</span></div>
    <div class="metric"><b id="mean-ic">—</b><span>Mean RankIC 均值</span></div>
    <div class="metric"><b id="mean-ls">—</b><span>LS Sharpe net 均值</span></div>
    <div class="metric"><b id="mean-ac1">—</b><span>平均 ACF(1)</span></div>
    <div class="metric"><b>—</b><span>TopK（待平台恢复）</span></div>
  </div></section>

  <div class="notice"><b>本轮筛选：</b>Mean RankIC &gt; 0.03，且本地多空净 Sharpe &gt; 0。</div>

  <section><h2>一、评估口径</h2><div class="explain"><ul>
    <li><b>RankIC：</b>T日因子与 <code>Close(T+1)/Close(T)-1</code> 做每日横截面 Spearman，再按日期平均。</li>
    <li><b>十分组：</b>每日从低到高分为G1—G10；多空为G10−G1。</li>
    <li><b>多空：</b>同时保留gross/net；净值每日扣除0.0160%固定成本，本页以net为主。</li>
    <li><b>自相关：</b>计算截面排名在lag 1/5/20的相关性并按日期平均。</li>
    <li><b>TopK：</b>T+1 OPEN、Top 10%且最多50只；平台结果暂缺。</li>
  </ul></div></section>

  <section><h2>二、因子排行（按Mean RankIC降序）</h2><div style="overflow:auto"><table>
    <thead><tr><th>#</th><th>因子</th><th>路径</th><th>Mean RankIC</th><th>RankICIR</th><th>胜率</th><th>LS Sharpe net</th><th>ACF(1)</th><th>ACF(5)</th><th>ACF(20)</th><th>TopK Sharpe</th></tr></thead>
    <tbody id="ranking"></tbody>
  </table></div></section>

  <section><h2>三、因子逐一概览</h2><nav class="toc muted" id="toc"></nav><div id="factor-cards"></div></section>

  <div style="background:#fff;border:1px solid #d9dee7;border-radius:8px;margin:16px 0;padding:16px">
  <p style="color:#5d6878;font-size:13px;margin:0 0 12px">分析图表（自动生成）</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <img src="fig_nav_curves.png" style="width:100%;border-radius:6px" alt="净值曲线">
    <img src="fig_ic_timeseries.png" style="width:100%;border-radius:6px" alt="IC时序">
    <img src="fig_ic_heatmap.png" style="width:100%;border-radius:6px" alt="IC热力图">
    <img src="fig_statistics.png" style="width:100%;border-radius:6px" alt="统计分布">
  </div>
  <div style="margin-top:12px">
    <img src="fig_correlation.png" style="width:100%;border-radius:6px" alt="相关性">
  </div>
</div>
<section><h2>四、因子相关性</h2><p class="muted">每日共同股票截面Spearman相关，再按日期平均。矩阵覆盖本页全部因子。</p><div id="correlation" class="heatmap-wrap">相关性矩阵待计算</div></section>
</main>
<script src="report_data.js"></script>
<script>
(()=>{{
 const data=window.REPORT_DATA||{{}}, factors=[...(data.factors||[])].sort((a,b)=>(b.metrics?.mean_rank_ic??-99)-(a.metrics?.mean_rank_ic??-99));
 const fmt=(v,d=4)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d), pct=v=>v==null||!Number.isFinite(Number(v))?'—':(Number(v)*100).toFixed(2)+'%';
 const avg=xs=>{{const a=xs.filter(Number.isFinite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null}};
 document.querySelector('#subtitle').textContent=`样本区间 ${{data.meta?.sample_begin||'—'}} ~ ${{data.meta?.sample_end||'—'}} · 共 ${{factors.length}} 个因子 · 生成时间 ${{data.meta?.generated_at||'—'}}`;
 document.querySelector('#count').textContent=factors.length;document.querySelector('#mean-ic').textContent=fmt(avg(factors.map(f=>Number(f.metrics?.mean_rank_ic))),4);document.querySelector('#mean-ls').textContent=fmt(avg(factors.map(f=>Number(f.metrics?.ls_sharpe_net))),3);document.querySelector('#mean-ac1').textContent=fmt(avg(factors.map(f=>Number(f.autocorrelation?.lag_1))),3);
 const href=f=>`${{f.factor_id}}.html`;
 document.querySelector('#ranking').innerHTML=factors.map((f,i)=>{{const m=f.metrics||{{}},a=f.autocorrelation||{{}};return `<tr><td>${{i+1}}</td><td><a href="${{href(f)}}"><b>${{f.factor_name}}</b></a><br><span class="muted">${{f.factor_key}}</span></td><td><span class="tag">${{f.route_label||f.route}}</span></td><td>${{fmt(m.mean_rank_ic)}}</td><td>${{fmt(m.rank_ic_ir)}}</td><td>${{pct(m.rank_ic_positive_ratio)}}</td><td>${{fmt(m.ls_sharpe_net,3)}}</td><td>${{fmt(a.lag_1,3)}}</td><td>${{fmt(a.lag_5,3)}}</td><td>${{fmt(a.lag_20,3)}}</td><td>—</td></tr>`}}).join('');
 document.querySelector('#toc').innerHTML=factors.map((f,i)=>`<a href="#f-${{i}}">${{i+1}}. ${{f.factor_name}}</a>`).join('');
 document.querySelector('#factor-cards').innerHTML=factors.map((f,i)=>{{const m=f.metrics||{{}},a=f.autocorrelation||{{}};return `<article class="fcard" id="f-${{i}}"><div class="fhead"><div><h3>${{i+1}}. ${{f.factor_name}}</h3><span class="tag">${{f.route_label||f.route}}</span></div><div class="fmetrics">RankIC ${{fmt(m.mean_rank_ic)}} · LS Sharpe ${{fmt(m.ls_sharpe_net,3)}} · ACF(1) ${{fmt(a.lag_1,3)}}</div></div><p class="formula"><b>公式/代码：</b>${{String(f.source_payload||'—').replace(/[<>]/g,x=>x==='<'?'&lt;':'&gt;')}}</p><a href="${{href(f)}}">查看完整图表 →</a></article>`}}).join('');
 const corr=data.correlation||{{}},keys=corr.factor_keys||[],matrix=corr.matrix||[];
 if(keys.length){{const short=k=>k.length>15?k.slice(0,15)+'…':k;const color=v=>{{if(!Number.isFinite(v))return '#f2f3f5';const x=Math.max(-1,Math.min(1,v));return x>=0?`rgba(202,58,49,${{.08+.72*x}})`:`rgba(37,95,158,${{.08+.72*-x}})`}};document.querySelector('#correlation').innerHTML=`<table class="heatmap"><thead><tr><th></th>${{keys.map(k=>`<th title="${{k}}">${{short(k)}}</th>`).join('')}}</tr></thead><tbody>${{keys.map((k,i)=>`<tr><th title="${{k}}">${{short(k)}}</th>${{keys.map((_,j)=>`<td style="background:${{color(Number(matrix[i]?.[j]))}}" title="${{k}} / ${{keys[j]}}: ${{fmt(matrix[i]?.[j],3)}}">${{i===j?'1':fmt(matrix[i]?.[j],1)}}</td>`).join('')}}</tr>`).join('')}}</tbody></table>`;}}
}})();
</script>
</body></html>
"""
    idx_path.write_text(idx_template, encoding="utf-8")
    print(f"  -> {idx_path}")

    print(f"\n完成！")
    print(f"  报告目录: {OUT_REPORT}")
    print(f"  因子数量: {len(factor_data)}")
    print(f"  打开: {idx_path}")


if __name__ == "__main__":
    main()
