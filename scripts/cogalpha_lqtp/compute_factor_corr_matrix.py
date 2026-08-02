#!/usr/bin/env python3
"""Cross-sectional mean Spearman correlation among screening factors."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"


def _factor_sql(path: Path, alias: str) -> str:
    p = path.as_posix().replace("'", "''")
    return f"""
    SELECT
      CAST(COALESCE(td, dt) AS INTEGER) AS trade_date,
      CAST(sym AS VARCHAR) AS symbol,
      CAST(val AS DOUBLE) AS value,
      '{alias}' AS factor_name
    FROM (
      SELECT
        TRY_CAST(trade_date AS INTEGER) AS td,
        TRY_CAST(strftime(TRY_CAST(datetime AS TIMESTAMP), '%Y%m%d') AS INTEGER) AS dt,
        COALESCE(symbol, asset) AS sym,
        value AS val
      FROM read_parquet('{p}')
      WHERE value IS NOT NULL AND isfinite(CAST(value AS DOUBLE))
    )
    """


def compute_corr(
    work: Path,
    names: list[str],
    *,
    sample_days: int = 120,
    min_names: int = 50,
) -> dict[str, Any]:
    lake = work / "factor_lake"
    paths = [(n, lake / n / "values.parquet") for n in names]
    paths = [(n, p) for n, p in paths if p.is_file() and p.stat().st_size > 0]
    if len(paths) < 2:
        return {"factors": [], "matrix": [], "mean_abs_offdiag": None, "n_pairs": 0}

    con = duckdb.connect()
    try:
        union_parts = [_factor_sql(path, name) for name, path in paths]
        stacked = " UNION ALL ".join(union_parts)
        sql = f"""
        WITH raw AS (
          {stacked}
        ),
        dates AS (
          SELECT trade_date
          FROM (
            SELECT DISTINCT trade_date AS trade_date FROM raw
          ) d
          ORDER BY trade_date DESC
          LIMIT {int(sample_days)}
        ),
        filt AS (
          SELECT r.factor_name, r.trade_date, r.symbol, r.value
          FROM raw r
          INNER JOIN dates d ON r.trade_date = d.trade_date
        ),
        day_n AS (
          SELECT factor_name, trade_date, count(*) AS n
          FROM filt
          GROUP BY factor_name, trade_date
        ),
        ranked AS (
          SELECT f.factor_name, f.trade_date, f.symbol,
                 RANK() OVER (PARTITION BY f.factor_name, f.trade_date ORDER BY f.value) AS rk
          FROM filt f
          INNER JOIN day_n dn
            ON f.factor_name = dn.factor_name AND f.trade_date = dn.trade_date
          WHERE dn.n >= {int(min_names)}
        ),
        daily_pair AS (
          SELECT a.trade_date,
                 a.factor_name AS fa,
                 b.factor_name AS fb,
                 corr(a.rk, b.rk) AS corr_day
          FROM ranked a
          INNER JOIN ranked b
            ON a.trade_date = b.trade_date
           AND a.symbol = b.symbol
           AND a.factor_name < b.factor_name
          GROUP BY a.trade_date, a.factor_name, b.factor_name
          HAVING count(*) >= {int(min_names)}
        )
        SELECT fa, fb, avg(corr_day) AS mean_rank_corr, count(*) AS n_days
        FROM daily_pair
        GROUP BY fa, fb
        """
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    factors = sorted({n for n, _ in paths})
    idx = {n: i for i, n in enumerate(factors)}
    mat = np.eye(len(factors), dtype=float)
    pairs = []
    for a, b, corr, n_days in rows:
        if a not in idx or b not in idx or corr is None:
            continue
        c = float(corr)
        mat[idx[a], idx[b]] = c
        mat[idx[b], idx[a]] = c
        pairs.append({"a": a, "b": b, "mean_rank_corr": c, "n_days": int(n_days)})

    off = mat[np.triu_indices(len(factors), k=1)]
    mean_abs = float(np.nanmean(np.abs(off))) if len(off) else None
    high = sorted(
        [p for p in pairs if abs(p["mean_rank_corr"]) >= 0.7],
        key=lambda x: -abs(x["mean_rank_corr"]),
    )[:80]
    return {
        "generated_at": datetime.now().isoformat(),
        "n_factors": len(factors),
        "sample_days": sample_days,
        "mean_abs_offdiag": mean_abs,
        "factors": factors,
        "matrix": mat.tolist(),
        "high_corr_pairs": high,
        "n_pairs": len(pairs),
    }


def render_corr_html(payload: dict[str, Any], out_path: Path) -> None:
    factors = payload.get("factors") or []
    matrix = payload.get("matrix") or []
    high = payload.get("high_corr_pairs") or []
    rows = []
    for p in high[:40]:
        rows.append(
            f"<tr><td>{p['a']}</td><td>{p['b']}</td>"
            f"<td>{p['mean_rank_corr']:.3f}</td><td>{p['n_days']}</td></tr>"
        )
    n = len(factors)
    peer = []
    for i, name in enumerate(factors):
        vals = [abs(matrix[i][j]) for j in range(n) if i != j]
        peer.append((name, float(np.mean(vals)) if vals else 0.0))
    peer.sort(key=lambda x: -x[1])
    top = [x[0] for x in peer[:25]]
    top_idx = [factors.index(x) for x in top]
    cells = []
    for i in top_idx:
        row = []
        for j in top_idx:
            v = matrix[i][j]
            a = max(0.0, min(1.0, (v + 1) / 2))
            color = f"rgba({int(255*(1-a))},{int(80+100*a)},{int(255*a)},0.85)"
            row.append(
                f'<td title="{factors[i]} × {factors[j]} = {v:.3f}" style="background:{color}"></td>'
            )
        cells.append("<tr>" + "".join(row) + "</tr>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>因子截面秩相关矩阵</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
table.corr td{{width:14px;height:14px;border:0}}
table.list{{border-collapse:collapse;width:100%;background:#fff}}
table.list th,table.list td{{border:1px solid #e2e8f0;padding:8px;font-size:13px}}
.muted{{color:#64748b}}
</style></head><body>
<h1>因子截面秩相关（Spearman / 日均）</h1>
<p class="muted">生成 {payload.get('generated_at')} · 因子数 {payload.get('n_factors')} ·
采样交易日 {payload.get('sample_days')} · 非对角 |corr| 均值 {payload.get('mean_abs_offdiag')}</p>
<p>口径：每日截面因子秩相关，再对交易日取均值（因子值相关，非收益相关）。
IC/ICIR 收益侧统一为 VWAP→VWAP T+1。</p>
<h2>高相关对 (|ρ|≥0.7)</h2>
<table class="list"><thead><tr><th>A</th><th>B</th><th>mean rank corr</th><th>n days</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan=4>无</td></tr>'}</tbody></table>
<h2>Top25 平均相关强度热力</h2>
<table class="corr">{''.join(cells)}</table>
<p><a href="/reports/factor_rankic_screening_index.html">返回因子索引</a></p>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--sample-days", type=int, default=80)
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args()
    work = args.work_dir
    cat_path = args.catalog or (work / "screening_reeval_catalog.json")
    names = [e["function_name"] for e in json.loads(cat_path.read_text(encoding="utf-8"))]
    payload = compute_corr(work, names, sample_days=args.sample_days)
    out_json = work / "reports" / "factor_rank_corr_matrix.json"
    out_html = work / "reports" / "factor_rank_corr_matrix.html"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_corr_html(payload, out_html)
    print(
        f"corr matrix n={payload.get('n_factors')} pairs={payload.get('n_pairs')} "
        f"mean_abs={payload.get('mean_abs_offdiag')} -> {out_html}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
