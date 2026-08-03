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
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.data_access_panel import (  # noqa: E402
    factor_values_cte_sql,
    resolve_factor_values_parquet,
)

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"


def _factor_sql(path: Path, alias: str) -> str:
    """Normalize lake parquet (trade_date/symbol or datetime/asset) via shared CTE helper."""
    inner = factor_values_cte_sql(path).strip().rstrip(";")
    safe_alias = alias.replace("'", "''")
    return f"""
    SELECT trade_date, symbol, value, '{safe_alias}' AS factor_name
    FROM ({inner})
    """


def _names_for_corr(work: Path, catalog_path: Path | None, *, min_ic: float) -> list[str]:
    """Prefer homepage VWAP panel names with RankIC > min_ic; else catalog."""
    prog_path = work / "screening_reeval_progress.json"
    names: list[str] = []
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        for row in prog.get("index_rows", []):
            mode = str(row.get("eval_mode") or "")
            if "vwap" not in mode:
                continue
            try:
                ic = float(row.get("mean_rank_ic", row.get("mean_ic")))
            except (TypeError, ValueError):
                continue
            if ic != ic or abs(ic) < min_ic:
                continue
            name = str(row.get("factor_name") or "")
            if name:
                names.append(name)
    if names:
        return sorted(set(names))
    cat_path = catalog_path or (work / "screening_reeval_catalog.json")
    if cat_path.exists():
        return [e["function_name"] for e in json.loads(cat_path.read_text(encoding="utf-8"))]
    return []


def compute_corr(
    work: Path,
    names: list[str],
    *,
    sample_days: int = 40,
    min_names: int = 50,
) -> dict[str, Any]:
    """Mean cross-sectional Spearman corr, day-by-day (avoids O(n²) DuckDB self-join OOM)."""
    paths: list[tuple[str, Path]] = []
    for name in names:
        p = resolve_factor_values_parquet(work, name)
        if p is not None and p.is_file() and p.stat().st_size > 0:
            paths.append((name, p))
    if len(paths) < 2:
        return {"factors": [], "matrix": [], "mean_abs_offdiag": None, "n_pairs": 0}

    factors = [n for n, _ in paths]
    n_f = len(factors)
    name_to_i = {n: i for i, n in enumerate(factors)}

    con = duckdb.connect()
    sum_corr = np.zeros((n_f, n_f), dtype=np.float64)
    cnt_corr = np.zeros((n_f, n_f), dtype=np.int32)
    try:
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit = '3GB'")
        probe_sql = _factor_sql(paths[0][1], paths[0][0])
        date_rows = con.execute(
            f"""
            SELECT trade_date FROM (
              SELECT DISTINCT trade_date FROM ({probe_sql})
            ) d
            ORDER BY trade_date DESC
            LIMIT {int(sample_days)}
            """
        ).fetchall()
        keep_dates = sorted({int(r[0]) for r in date_rows})
        if not keep_dates:
            return {"factors": [], "matrix": [], "mean_abs_offdiag": None, "n_pairs": 0}

        # One factor at a time into a slim table, then process dates sequentially.
        con.execute(
            "CREATE TEMP TABLE fac_day (trade_date INTEGER, symbol VARCHAR, value DOUBLE, fi INTEGER)"
        )
        date_list = ",".join(str(d) for d in keep_dates)
        for name, path in paths:
            fi = name_to_i[name]
            sql = _factor_sql(path, name)
            con.execute(
                f"""
                INSERT INTO fac_day
                SELECT trade_date, symbol, value, {fi}
                FROM ({sql})
                WHERE trade_date IN ({date_list})
                  AND value IS NOT NULL AND isfinite(value)
                """
            )

        for td in keep_dates:
            # ranks per factor for this day
            rows = con.execute(
                f"""
                WITH d AS (
                  SELECT symbol, value, fi
                  FROM fac_day
                  WHERE trade_date = {int(td)}
                ),
                cnt AS (
                  SELECT fi, count(*) AS n FROM d GROUP BY fi
                ),
                ranked AS (
                  SELECT d.symbol, d.fi,
                         RANK() OVER (PARTITION BY d.fi ORDER BY d.value) AS rk
                  FROM d
                  JOIN cnt c ON d.fi = c.fi
                  WHERE c.n >= {int(min_names)}
                )
                SELECT symbol, fi, rk FROM ranked
                """
            ).fetchall()
            if not rows:
                continue
            # pivot: symbol -> vector of ranks (nan if missing)
            sym_idx: dict[str, int] = {}
            syms: list[str] = []
            for sym, _fi, _rk in rows:
                if sym not in sym_idx:
                    sym_idx[sym] = len(syms)
                    syms.append(sym)
            mat = np.full((len(syms), n_f), np.nan, dtype=np.float64)
            for sym, fi, rk in rows:
                mat[sym_idx[sym], int(fi)] = float(rk)
            # Vectorized pairwise Spearman via Pearson-on-ranks (pandas handles NaN).
            import pandas as pd  # local import keeps module light when unused

            cmat = pd.DataFrame(mat).corr(min_periods=int(min_names)).to_numpy(dtype=float)
            finite = np.isfinite(cmat)
            np.fill_diagonal(finite, False)
            sum_corr[finite] += cmat[finite]
            cnt_corr[finite] += 1
            del mat, rows, sym_idx, syms, cmat
        con.execute("DROP TABLE fac_day")
    finally:
        con.close()

    mat_out = np.eye(n_f, dtype=float)
    pairs = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            if cnt_corr[i, j] <= 0:
                continue
            c = float(sum_corr[i, j] / cnt_corr[i, j])
            mat_out[i, j] = c
            mat_out[j, i] = c
            pairs.append(
                {
                    "a": factors[i],
                    "b": factors[j],
                    "mean_rank_corr": c,
                    "n_days": int(cnt_corr[i, j]),
                }
            )

    off = mat_out[np.triu_indices(n_f, k=1)]
    mean_abs = float(np.nanmean(np.abs(off))) if len(off) else None
    high = sorted(
        [p for p in pairs if abs(p["mean_rank_corr"]) >= 0.7],
        key=lambda x: -abs(x["mean_rank_corr"]),
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "n_factors": n_f,
        "sample_days": sample_days,
        "mean_abs_offdiag": mean_abs,
        "factors": factors,
        "matrix": mat_out.tolist(),
        "high_corr_pairs": high,
        "n_pairs": len(pairs),
        "method": "day_by_day_rank_pearson",
    }


def _short_label(name: str, max_len: int = 28) -> str:
    s = name
    for prefix in ("factor_", "week2_", "alphasage_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _order_for_heatmap(mat: np.ndarray) -> list[int]:
    """Cluster by |ρ| so related factors sit together; fallback to mean |ρ|."""
    n = mat.shape[0]
    if n <= 2:
        return list(range(n))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        dist = np.clip(1.0 - np.abs(mat), 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) * 0.5
        # numerical floor so squareform is happy
        dist = np.maximum(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        return leaves_list(Z).tolist()
    except Exception:  # noqa: BLE001
        peer = []
        for i in range(n):
            vals = [abs(mat[i, j]) for j in range(n) if i != j]
            peer.append(float(np.mean(vals)) if vals else 0.0)
        return list(np.argsort(-np.asarray(peer)))


def _heatmap_png_b64(names: list[str], sub: np.ndarray) -> str:
    """Render a labeled Spearman heatmap as base64 PNG (full matrix)."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    if n <= 40:
        cell, fs, lab_len, dpi, annotate = 0.42, 8.0, 28, 140, True
    elif n <= 90:
        cell, fs, lab_len, dpi, annotate = 0.22, 5.0, 18, 120, False
    else:
        cell, fs, lab_len, dpi, annotate = 0.14, 3.2, 14, 110, False

    labels = [_short_label(x, lab_len) for x in names]
    fig_w = max(10.0, cell * n + 4.0)
    fig_h = max(8.5, cell * n + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(axis="both", length=0, pad=1)
    ax.set_title(
        f"All factors (n={n}) · mean cross-sectional Spearman ρ",
        fontsize=12 if n <= 90 else 11,
        pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03 if n > 80 else 0.046, pad=0.02)
    cbar.set_label("rank corr", fontsize=9)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = float(sub[i, j])
                if i == j or abs(v) < 0.85:
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=6.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_corr_html(payload: dict[str, Any], out_path: Path) -> str:
    """Write standalone corr page; return HTML fragment for homepage embed."""
    factors = payload.get("factors") or []
    matrix = payload.get("matrix") or []
    high = payload.get("high_corr_pairs") or []
    # Rebuild from full matrix if payload was truncated by older runs.
    if payload.get("matrix") and payload.get("factors"):
        factors_all = payload["factors"]
        mat_all = payload["matrix"]
        n_all = len(factors_all)
        old_days = {
            (min(p["a"], p["b"]), max(p["a"], p["b"])): int(p.get("n_days") or 0)
            for p in high
            if p.get("a") and p.get("b")
        }
        default_days = int(payload.get("sample_days") or 0)
        rebuilt = []
        for i in range(n_all):
            for j in range(i + 1, n_all):
                c = float(mat_all[i][j])
                if abs(c) < 0.7:
                    continue
                a, b = factors_all[i], factors_all[j]
                key = (min(a, b), max(a, b))
                rebuilt.append(
                    {
                        "a": a,
                        "b": b,
                        "mean_rank_corr": c,
                        "n_days": old_days.get(key) or default_days,
                    }
                )
        if len(rebuilt) >= len(high):
            high = sorted(rebuilt, key=lambda x: -abs(x["mean_rank_corr"]))
            payload["high_corr_pairs"] = high
    rows = []
    for p in high:
        rows.append(
            f"<tr><td>{html_escape(p['a'])}</td><td>{html_escape(p['b'])}</td>"
            f"<td>{p['mean_rank_corr']:.3f}</td><td>{p.get('n_days', '—')}</td></tr>"
        )
    n = len(factors)
    mat = np.asarray(matrix, dtype=float)
    peer = []
    for i, name in enumerate(factors):
        vals = [abs(mat[i, j]) for j in range(n) if i != j]
        peer.append((name, float(np.mean(vals)) if vals else 0.0))
    peer.sort(key=lambda x: -x[1])

    order = _order_for_heatmap(mat)
    ordered_names = [factors[i] for i in order]
    sub = mat[np.ix_(order, order)]
    try:
        img_b64 = _heatmap_png_b64(ordered_names, sub)
        heat = (
            f'<img class="corr-heat" alt="factor rank correlation heatmap" '
            f'src="data:image/png;base64,{img_b64}"/>'
        )
    except Exception as exc:  # noqa: BLE001
        heat = f'<p class="muted">热力图生成失败：{html_escape(str(exc))}</p>'

    peer_by_name = {name: v for name, v in peer}
    labels = "".join(
        f"<li><code>{html_escape(x)}</code> · 平均|ρ|={peer_by_name.get(x, 0.0):.3f}</li>"
        for x, _ in peer
    )
    pair_table = (
        '<table class="list"><thead><tr><th>A</th><th>B</th><th>mean rank corr</th><th>n days</th></tr></thead>'
        f"<tbody>{''.join(rows) if rows else '<tr><td colspan=4>无 |ρ|≥0.7 的对</td></tr>'}</tbody></table>"
    )
    mean_abs = payload.get("mean_abs_offdiag")
    mean_abs_s = "—" if mean_abs is None else f"{float(mean_abs):.3f}"
    annot_note = (
        "格子上标出 |ρ|≥0.85 的数值。"
        if n <= 40
        else "因子较多，格子不标数值以免糊成一片；高相关对见下表。"
    )
    fragment = f"""
<section class="corr-embed" id="factor-corr">
  <h2>因子截面秩相关矩阵</h2>
  <p class="muted">生成 {html_escape(str(payload.get('generated_at') or ''))} · 因子数 {payload.get('n_factors')} ·
  采样交易日 {payload.get('sample_days')} · 非对角 |corr| 均值 {mean_abs_s}
  · <a href="/reports/factor_rank_corr_matrix.html">打开完整页</a></p>
  <p>口径：每个交易日做因子截面 Spearman 相关，再对交易日取平均。下图为<strong>全部 {n} 个因子</strong>
  的完整相关矩阵（按 |ρ| 层次聚类排序；红=正相关，蓝=负相关）。{annot_note}</p>
  {heat}
  <details><summary>全部因子名单（按平均|ρ|降序，共 {n}）</summary><ol>{labels}</ol></details>
  <h3>高相关对 (|ρ|≥0.7，共 {len(high)} 对，已全部列出)</h3>
  <div class="corr-pair-wrap">{pair_table}</div>
</section>
"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>因子截面秩相关矩阵</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
img.corr-heat{{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:8px;background:#fff}}
table.list{{border-collapse:collapse;width:100%;background:#fff;margin-top:12px}}
table.list th,table.list td{{border:1px solid #e2e8f0;padding:8px;font-size:13px;word-break:break-all}}
.corr-pair-wrap{{max-height:70vh;overflow:auto;border:1px solid #e2e8f0;border-radius:8px;background:#fff}}
.muted{{color:#64748b}}
details{{margin:14px 0}}
</style></head><body>
{fragment}
<p><a href="/reports/factor_rankic_screening_index.html">返回因子索引</a></p>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return fragment


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--sample-days", type=int, default=40)
    parser.add_argument("--min-ic", type=float, default=0.02, help="Only factors with |RankIC|>=min-ic")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--names-file", type=Path, default=None)
    args = parser.parse_args()
    work = args.work_dir
    if args.names_file and args.names_file.exists():
        names = [ln.strip() for ln in args.names_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        names = _names_for_corr(work, args.catalog, min_ic=args.min_ic)
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
