#!/usr/bin/env python3
"""Build unified RankIC > threshold factor index across all local report sources."""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WORK_DIR = ROOT / "data/cogalpha_lqtp_production"
DEFAULT_THRESHOLD = 0.02

# href may be in-page #batch_factor_N (Top30 detail) or absolute /reports_screening_reeval/...
SCREENING_ROW_RE = re.compile(
    r"<td>(\d+)</td><td>(\d+)</td>"
    r"<td><a href=['\"]([^'\"]+)['\"]>(factor_\d+)</a></td>"
    r"<td>([^<]+)</td><td>([^<]+)</td>"
)


@dataclass
class FactorRow:
    factor_id: str
    display_name: str
    rank_ic: float
    source: str
    report_href: str
    route: str = ""
    rank_icir: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    dedupe_key: str = ""

    def __post_init__(self) -> None:
        if not self.dedupe_key:
            self.dedupe_key = self.display_name.strip().lower()


def _parse_pct(text: str) -> float | None:
    s = str(text or "").strip().replace("—", "").replace("-", "")
    if not s or s == "nan":
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    try:
        v = float(s)
        return v / 100.0 if abs(v) > 1.5 else v
    except ValueError:
        return None


def _parse_mean_rank_ic_from_report(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<b>([0-9.+\-eE]+)</b><span>Mean RankIC", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r'"mean_rank_ic"\s*:\s*([0-9.eE+\-]+)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _load_cogalpha_batch(work: Path) -> list[FactorRow]:
    rows: list[FactorRow] = []
    reports = work / "reports"
    index_by_name: dict[str, dict[str, Any]] = {}
    prog_path = work / "production_progress.json"
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        for item in prog.get("index_rows", []):
            name = str(item.get("factor_name", ""))
            if name:
                index_by_name[name] = item

    seen: set[str] = set()
    for html_path in sorted(reports.glob("factor_*.html")):
        name = html_path.stem
        if name in seen:
            continue
        seen.add(name)
        ic = _parse_mean_rank_ic_from_report(html_path)
        if ic is None:
            meta = index_by_name.get(name, {})
            try:
                ic = float(meta.get("mean_rank_ic", meta.get("mean_ic")))
            except (TypeError, ValueError):
                continue
        meta = index_by_name.get(name, {})
        rows.append(
            FactorRow(
                factor_id=name,
                display_name=name,
                rank_ic=float(ic),
                source="cogalpha_lqtp (7/18)",
                report_href=f"/reports/{html_path.name}",
                route=str(meta.get("engine") or meta.get("eval_route") or ""),
                rank_icir=_safe_float(meta.get("rank_icir", meta.get("icir"))),
            )
        )
    return rows


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _load_report_7_26(work: Path) -> list[FactorRow]:
    rows: list[FactorRow] = []
    base = work / "reports_7_26"
    sel_path = base / "selected_factors.json"
    if not sel_path.exists():
        return rows
    payload = json.loads(sel_path.read_text(encoding="utf-8"))
    for item in payload.get("factors", []):
        key = str(item.get("factor_key") or item.get("factor_id") or "")
        name = str(item.get("factor_name") or key)
        ic = _safe_float(item.get("mean_rank_ic"))
        if ic is None or not key:
            continue
        html_name = f"{key}.html"
        if not (base / html_name).exists() and str(item.get("factor_id", "")).startswith("alphasage_"):
            html_name = f"{item['factor_id']}.html"
        rows.append(
            FactorRow(
                factor_id=key,
                display_name=name,
                rank_ic=ic,
                source="factor_pool 优选 (7/26)",
                report_href=f"/reports_7_26/{html_name}",
                route=str(item.get("route") or ""),
                dedupe_key=name.strip().lower(),
            )
        )
    return rows


def _load_screening_html(work: Path) -> list[FactorRow]:
    rows: list[FactorRow] = []
    path = work / "reports/cogalpha_factor_screening_top30.html"
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    # Only Top30 detail cards have id= anchors; RankIC/Q10 leaderboard rows often link to missing hashes.
    detail_ids = set(re.findall(r'id=["\'](\d+_factor_\d+)["\']', text))
    reeval_dir = work / "reports_screening_reeval"
    id_to_fn: dict[str, str] = {}
    cat_path = work / "screening_reeval_catalog.json"
    if cat_path.exists():
        for entry in json.loads(cat_path.read_text(encoding="utf-8")):
            fid = str(entry.get("factor_id") or "")
            fn = str(entry.get("function_name") or "")
            if fid and fn:
                id_to_fn[fid] = fn

    best: dict[str, FactorRow] = {}
    for m in SCREENING_ROW_RE.finditer(text):
        batch, href_raw, fid, ic_raw, icir_raw = (
            m.group(2),
            m.group(3),
            m.group(4),
            m.group(5),
            m.group(6),
        )
        ic = _parse_pct(ic_raw)
        if ic is None:
            continue
        icir = _parse_pct(icir_raw)
        anchor = ""
        if href_raw.startswith("#"):
            anchor = href_raw[1:]
        elif re.fullmatch(r"\d+_factor_\d+", href_raw):
            anchor = href_raw
        else:
            # reconstructed anchor for dedupe when href was remapped off-page
            anchor = f"{batch}_{fid}"
        fn = id_to_fn.get(fid, "")
        reeval_report = reeval_dir / f"{fn}.html" if fn else None
        if reeval_report is not None and reeval_report.is_file():
            href = f"/reports_screening_reeval/{fn}.html"
        elif anchor in detail_ids:
            href = f"/reports/cogalpha_factor_screening_top30.html#{anchor}"
        elif href_raw.startswith("/reports_screening_reeval/") and (work / href_raw.lstrip("/")).is_file():
            href = href_raw
        else:
            # Avoid dead in-page hashes (browser stays at top of 26MB report).
            href = "/reports/cogalpha_factor_screening_top30.html"
        row = FactorRow(
            factor_id=anchor or fid,
            display_name=fid,
            rank_ic=ic,
            source=f"screening results/{batch}",
            report_href=href,
            rank_icir=icir,
            dedupe_key=f"screening:{anchor or fid}".lower(),
            extra={"batch": batch},
        )
        prev = best.get(row.dedupe_key)
        if prev is None or row.rank_ic > prev.rank_ic:
            best[row.dedupe_key] = row
    rows.extend(best.values())
    return rows


def _load_external_packs(work: Path) -> list[FactorRow]:
    rows: list[FactorRow] = []
    cat_path = ROOT / "data/external_factor_packs/unified_catalog.json"
    if not cat_path.exists():
        return rows
    payload = json.loads(cat_path.read_text(encoding="utf-8"))
    for item in payload.get("factors", []):
        ic = _safe_float(item.get("rank_ic"))
        if ic is None:
            continue
        pack = str(item.get("pack") or "external")
        display = str(item.get("display_name") or "")
        fn = str(item.get("function_name") or display)
        rows.append(
            FactorRow(
                factor_id=fn,
                display_name=display,
                rank_ic=float(ic),
                source=f"external pack ({pack})",
                report_href=f"/reports_screening_reeval/{fn}.html",
                route=str(item.get("route") or ""),
                dedupe_key=display.lower(),
                extra={
                    "name_zh": (item.get("extra") or {}).get("name_zh", ""),
                    "formula": item.get("formula", ""),
                    "pack": pack,
                },
            )
        )
    return rows


def _load_week2(work: Path) -> tuple[list[FactorRow], list[dict[str, Any]]]:
    rows: list[FactorRow] = []
    details: list[dict[str, Any]] = []
    cat_path = ROOT / "week2_pv_factors/source/week2_factors_catalog.json"
    if not cat_path.exists():
        return rows, details
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    for item in cat.get("factors", []):
        ic_pct = _safe_float(item.get("rank_ic_pct"))
        if ic_pct is None:
            continue
        ic = ic_pct / 100.0
        slug = str(item.get("slug") or item.get("id") or "")
        anchor = f"week2_{item.get('id', slug)}"
        rows.append(
            FactorRow(
                factor_id=str(item.get("id", slug)),
                display_name=f"week2:{slug}",
                rank_ic=ic,
                source="Week2 价量目录",
                report_href=f"#{anchor}",
                route=item.get("tier", ""),
                dedupe_key=f"week2:{slug}".lower(),
                extra={"name_zh": item.get("name_zh", ""), "formula": item.get("formula", "")},
            )
        )
        details.append({**item, "anchor": anchor, "rank_ic": ic})
    return rows, details


def _is_vwap_panel_row(row: dict[str, Any]) -> bool:
    mode = str(row.get("eval_mode") or "")
    kind = str(row.get("return_kind") or "")
    return "vwap" in mode or "vwap" in kind


def _ic_is_usable(row: dict[str, Any]) -> bool:
    """Homepage ranks by RankIC only; ignore LS blow-ups (VWAP LS cum can explode)."""
    ic = _safe_float(row.get("mean_rank_ic", row.get("mean_ic")))
    if ic is None or ic != ic:
        return False
    # RankIC itself outside (-1,1) is impossible for Spearman.
    if abs(ic) > 1.0 + 1e-9:
        return False
    return True


def _load_progress_ic_maps(work: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (vwap_panel_for_homepage, current_progress).

    Homepage ranking uses **only** VWAP→VWAP panel metrics from screening_reeval_progress.
    """
    current: dict[str, dict[str, Any]] = {}
    preferred: dict[str, dict[str, Any]] = {}
    prog_path = work / "screening_reeval_progress.json"
    if prog_path.exists():
        for row in json.loads(prog_path.read_text(encoding="utf-8")).get("index_rows", []):
            name = str(row.get("factor_name") or "")
            if not name:
                continue
            current[name] = row
            if _is_vwap_panel_row(row) and _ic_is_usable(row):
                preferred[name] = row
    return preferred, current


def _apply_reeval_overlay(work: Path, rows: list[FactorRow]) -> dict[str, int]:
    """Overwrite RankIC/ICIR with VWAP→VWAP panel metrics only."""
    cat_path = work / "screening_reeval_catalog.json"
    stats = {
        "linked": 0,
        "ic_from_vwap": 0,
        "ic_pending_vwap": 0,
        "renamed": 0,
    }
    if not cat_path.exists():
        return stats
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    display_to_fn = {
        str(e.get("manifest_display_name") or ""): str(e["function_name"])
        for e in catalog
        if e.get("function_name")
    }
    fn_set = {str(e["function_name"]) for e in catalog if e.get("function_name")}
    id_to_fn = {
        str(e.get("factor_id") or ""): str(e["function_name"])
        for e in catalog
        if e.get("factor_id") and e.get("function_name")
    }
    preferred, _current = _load_progress_ic_maps(work)
    reeval_dir = work / "reports_screening_reeval"

    for row in rows:
        fn = (
            display_to_fn.get(row.display_name)
            or id_to_fn.get(row.display_name)
            or (row.display_name if row.display_name in fn_set else "")
            or (row.factor_id if row.factor_id in fn_set else "")
        )
        if not fn:
            # week2:slug → week2_slug
            if row.display_name.startswith("week2:"):
                cand = "week2_" + row.display_name.split(":", 1)[1]
                if cand in fn_set:
                    fn = cand
        if not fn:
            continue

        report = reeval_dir / f"{fn}.html"
        if report.is_file():
            row.report_href = f"/reports_screening_reeval/{fn}.html"
            stats["linked"] += 1

        if fn and fn != row.display_name and (
            row.display_name.startswith("factor_") and row.display_name[7:].isdigit()
            or row.display_name.startswith(("week2:", "extra20:", "weekly:"))
        ):
            row.extra = {**row.extra, "alias": row.display_name}
            row.display_name = fn
            stats["renamed"] += 1

        # Canonicalize dedupe onto the real function name.
        row.dedupe_key = fn.strip().lower()
        row.factor_id = fn

        meta = preferred.get(fn)
        if meta is not None:
            ic = _safe_float(meta.get("mean_rank_ic", meta.get("mean_ic")))
            if ic is not None:
                row.rank_ic = abs(ic)  # sign-reconciled display
                icir = _safe_float(meta.get("rank_icir", meta.get("icir")))
                row.rank_icir = abs(icir) if icir is not None else None
                row.source = "screening_reeval (VWAP→VWAP)"
                row.route = str(meta.get("engine") or meta.get("eval_route") or row.route or "")
                ex = dict(row.extra or {})
                for k in ("mean_daily_coverage", "ls_mean_one_way_turnover", "long_short_sharpe"):
                    if meta.get(k) is not None:
                        ex[k] = meta.get(k)
                row.extra = ex
                stats["ic_from_vwap"] += 1
                continue

        # No VWAP panel yet → exclude from homepage ranking (do not show close/source IC).
        row.rank_ic = float("nan")
        row.rank_icir = None
        stats["ic_pending_vwap"] += 1
    return stats


def dedupe_rows(rows: list[FactorRow]) -> list[FactorRow]:
    best: dict[str, FactorRow] = {}
    for row in rows:
        key = row.dedupe_key or row.display_name.strip().lower()
        prev = best.get(key)
        if prev is None or row.rank_ic > prev.rank_ic:
            best[key] = row
    return sorted(best.values(), key=lambda r: r.rank_ic, reverse=True)


def _fmt_opt(v: Any, *, pct: bool = False, digits: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    if pct:
        return f"{f:.1%}"
    return f"{f:.{digits}f}"


def render_html(
    *,
    rows: list[FactorRow],
    week2_details: list[dict[str, Any]],
    threshold: float,
    out_path: Path,
    work_dir: Path,
    corr_fragment: str = "",
) -> None:
    from scripts.cogalpha_lqtp.report_theme import SUMMARY_CSS  # noqa: E402

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pos = sum(1 for r in rows if r.rank_ic > 0)
    table_rows = []
    toc_items = []
    for i, r in enumerate(rows, 1):
        icir = f"{r.rank_icir:.4f}" if r.rank_icir is not None else "—"
        ex = r.extra or {}
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{html_lib.escape(r.report_href)}">{html_lib.escape(r.display_name)}</a></td>'
            f"<td>{html_lib.escape(r.source)}</td>"
            f"<td>{html_lib.escape(r.route or '—')}</td>"
            f"<td>{r.rank_ic:.4f}</td>"
            f"<td>{icir}</td>"
            f"<td>{_fmt_opt(ex.get('mean_daily_coverage'), pct=True)}</td>"
            f"<td>{_fmt_opt(ex.get('ls_mean_one_way_turnover'), pct=True)}</td>"
            f"<td>{_fmt_opt(ex.get('long_short_sharpe'), digits=3)}</td>"
            "</tr>"
        )
        toc_items.append(
            f'<li><a href="{html_lib.escape(r.report_href)}">{html_lib.escape(r.display_name)}</a>'
            f' <span class="muted">({r.rank_ic:.2%})</span></li>'
        )

    w2_blocks = []
    w2_anchors = {d["anchor"] for d in week2_details}
    for d in week2_details:
        if d["anchor"] not in w2_anchors:
            continue
        if not any(r.report_href == f"#{d['anchor']}" for r in rows):
            continue
        w2_blocks.append(
            f"""<section class="card" id="{html_lib.escape(d['anchor'])}">
  <h3>{html_lib.escape(d.get('id',''))} · {html_lib.escape(d.get('name_zh',''))}</h3>
  <p class="muted">slug: <code>{html_lib.escape(d.get('slug',''))}</code> · RankIC: {d['rank_ic']:.2%} · tier: {html_lib.escape(str(d.get('tier','')))}</p>
  <pre>{html_lib.escape(d.get('formula',''))}</pre>
  <p class="note">{html_lib.escape(d.get('note',''))}</p>
</section>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>因子精选 · RankIC &gt; {threshold:.0%}</title>
  <style>
{SUMMARY_CSS}
    .hero-sub {{ margin: 8px 0 0; opacity: 0.9; }}
    .cards {{ grid-template-columns: repeat(auto-fit,minmax(130px,1fr)); }}
    .toc {{ columns: 2; column-gap: 28px; font-size: 14px; margin: 16px 0; }}
    .toc li {{ margin: 6px 0; break-inside: avoid; }}
    .week2-section {{ margin-top: 40px; }}
    .corr-embed {{ margin: 40px 0 12px; padding: 18px 20px; background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; }}
    .corr-embed img.corr-heat {{ max-width: 100%; height: auto; display: block; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }}
    .corr-embed .corr-pair-wrap {{ max-height: 70vh; overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }}
    .corr-embed table.list {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    .corr-embed table.list th, .corr-embed table.list td {{ border: 1px solid #e2e8f0; padding: 6px 8px; font-size: 13px; word-break: break-all; }}
    @media (max-width:900px) {{ .toc {{ columns: 1; }} }}
  </style>
</head>
<body>
<header class="hero">
  <h1>因子精选汇总（RankIC &gt; {threshold:.0%}）</h1>
  <p class="hero-sub muted">VWAP→VWAP 面板筛选；按 RankIC 降序，同名去重保留更高 IC</p>
  <p class="muted">生成时间 {generated} · 工作目录 <code>{html_lib.escape(str(work_dir))}</code></p>
</header>
<main>
  <section>
    <div class="cards">
      <div class="metric"><b>{len(rows)}</b><span>入选因子（去重后）</span></div>
      <div class="metric"><b>{threshold:.0%}</b><span>RankIC 阈值</span></div>
      <div class="metric"><b>{pos}</b><span>RankIC &gt; 0</span></div>
    </div>
    <div class="notice">
      <p><b>收益口径</b>：本页 RankIC / RankICIR <b>全部</b>为 VWAP→VWAP（收盘 T 信号 →
      vwap(T+2)/vwap(T+1)-1，避免 vwap(T+1)/vwap(T) 把当日 VWAP 泄漏进收益）。不混用 close。
      尚未完成 VWAP 重评的因子不进入本表。负向因子已在公式/落值取负，表中 RankIC≥0。</p>
      <p><b>多空扣费</b>：按真实 G10/G1 等权组合日换手扣佣金（非固定 40%）。表中「多空换手」为两腿单边换手之和的日均。</p>
      <p><b>覆盖率</b>：当日（因子∩收益）股票数 / 收益宇宙股票数。</p>
      <p><b>因子相关</b>：见文末热力图与高相关对表，或 <a href="/reports/factor_rank_corr_matrix.html">完整页</a>。</p>
    </div>
  </section>

  <section>
    <h2>快速目录</h2>
    <ul class="toc">{''.join(toc_items)}</ul>
  </section>

  <section>
    <h2>因子列表（按 Mean RankIC 降序）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>因子</th><th>来源</th><th>算值路径</th><th>Mean RankIC</th><th>RankICIR</th>
          <th>日覆盖率</th><th>多空换手</th><th>多空Sharpe</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </section>

  {"<section class='week2-section'><h2>Week2 因子详情（页内）</h2>" + ''.join(w2_blocks) + "</section>" if w2_blocks else ""}

  {corr_fragment}
</main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render unified RankIC screening index")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: work-dir/reports/factor_rankic_screening_index.html)",
    )
    parser.add_argument(
        "--skip-corr",
        action="store_true",
        help="Skip in-process corr matrix (reuse existing fragment/link; avoids OOM)",
    )
    parser.add_argument(
        "--corr-sample-days",
        type=int,
        default=40,
        help="Sample days for corr matrix when not --skip-corr (default 40)",
    )
    args = parser.parse_args()
    work = args.work_dir
    out = args.out or (work / "reports/factor_rankic_screening_index.html")

    all_rows: list[FactorRow] = []
    all_rows.extend(_load_cogalpha_batch(work))
    all_rows.extend(_load_report_7_26(work))
    all_rows.extend(_load_screening_html(work))
    week2_rows, week2_details = _load_week2(work)
    all_rows.extend(week2_rows)
    all_rows.extend(_load_external_packs(work))
    # Inject VWAP→VWAP panel rows as first-class homepage sources.
    preferred_map, _current_map = _load_progress_ic_maps(work)
    for name, item in preferred_map.items():
        ic = _safe_float(item.get("mean_rank_ic", item.get("mean_ic")))
        if not name or ic is None:
            continue
        href = f"/reports_screening_reeval/{name}.html"
        if not (work / "reports_screening_reeval" / f"{name}.html").is_file():
            href = f"/reports/{name}.html"
        icir = _safe_float(item.get("rank_icir", item.get("icir")))
        all_rows.append(
            FactorRow(
                factor_id=name,
                display_name=name,
                rank_ic=abs(float(ic)),
                source="screening_reeval (VWAP→VWAP)",
                report_href=href,
                route=str(item.get("engine") or item.get("eval_route") or ""),
                rank_icir=abs(icir) if icir is not None else None,
                dedupe_key=name.strip().lower(),
                extra={
                    "mean_daily_coverage": item.get("mean_daily_coverage"),
                    "ls_mean_one_way_turnover": item.get("ls_mean_one_way_turnover"),
                    "long_short_sharpe": item.get("long_short_sharpe"),
                },
            )
        )

    # Overlay: force every listed factor onto VWAP panel IC (or drop if pending).
    overlay_stats = _apply_reeval_overlay(work, all_rows)
    # Attach coverage / LS turnover from VWAP progress when available.
    for r in all_rows:
        item = preferred_map.get(r.display_name) or preferred_map.get(r.factor_id)
        if not item:
            continue
        ex = dict(r.extra or {})
        for k in ("mean_daily_coverage", "ls_mean_one_way_turnover", "long_short_sharpe"):
            if item.get(k) is not None:
                ex[k] = item.get(k)
        r.extra = ex
    finite_rows = [r for r in all_rows if r.rank_ic == r.rank_ic]
    filtered = [r for r in finite_rows if r.rank_ic > args.threshold]
    deduped = dedupe_rows(filtered)

    corr_fragment = ""
    corr_html_path = work / "reports" / "factor_rank_corr_matrix.html"
    if args.skip_corr:
        if corr_html_path.is_file():
            corr_fragment = (
                '<section class="corr-embed"><h2>因子截面秩相关矩阵</h2>'
                '<p class="muted">见独立页 '
                '<a href="/reports/factor_rank_corr_matrix.html">factor_rank_corr_matrix.html</a>'
                "（本次跳过重算以控制内存）。</p></section>"
            )
            print("corr skipped (--skip-corr); linking existing page")
        else:
            corr_fragment = (
                '<section class="corr-embed"><h2>因子截面秩相关矩阵</h2>'
                '<p class="muted">尚未生成；重评完成后单独跑 '
                "<code>python -m scripts.cogalpha_lqtp.compute_factor_corr_matrix</code></p></section>"
            )
            print("corr skipped (--skip-corr); no existing page")
    else:
        try:
            from scripts.cogalpha_lqtp.compute_factor_corr_matrix import (  # noqa: E402
                compute_corr,
                render_corr_html,
            )

            corr_json = work / "reports" / "factor_rank_corr_matrix.json"
            corr_payload = None
            if corr_json.is_file():
                try:
                    corr_payload = json.loads(corr_json.read_text(encoding="utf-8"))
                    if not (corr_payload.get("factors") and corr_payload.get("matrix")):
                        corr_payload = None
                except Exception:  # noqa: BLE001
                    corr_payload = None
            if corr_payload is None:
                corr_names = [r.display_name for r in deduped]
                corr_payload = compute_corr(
                    work, corr_names, sample_days=max(20, int(args.corr_sample_days))
                )
                corr_json.write_text(
                    json.dumps(corr_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            corr_fragment = render_corr_html(corr_payload, corr_html_path)
            # persist rebuilt high_corr_pairs (older JSON may have been truncated)
            corr_json.write_text(
                json.dumps(corr_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"corr n={corr_payload.get('n_factors')} pairs={corr_payload.get('n_pairs')} "
                f"high={len(corr_payload.get('high_corr_pairs') or [])} "
                f"mean_abs={corr_payload.get('mean_abs_offdiag')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"corr matrix skipped: {exc}")
            corr_fragment = (
                '<section class="corr-embed"><h2>因子截面秩相关矩阵</h2>'
                f'<p class="muted">生成失败：{html_lib.escape(str(exc))}；可稍后重跑 '
                "<code>python -m scripts.cogalpha_lqtp.compute_factor_corr_matrix</code></p></section>"
            )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "threshold": args.threshold,
        "raw_count": len(all_rows),
        "filtered_count": len(filtered),
        "deduped_count": len(deduped),
        "overlay_stats": overlay_stats,
        "rank_ic_policy": "vwap_to_vwap_only; display_nonneg; pending_excluded",
        "factors": [
            {
                "display_name": r.display_name,
                "rank_ic": r.rank_ic,
                "rank_icir": r.rank_icir,
                "source": r.source,
                "report_href": r.report_href,
                "route": r.route,
                "alias": (r.extra or {}).get("alias", ""),
                "mean_daily_coverage": (r.extra or {}).get("mean_daily_coverage"),
                "ls_mean_one_way_turnover": (r.extra or {}).get("ls_mean_one_way_turnover"),
                "long_short_sharpe": (r.extra or {}).get("long_short_sharpe"),
            }
            for r in deduped
        ],
    }
    (work / "factor_rankic_screening_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    render_html(
        rows=deduped,
        week2_details=week2_details,
        threshold=args.threshold,
        out_path=out,
        work_dir=work,
        corr_fragment=corr_fragment,
    )
    print(
        f"ok {len(deduped)} factors "
        f"(raw={len(all_rows)}, finite={len(finite_rows)}, after_filter={len(filtered)}) "
        f"overlay={overlay_stats} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
