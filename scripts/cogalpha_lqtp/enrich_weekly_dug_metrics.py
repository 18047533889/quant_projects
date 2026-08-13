#!/usr/bin/env python3
"""Enrich weekly-dug: keep |RankIC|>thr, annotate neutralization, fill neu-IC for raw ones, corr.

Memory-safe: sequential materialize + DuckDB extended eval; day-by-day corr.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.compute_factor_corr_matrix import (  # noqa: E402
    _heatmap_png_b64,
    _order_for_heatmap,
    compute_corr,
)
from scripts.cogalpha_lqtp.crawl_candidate_pool_neutral_rankic import (  # noqa: E402
    patch_screening_html,
)
from scripts.cogalpha_lqtp.data_access_panel import (  # noqa: E402
    ashare_materialize_data_source_config,
    resolve_factor_values_parquet,
)
from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    ensure_eval_aux_cache,
    merge_extended_into_analysis,
)
from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb  # noqa: E402
from scripts.cogalpha_lqtp.materialize import materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import (  # noqa: E402
    ensure_memory_floor,
    read_mem_available_gb,
    release_memory,
    wait_for_memory,
)

DEFAULT_WORK = ROOT / "data/cogalpha_lqtp_production"
THRESHOLD = 0.02


def _abs_ic(r: dict[str, Any]) -> float:
    for k in ("display_rank_ic", "abs_mean_rank_ic", "platform_mean_ic", "mean_rank_ic"):
        v = r.get(k)
        try:
            x = abs(float(v))
        except (TypeError, ValueError):
            continue
        if x == x:
            return x
    return 0.0


def _is_already_neutralized(r: dict[str, Any]) -> bool:
    if r.get("already_neutralized") is True:
        return True
    if r.get("from_pack_top_pick") or r.get("materialize") == "platform_only":
        return False
    label = str(r.get("label") or r.get("source_label") or "")
    if "包内DSL" in label:
        return False
    if r.get("neutral_probe") or str(r.get("source") or "") in {
        "evoalpha",
        "alphasage",
        "pool_a",
        "pool_b",
    }:
        return True
    if "中性化候选" in label:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--corr-sample-days", type=int, default=40)
    ap.add_argument("--skip-materialize", action="store_true")
    ap.add_argument("--skip-corr", action="store_true")
    args = ap.parse_args()
    work: Path = args.work_dir
    weekly_path = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(weekly_path.read_text(encoding="utf-8"))

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for r in weekly.get("selected") or []:
        ic = _abs_ic(r)
        if ic <= float(args.threshold):
            dropped.append(f"{r.get('factor_id')}:{ic:.4f}")
            continue
        r["already_neutralized"] = _is_already_neutralized(r)
        if r["already_neutralized"]:
            r["neutralization_note"] = "已中性化（候选池落盘）；免补行业/市值中性IC"
            for k in (
                "industry_neutral_mean_rank_ic",
                "size_neutral_mean_rank_ic",
                "industry_size_neutral_mean_rank_ic",
            ):
                r.pop(k, None)
        else:
            r["neutralization_note"] = "未中性化（公式重算）；补行业/市值/双中性 RankIC"
        kept.append(r)
    kept.sort(key=lambda x: _abs_ic(x), reverse=True)
    weekly["selected"] = kept
    weekly["n_selected"] = len(kept)
    weekly["dropped_below_threshold"] = dropped
    weekly["threshold"] = float(args.threshold)
    print(f"kept={len(kept)} dropped={dropped}", flush=True)

    fwd = work / "lqtp_fwd_vwap_returns_cache.parquet"
    if not fwd.exists():
        raise SystemExit(f"missing fwd cache: {fwd}")
    print("ensure aux cache ...", flush=True)
    aux = ensure_eval_aux_cache(work, start=args.start, end=args.end)
    ind = Path(aux["industry"])
    mcap = Path(aux["market_cap"])
    print(f"aux ind={ind} exists={ind.exists()} mcap={mcap} exists={mcap.exists()}", flush=True)

    need = [r for r in kept if not r["already_neutralized"]]
    print(f"need_neu_metrics={len(need)}", flush=True)

    for i, r in enumerate(need, 1):
        fid = r["factor_id"]
        dsl = (r.get("lqtp_formula") or r.get("dsl") or "").strip()
        print(f"[{i}/{len(need)}] {fid} mem={read_mem_available_gb():.1f}G", flush=True)
        try:
            path = resolve_factor_values_parquet(work, fid)
            if path is None or not path.is_file():
                if args.skip_materialize:
                    r["neu_metrics_error"] = "missing_lake_skipped"
                    print("  skip materialize", flush=True)
                    continue
                wait_for_memory(required_gb=8.0, timeout_sec=900)
                ensure_memory_floor(6.0)
                cfg = ashare_materialize_data_source_config(
                    start_date=args.start, end_date=args.end
                )
                path = materialize_factor(
                    factor_id=fid,
                    dsl=dsl,
                    data_source_cfg=cfg,
                    lake_root=work / "factor_lake",
                )
                release_memory()
                print(f"  wrote {path}", flush=True)

            wait_for_memory(required_gb=4.0, timeout_sec=300)
            analysis = analyze_factor_parquet_duckdb(
                factor_path=path,
                fwd_returns_path=fwd,
                return_kind="vwap_to_vwap",
            )
            extended = compute_extended_eval(
                factor_path=path,
                fwd_returns_path=fwd,
                industry_path=ind if ind.exists() else None,
                market_cap_path=mcap if mcap.exists() else None,
            )
            analysis = merge_extended_into_analysis(analysis, extended)
            if analysis.get("ls_mean_one_way_turnover") is not None:
                r["ls_mean_one_way_turnover"] = analysis["ls_mean_one_way_turnover"]
            if analysis.get("long_short_return") is not None:
                r["long_short_return_sum"] = analysis["long_short_return"]
            if analysis.get("mean_daily_coverage") is not None and r.get("mean_daily_coverage") is None:
                r["mean_daily_coverage"] = analysis["mean_daily_coverage"]
            r["industry_neutral_mean_rank_ic"] = analysis.get("industry_neutral_mean_rank_ic")
            r["industry_neutral_rank_icir"] = analysis.get("industry_neutral_rank_icir")
            r["size_neutral_mean_rank_ic"] = analysis.get("size_neutral_mean_rank_ic")
            r["size_neutral_rank_icir"] = analysis.get("size_neutral_rank_icir")
            r["industry_size_neutral_mean_rank_ic"] = analysis.get(
                "industry_size_neutral_mean_rank_ic"
            )
            r["factor_rank_turnover"] = analysis.get("factor_rank_turnover")
            r["ic_half_life_days"] = analysis.get("ic_half_life_days")
            r["neu_metrics_ok"] = True
            print(
                f"  ind={r.get('industry_neutral_mean_rank_ic')} "
                f"size={r.get('size_neutral_mean_rank_ic')} "
                f"both={r.get('industry_size_neutral_mean_rank_ic')} "
                f"to={r.get('ls_mean_one_way_turnover')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            r["neu_metrics_ok"] = False
            r["neu_metrics_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {exc}", flush=True)
        finally:
            release_memory()
            gc.collect()

    if not args.skip_corr:
        names = []
        for r in kept:
            p = resolve_factor_values_parquet(work, r["factor_id"])
            if p is not None and p.is_file():
                names.append(r["factor_id"])
        print(f"corr names={len(names)}", flush=True)
        wait_for_memory(required_gb=6.0, timeout_sec=600)
        corr = compute_corr(work, names, sample_days=int(args.corr_sample_days), min_names=50)
        if corr.get("factors") and corr.get("matrix"):
            import numpy as np

            mat = np.asarray(corr["matrix"], dtype=float)
            order = _order_for_heatmap(mat)
            names_o = [corr["factors"][i] for i in order]
            sub = mat[np.ix_(order, order)]
            try:
                corr["heatmap_b64"] = _heatmap_png_b64(names_o, sub)
            except Exception as exc:  # noqa: BLE001
                print(f"heatmap fail: {exc}", flush=True)
                corr["heatmap_b64"] = ""
        weekly["weekly_corr"] = {
            "generated_at": corr.get("generated_at"),
            "n_factors": corr.get("n_factors"),
            "sample_days": corr.get("sample_days"),
            "mean_abs_offdiag": corr.get("mean_abs_offdiag"),
            "high_corr_pairs": corr.get("high_corr_pairs") or [],
            "n_pairs": corr.get("n_pairs"),
            "heatmap_b64": corr.get("heatmap_b64") or "",
            "factors": corr.get("factors") or [],
        }
        slim = {k: v for k, v in weekly["weekly_corr"].items() if k != "heatmap_b64"}
        (work / "reports/weekly_dug_corr_matrix.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"corr n={corr.get('n_factors')} mean|ρ|={corr.get('mean_abs_offdiag')} "
            f"high={len(corr.get('high_corr_pairs') or [])}",
            flush=True,
        )
        release_memory()

    weekly["platform_note"] = (
        f"仅保留 |RankIC|>{args.threshold:.0%} 共 {len(kept)} 条。"
        f"已中性化 {sum(1 for r in kept if r.get('already_neutralized'))} 条只作标注、免补中性IC；"
        f"未中性化 {sum(1 for r in kept if not r.get('already_neutralized'))} 条已补行业/市值/双中性IC；"
        f"本周内部相关性见下方。"
    )
    weekly["enrichment_summary"] = {
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "n_already_neutralized": sum(1 for r in kept if r.get("already_neutralized")),
        "n_neu_metrics_ok": sum(1 for r in kept if r.get("neu_metrics_ok")),
        "n_with_corr_panel": (weekly.get("weekly_corr") or {}).get("n_factors"),
    }
    weekly["generated_at"] = datetime.now(timezone.utc).isoformat()
    weekly_path.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")

    args_ns = argparse.Namespace(
        work_dir=work,
        html=work / "reports/factor_rankic_screening_index.html",
        threshold=float(args.threshold),
    )
    patch_screening_html(args_ns, payload=weekly)
    print("done", weekly["enrichment_summary"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
