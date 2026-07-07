#!/usr/bin/env python3
"""生成 disk.v1 candidate_pool：config.json + 每因子 manifest.json。"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.dsl_validate import validate_formula  # noqa: E402

CAMPAIGN_ID = "manual_ashare_week2_pv_202606301800"
UNIVERSE_ID = "A_SHARE_ALL_A_EX_ST"
FREQUENCY_BUCKET = "daily"
MINED_BY = "sunhaiwei"
BORN_BASE = datetime(2026, 6, 30, 18, 0, 0, tzinfo=timezone.utc)


def candidate_hash(formula: str) -> str:
    normalized = " ".join(formula.split())
    payload = f"{normalized}|{UNIVERSE_ID}|{FREQUENCY_BUCKET}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def load_catalog() -> list[dict]:
    path = PACKAGE_ROOT / "source" / "week2_factors_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["factors"]


def build_campaign(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (PACKAGE_ROOT / "candidate_pool")
    factors = load_catalog()
    campaign_dir = out_dir / CAMPAIGN_ID

    if campaign_dir.exists():
        for stale in campaign_dir.rglob("manifest 2.json"):
            stale.unlink()
        for child in campaign_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    config = {
        "schema_version": "disk.v1",
        "campaign_id": CAMPAIGN_ID,
        "generator_name": "manual",
        "generator_version": "1.0.0",
        "market": "ashare",
        "universe_id": UNIVERSE_ID,
        "signal_structure": "cross_sectional",
        "asset_class": "equity",
        "frequency_bucket": FREQUENCY_BUCKET,
        "domain_root": "price_volume",
        "domain": "price_volume",
        "mining_scope": {
            "primary_tables": ["StockDailyBar", "StockCapitalDaily"],
            "auxiliary_tables": ["StockList", "StockStatus", "StockIndustry", "Calendar"],
            "forbidden_tables": ["StockBalance", "StockIncome", "StockCashFlow"],
        },
        "operator_policy": "afv_us_pv_daily",
        "data_source": {
            "local": "data/a_share/lqtp_data/",
            "cos": "cos://qs-cold/clean_data/ashare/lqtp_data/",
        },
        "mined_by": MINED_BY,
        "created_at": "2026-06-30T18:00:00Z",
        "mining_config": {
            "train_period": ["2014-01-01", "2021-12-31"],
            "valid_period": ["2022-01-01", "2023-12-31"],
            "test_period": ["2024-01-01", "2026-06-25"],
        },
        "mining_run_stats": {
            "llm_model": "manual-week2-a-pv-port",
            "llm_provider": "manual",
            "llm_base_url": "",
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "started_at": "2026-06-30T18:00:00Z",
            "finished_at": "2026-06-30T18:10:00Z",
            "duration_seconds": 600,
            "candidates_generated": len(factors),
            "candidates_submitted": len(factors),
        },
    }

    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    submitted = 0
    for seq, item in enumerate(factors, start=1):
        formula = item["formula"].strip()
        ok, err = validate_formula(formula)
        if not ok:
            raise RuntimeError(f"{item['id']} invalid: {err}")

        ts = BORN_BASE.replace(second=min(seq, 59))
        h8 = candidate_hash(formula)
        candidate_id = f"manual_{ts.strftime('%Y%m%d%H%M%S')}_{h8}"

        tier = item.get("tier", "qualified")
        rank_ic = item.get("rank_ic_pct")
        ic_pct = item.get("ic_pct")
        icir = item.get("icir")
        rank_icir = item.get("rank_icir")
        src = item.get("source_pdf", "Week2汇报.pdf")
        metrics_part = f"Rank IC {rank_ic}%"
        if ic_pct is not None:
            metrics_part = f"IC {ic_pct}% / {metrics_part}"
        if icir is not None:
            metrics_part += f" / ICIR {icir}"
        if rank_icir is not None:
            metrics_part += f" / Rank ICIR {rank_icir}"
        manifest = {
            "schema_version": "disk.v1",
            "candidate_id": candidate_id,
            "campaign_id": CAMPAIGN_ID,
            "formula": formula,
            "expression_type": "dsl",
            "generator_name": "manual",
            "generator_version": "1.0.0",
            "mined_by": MINED_BY,
            "market": "ashare",
            "universe_id": UNIVERSE_ID,
            "domain_root": "price_volume",
            "domain": "price_volume",
            "frequency_bucket": FREQUENCY_BUCKET,
            "signal_structure": "cross_sectional",
            "asset_class": "equity",
            "born_timestamp": ts.isoformat().replace("+00:00", "Z"),
            "metrics": {
                "train": {"ic": None, "icir": None, "rank_ic": None},
                "valid": {"ic": None, "icir": None, "rank_ic": None},
                "test": {"ic": None, "icir": None, "rank_ic": None},
            },
            "description": (
                f"A股价量因子 {item['id']} {item['name_zh']}（{item['cluster']}，{tier}，"
                f"来源 {src}，{metrics_part}）。{item.get('note', '')}"
            ),
        }

        cand_dir = campaign_dir / candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        submitted += 1

    print(f"campaign -> {campaign_dir}")
    print(f"manifests: {submitted}")
    return campaign_dir


def main() -> int:
    build_campaign()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
