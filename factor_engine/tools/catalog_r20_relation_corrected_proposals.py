#!/usr/bin/env python3
"""Build semantically explicit R20 Top-10 shareholder relation proposals."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/factor_catalog_20260916"
TABLE = "StockTopTenShareholder"


def source_col(field: str, rank: int) -> str:
    return f"source_col('{TABLE}','{field}','ShareholderRank',{rank})"


def ranked(field: str) -> list[str]:
    return [source_col(field, rank) for rank in range(1, 11)]


def call(name: str, args: list[str], *scalars: object) -> str:
    values = [*args, *(repr(value) for value in scalars)]
    return f"{name}({','.join(values)})"


def add_all(values: list[str]) -> str:
    value = values[0]
    for item in values[1:]:
        value = f"add({value},{item})"
    return value


RATIOS = ranked("ShareRatio")
NUMBERS = ranked("ShareNumber")
TOP10_SUM = call("relation_topk_sum", RATIOS)
HHI = call("relation_hhi", RATIOS)
ENTROPY = call("relation_entropy", RATIOS)


def base_formula(name: str) -> tuple[str, str]:
    if "topk_concentration" in name:
        return call("relation_topk_concentration", RATIOS, 5), "前五大股东持股比例占前十大已披露持股比例合计的比例"
    if "distribution_skew" in name:
        return call("relation_distribution_skew", RATIOS), "前十大股东持股比例在十个排名槽位上的截面偏度"
    if "distribution_pearson_kurtosis" in name:
        return call("relation_distribution_pearson_kurtosis", RATIOS), "前十大股东持股比例在十个排名槽位上的 Pearson 峰度"
    if "distribution_excess_kurtosis" in name:
        return call("relation_distribution_excess_kurtosis", RATIOS), "前十大股东持股比例在十个排名槽位上的超额峰度"
    if "entropy_change" in name:
        return f"relation_entropy_change({ENTROPY},2)", "前十大已披露持股比例归一化熵在两交易日决策网格上的变化"
    if "relation_entropy_" in name:
        return ENTROPY, "前十大已披露持股比例的归一化分布熵"
    if "hhi_change" in name:
        return f"relation_hhi_change({HHI},2)", "前十大已披露持股比例 HHI 在两交易日决策网格上的变化"
    if "concentration_acceleration" in name:
        return f"relation_concentration_acceleration({HHI},2)", "前十大已披露持股比例 HHI 的两交易日二阶变化"
    if "relation_hhi_" in name:
        return HHI, "前十大已披露持股比例按观测 Top-10 合计归一化后的 HHI"
    if "topk_sum" in name:
        return TOP10_SUM, "前十大股东已披露持股比例合计"
    if "weighted_change" in name:
        changes = [f"relation_weighted_change({number},{ratio})" for number, ratio in zip(NUMBERS, RATIOS)]
        return add_all(changes), "十个排名槽位持股数量变化乘当期持股比例后的合计"
    if "weighted_std_ex_self" in name:
        return f"holder_top10_weighted_std('{TABLE}')", "前十大股东持股数量按持股比例加权的总体标准差"
    if "category_signed_contribution" in name:
        return f"relation_category_signed_contribution({TOP10_SUM},industry_code)", "个股前十大已披露持股比例合计占同日同行业绝对量合计的有符号贡献"
    if "diffusion_score" in name:
        return f"relation_diffusion_score({TOP10_SUM},industry_code,0.5,2)", "前十大已披露持股比例合计在同行业均匀关系图上的两步扩散值"
    if "rank_weighted_sum" in name:
        return call("relation_rank_weighted_sum", RATIOS), "前十大股东持股比例的逆名次加权均值"
    if "rank_mobility" in name:
        return f"holder_top10_two_day_rank_migration('{TABLE}')", "同一股东跨两交易日决策网格的持股比例加权排名迁移"
    if "share_mobility" in name:
        return call("relation_share_mobility", RATIOS, 2), "十个排名槽位持股比例在两交易日决策网格上的平均绝对变化"
    if "distinct_count" in name:
        return f"holder_top10_disclosure_count('{TABLE}')", "当前快照前十大披露集合中的不同股东数量"
    raise ValueError(f"unsupported relation family: {name}")


def interaction(name: str, base: str) -> tuple[str, str]:
    if name.endswith("_direct"):
        return base, ""
    if "_return_interaction" in name:
        rhs, note = "field('ret', table='StockDailyBarAdj')", "与当日复权收益交互"
    elif "_turnover_interaction" in name:
        rhs, note = "turnover_ratio", "与当日换手率交互"
    elif "_freefloat_interaction" in name:
        rhs, note = "safe_div_null(market_cap,free_market_cap)", "与总市值/自由流通市值比交互"
    elif "_pledge_proxy_interaction" in name:
        rhs, note = f"holder_top10_pledge_ratio('{TABLE}')", "与前十大股东质押股数/持股数比交互"
    elif "_goodwill_interaction" in name:
        rhs, note = "safe_div_null(goodwill,total_assets)", "与商誉/总资产比交互"
    else:
        raise ValueError(f"unknown interaction suffix: {name}")
    return f"multiply({base},{rhs})", note


def load_rows() -> list[dict]:
    relation_ids = {
        row["id"]
        for row in map(json.loads, (EVIDENCE / "r20_minute_relation_proposals.jsonl").read_text(encoding="utf-8").splitlines())
        if any("前十大股东" in str(change) for change in row.get("changes", []))
    }
    relation_ids.update(f"R66_{i:05d}" for i in (*range(1235, 1247), *range(1283, 1289)))
    found: dict[str, dict] = {}
    for filename in ("r20_additional_original_context.jsonl.gz", "r20_original_context.jsonl.gz"):
        with gzip.open(EVIDENCE / filename, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("id") in relation_ids:
                    found[row["id"]] = row
    missing = sorted(relation_ids - found.keys())
    if missing or len(relation_ids) != 108:
        raise RuntimeError(f"relation coverage mismatch: ids={len(relation_ids)}, missing={missing}")
    return sorted(found.values(), key=lambda row: row["source_row"])


def runtime():
    path = ROOT / "evidence/factor_catalog_20260915/compile_catalog.py"
    spec = importlib.util.spec_from_file_location("r20_relation_compile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser, engine = module.build_runtime()
    path = ROOT / "evidence/factor_catalog_20260915/smoke_catalog.py"
    spec = importlib.util.spec_from_file_location("r20_relation_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(smoke)
    return parser, engine, smoke.bind_fields


def main() -> None:
    parser_cli = argparse.ArgumentParser()
    parser_cli.add_argument("--output", default=str(EVIDENCE / "r20_relation_corrected_proposals.jsonl"))
    args = parser_cli.parse_args()
    parser, engine, bind_fields = runtime()
    from factor_engine.api.factor import Factor

    output: list[dict] = []
    failures: list[dict] = []
    for row in load_rows():
        context = row.get("original_context") or {}
        name = str(context.get("因子名称") or "")
        try:
            base, definition = base_formula(name)
            formula, interaction_note = interaction(name, base)
            if interaction_note:
                definition = f"{definition}；{interaction_note}"
            expr = parser.parse(formula)
            bindings, binding_failures = bind_fields(expr)
            if binding_failures:
                raise RuntimeError(json.dumps(binding_failures, ensure_ascii=False))
            engine.compile(Factor(name=str(row["id"]), expr=expr, source_expr=formula, surface="compat_research"))
            output.append({
                "source_row": row["source_row"],
                "id": row["id"],
                "before_formula": row["current_formula"],
                "current_formula": formula,
                "changes": list(row.get("changes") or []) + [f"语义重建（非等价）：{definition}"],
                "current_definition": definition,
                "semantic_redesign": True,
                "compile_status": "COMPILED",
                "error": "",
                "bindings": bindings,
                "original_error": row.get("error", ""),
            })
        except Exception as exc:
            failures.append({"source_row": row.get("source_row"), "id": row.get("id"), "name": name, "error": f"{type(exc).__name__}: {exc}"})
    destination = Path(args.output)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(json.dumps({"target": 108, "compiled": len(output), "failed": len(failures), "sha256": digest, "failures": failures}, ensure_ascii=False, indent=2))
    if failures or len(output) != 108:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
