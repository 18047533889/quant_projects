#!/usr/bin/env python3
"""Generate field/operator/family/horizon coverage for the cold-start library."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
try:
    from _bootstrap import REPO_ROOT
except ImportError:
    REPO_ROOT=Path(__file__).resolve().parents[2]
from factor_cold_start.catalog import load_catalog
from factor_cold_start.generator import EXCLUDED_OPERATOR_REASONS,existing_formula_hashes
PACKAGE_ROOT=Path(__file__).resolve().parents[1]

def _operator_targets(market:str,surface:str):
    # Dynamic reviewed extensions update the surface during registry bootstrap;
    # always freeze the real runtime before calculating coverage.
    from cleaned_operators import load_all
    load_all()
    from cleaned_operators.operator_surface import DAILY_CANONICALS,EXTENDED_ONLY_CANONICALS
    source=set(DAILY_CANONICALS if surface=="daily" else EXTENDED_ONLY_CANONICALS);excluded={name:reason for name,reason in EXCLUDED_OPERATOR_REASONS.items() if name in source}
    if market=="us" and surface=="extended":excluded["real_turnover_rate"]="requires a verified free-float share-count field; current US contract does not guarantee one"
    return source-set(excluded),excluded

def build_report(repo_root:Path=REPO_ROOT)->dict[str,Any]:
    catalogs={};total=0;all_ids=set()
    for market in ("ashare","us"):
        for surface in ("daily","extended"):
            rows=load_catalog(market,surface);key=f"{market}_{surface}";total+=len(rows);all_ids.update(r.factor_id for r in rows);operators=Counter(op for r in rows for op in r.operators);fields=Counter(f for r in rows for f in r.required_fields);families=Counter(r.family for r in rows);tiers=Counter(r.availability_tier for r in rows);complexities=Counter(r.complexity for r in rows);horizon=Counter("none" if r.horizon is None else "1-10" if r.horizon<=10 else "11-60" if r.horizon<=60 else "61-252" for r in rows);targets,excluded=_operator_targets(market,surface);covered=targets&set(operators)
            catalogs[key]={"factor_count":len(rows),"family_count":len(families),"families":dict(sorted(families.items())),"availability_tiers":dict(sorted(tiers.items())),"complexities":dict(sorted(complexities.items())),"horizon_buckets":dict(sorted(horizon.items())),"field_count":len(fields),"fields":dict(sorted(fields.items())),"operator_count_all":len(operators),"operators":dict(sorted(operators.items())),"eligible_operator_count":len(targets),"eligible_operator_covered":len(covered),"eligible_operator_coverage_pct":round(100*len(covered)/max(1,len(targets)),2),"missing_eligible_operators":sorted(targets-covered),"excluded_operators":excluded}
    return {"schema_version":"factor_cold_start.coverage.v1","summary":{"factor_count":total,"catalog_count":len(catalogs),"unique_factor_ids":len(all_ids),"existing_gtja_week2_formula_keys":len(existing_formula_hashes(repo_root))},"catalogs":catalogs}
def _markdown(report):
    lines=["# Factor cold-start coverage","",f"- Total factors: **{report['summary']['factor_count']}**",f"- Catalogs: **{report['summary']['catalog_count']}**",f"- Existing GTJA/Week2 formulas excluded structurally: **{report['summary']['existing_gtja_week2_formula_keys']}**","","## Catalog summary","","| Catalog | Factors | Families | Fields | Eligible operators | Coverage |","|---|---:|---:|---:|---:|---:|"]
    for key,data in report["catalogs"].items():lines.append(f"| `{key}` | {data['factor_count']} | {data['family_count']} | {data['field_count']} | {data['eligible_operator_covered']}/{data['eligible_operator_count']} | {data['eligible_operator_coverage_pct']}% |")
    for key,data in report["catalogs"].items():
        lines.extend(["",f"## `{key}`","","### Families","","| Family | Count |","|---|---:|"])
        for family,count in sorted(data["families"].items(),key=lambda x:(-x[1],x[0])):lines.append(f"| `{family}` | {count} |")
        if data["missing_eligible_operators"]:lines.extend(["","### Missing eligible operators","",", ".join(f"`{x}`" for x in data["missing_eligible_operators"])])
        lines.extend(["","### Intentionally excluded operators",""])
        for op,reason in sorted(data["excluded_operators"].items()):lines.append(f"- `{op}`: {reason}")
    return "\n".join(lines)+"\n"
def write_reports(repo_root:Path=REPO_ROOT):
    report=build_report(repo_root);out=PACKAGE_ROOT/"reports";out.mkdir(parents=True,exist_ok=True);(out/"coverage.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(out/"coverage.md").write_text(_markdown(report),encoding="utf-8");return report
def main():
    report=write_reports(REPO_ROOT);print(json.dumps(report["summary"],ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
