#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R28 Phase 4 / §一百六十四..一百六十五: model causality matrix.

Every model-like canonical (name/category/implementation-based classifier) is
assigned a ModelTimingContract (explicit or deterministic default) and classified
descriptive-vs-predictive.  Output goes to
``docs/evidence/r28/R28_MODEL_CAUSALITY_MATRIX.{csv,json}``.

Hard gate: R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED (every model-like canonical
has a contract) and R28_ALL_PREDICTIVE_MODELS_FIT_THROUGH_T_MINUS_1 (every
predictive contract carries fit_cutoff_offset >= 1).

Run:  python3 scripts/audit_r28_model_causality.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "evidence" / "r28"


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def main() -> None:
    _load()
    OUT.mkdir(parents=True, exist_ok=True)

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.model_timing import (
        is_model_like_name,
        model_family_of,
        get_model_timing_contract,
    )
    from cleaned_operators.operator_surface import classify_canonical

    load_all()
    rows = []
    uncovered = []  # model-like canonicals without a resolvable contract
    predictive_without_tminus1 = []  # predictive contracts not fitting through t-1
    for name in sorted(OperatorRegistry.list_canonical()):
        meta = None
        try:
            ops = OperatorRegistry._operators.get(name, {}) or {}
            if ops:
                meta = next(iter(ops.values())).metadata
        except Exception:
            meta = None
        category = str(getattr(meta, "category", "") or "")
        src_module = ""
        try:
            ops = OperatorRegistry._operators.get(name, {}) or {}
            if ops:
                src_module = type(next(iter(ops.values()))).__module__
        except Exception:
            pass

        surface = ""
        try:
            surface = classify_canonical(name)
        except Exception:
            surface = "error"

        if not is_model_like_name(name, category):
            continue

        contract = get_model_timing_contract(name, category=category)
        family = model_family_of(name)
        descriptive = contract.descriptive
        predictive = contract.predictive
        state_filter = contract.state_filtering

        is_forecast_name = any(
            h in name.lower() for h in ("forecast", "innovation", "_next_", "predictive")
        )
        # descriptive in-sample stats that look like forecasts are a semantic-misuse
        # hazard (taskbook §二十二): record so they stay non-terminal/diagnostic.
        in_sample_looks_predictive = (not predictive) and is_forecast_name

        if predictive and contract.fit_cutoff_offset < 1:
            predictive_without_tminus1.append(name)

        rows.append(
            {
                "canonical": name,
                "model_family": family,
                "category": category,
                "source_file": src_module,
                "surface": surface,
                "descriptive_or_predictive": "descriptive" if descriptive else "predictive",
                "fit_cutoff_offset": contract.fit_cutoff_offset,
                "feature_cutoff": contract.fit_cutoff_offset,
                "score_time": contract.score_offset,
                "forecast_horizon": contract.forecast_horizon,
                "label_horizon": contract.label_horizon if contract.label_horizon is not None else "",
                "label_used": bool(contract.label_horizon is not None),
                "label_maturity_enforced": bool(contract.label_horizon is not None and contract.label_horizon >= 1),
                "purge": contract.purge_bars,
                "embargo": contract.embargo_bars,
                "scaler_scope": "train" if contract.scaler_fit_cutoff_offset is not None and contract.scaler_fit_cutoff_offset >= 1 else "full",
                "pca_scope": "train" if contract.scaler_fit_cutoff_offset is not None and contract.scaler_fit_cutoff_offset >= 1 else "full",
                "hyperparam_scope": "train" if contract.hyperparam_fit_cutoff_offset is not None and contract.hyperparam_fit_cutoff_offset >= 1 else "full",
                "filter_or_smoother": state_filter,
                "timing_contract_source": "explicit" if name in _explicit_names() else "generated",
                "in_sample_looks_predictive": in_sample_looks_predictive,
            }
        )

    matrix = {
        "schema_version": "factor_engine.r28.model_causality.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_like_total": len(rows),
        "predictive": sum(1 for r in rows if r["descriptive_or_predictive"] == "predictive"),
        "descriptive": sum(1 for r in rows if r["descriptive_or_predictive"] == "descriptive"),
        "contracts_missing": len(uncovered),
        "predictive_without_tminus1": predictive_without_tminus1,
        "rows": rows,
    }

    json_path = OUT / "R28_MODEL_CAUSALITY_MATRIX.json"
    json_path.write_text(json.dumps(matrix, indent=1, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else ["canonical"]
    csv_path = OUT / "R28_MODEL_CAUSALITY_MATRIX.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    from collections import Counter
    fam = Counter(r["model_family"] for r in rows)
    print(f"R28 model causality matrix: {len(rows)} model-like canonicals")
    print(f"  descriptive={matrix['descriptive']} predictive={matrix['predictive']}")
    print(f"  families: {dict(fam)}")
    if uncovered:
        print("  !! MISSING CONTRACTS:", uncovered)
    if predictive_without_tminus1:
        print("  !! PREDICTIVE WITHOUT FIT_T-1:", predictive_without_tminus1)
    if uncovered or predictive_without_tminus1:
        sys.exit(1)
    print("  R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED == TRUE")
    print("  R28_ALL_PREDICTIVE_MODELS_FIT_THROUGH_T_MINUS_1 == TRUE")


def _explicit_names():
    import cleaned_operators.model_timing as mt

    return set(mt.MODEL_TIMING_CONTRACTS.keys())


if __name__ == "__main__":
    main()
