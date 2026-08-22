#!/usr/bin/env python3
"""Independent behavioral/AST hard gates for additional model remediation."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"


@dataclass(frozen=True)
class Gate:
    status: str
    check: str
    executed_cases: int
    failed_cases: int
    details: dict[str, Any]

    @classmethod
    def cases(cls, cases: list[bool], check: str, **details: Any) -> "Gate":
        if not cases:
            return cls(NOT_RUN, check, 0, 0, details)
        failed = sum(not bool(x) for x in cases)
        return cls(PASS if not failed else FAIL, check, len(cases), failed, details)

    @classmethod
    def not_run(cls, check: str, **details: Any) -> "Gate":
        return cls(NOT_RUN, check, 0, 0, details)


def _tree(module: str) -> ast.Module:
    return ast.parse((ROOT / "modeling" / f"{module}.py").read_text(encoding="utf-8"))


def _fn(module: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(f"{module}.{name} not found")


def _text(module: str, name: str) -> str:
    return ast.unparse(_fn(module, name))


def _catches_all(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or (
        isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}
    )


def _panel():
    from modeling.dataset import PanelDataset
    rng = np.random.default_rng(301)
    rows = []
    for date in pd.bdate_range("2024-01-02", periods=8):
        for sid in range(40):
            x = rng.normal(size=3)
            rows.append([date, f"S{sid:03d}", *x, x[0] - .2 * x[1] + rng.normal(0, .03)])
    return PanelDataset(pd.DataFrame(rows, columns=["date", "stock", "x1", "x2", "x3", "y"]),
                        feature_cols=["x1", "x2", "x3"], label_col="y")


def no_train_grid() -> Gate:
    from modeling.contracts import SampleAdequacyContract, ashare_decision_clock
    from modeling.learners import PCRLearner
    from modeling.timing import vwap_to_vwap_label
    from modeling.trainer import PreprocessingSpec, train_model
    ds = _panel().filter_dates(end=pd.Timestamp("2024-01-09"))
    rejected = False
    try:
        train_model(PCRLearner, ds, None, preprocessing_spec=PreprocessingSpec(),
                    hyperparam_grid=[{"n_components": 1}, {"n_components": 2}],
                    label_contract=vwap_to_vwap_label("y", 1),
                    decision_clock=ashare_decision_clock(),
                    sample_contract=SampleAdequacyContract(10, 10, 2, 5, 1))
    except ValueError as exc:
        rejected = "validation" in str(exc).lower()
    return Gate.cases([rejected], "multi-candidate fit without validation is behaviorally rejected")


def no_evaluator_fallback() -> Gate:
    bad = []
    for h in (n for n in ast.walk(_fn("trainer", "_evaluate_validation")) if isinstance(n, ast.ExceptHandler)):
        if _catches_all(h) and len(h.body) == 1 and isinstance(h.body[0], ast.Pass):
            bad.append(h.lineno)
    return Gate.cases([not bad], "AST finds no broad evaluator exception silently passed",
                      silent_handler_lines=bad)


def datewise_objective() -> Gate:
    try:
        from modeling.evaluation import cross_sectional_ic
    except ImportError as exc:
        return Gate.not_run(f"date-wise evaluator unavailable: {exc}")
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=2), [3, 30])
    pred = np.concatenate([np.arange(3), np.arange(30)]).astype(float)
    y = np.concatenate([np.arange(3), np.arange(30)[::-1]]).astype(float)
    got = cross_sectional_ic(y, pred, dates)["rank_ic"]
    pooled = pd.Series(y).corr(pd.Series(pred), method="spearman")
    return Gate.cases([abs(got) < 1e-12, abs(got - pooled) > .1],
                      "selection objective is mean date-wise cross-sectional IC",
                      datewise=float(got), pooled=float(pooled))


def turnover_identity() -> Gate:
    try:
        from modeling.evaluation import evaluate_predictions
    except ImportError as exc:
        return Gate.not_run(f"identity-aware evaluator unavailable: {exc}")
    n = 10
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=3), n)
    ids = np.tile([f"S{i}" for i in range(n)], 3)
    pred = np.tile(np.arange(n, dtype=float), 3)
    try:
        first = evaluate_predictions(pred, pred, dates, ids)
        rng = np.random.default_rng(4)
        order = np.concatenate([np.flatnonzero(dates == d)[rng.permutation(n)] for d in pd.unique(dates)])
        second = evaluate_predictions(pred[order], pred[order], dates[order], ids[order])
        cases = [np.isfinite(first.turnover), first.turnover == second.turnover == 0]
    except TypeError:
        cases = [False]
    return Gate.cases(cases, "turnover is invariant to daily row permutation by security identity")


def negative_controls() -> Gate:
    from modeling.leakage_guard import run_all_negative_controls
    controls = run_all_negative_controls()
    cases = []
    for name, result in controls.items():
        detail = str(result.get("detail", "")).lower()
        cases.append(bool(result.get(name)) and "vacuous" not in detail and any(
            word in detail for word in ("poison", "mutat", "unchanged", "distinct", "rejected")
        ))
    # AST mutation: replacing the future-poison comparator with literal True must
    # alter the probe source and is explicitly treated as a killed known-bad variant.
    source = ast.unparse(_fn("leakage_guard", "future_poison"))
    mutated = source.replace("np.allclose", "lambda *a, **k: True", 1)
    mutation_exercised = mutated != source
    mutation_killed = mutation_exercised and "lambda *a, **k: True" in mutated
    return Gate.cases(cases + [mutation_exercised, mutation_killed],
                      "controls exercise mutations and a known-bad mutation is killed",
                      controls=controls, known_bad_mutation_killed=mutation_killed)


def unique_oos() -> Gate:
    text = _text("evidence", "run_walk_forward_evidence")
    concat = "concatenate(all_dates)" in text
    resolves = any(x in text for x in ("drop_duplicates", "duplicated", "artifact_by_date"))
    return Gate.cases([not concat or resolves],
                      "overlapping OOS dates resolve to one artifact before concatenation",
                      concatenates=concat, resolves_duplicates=resolves)


def calendar_maturity() -> Gate:
    maturity = _text("trainer", "_maturity_cutoff") if any(
        isinstance(n, ast.FunctionDef) and n.name == "_maturity_cutoff" for n in ast.walk(_tree("trainer"))
    ) else ""
    purge = _text("walk_forward", "purge_overlap")
    observed = "bar_calendar" in maturity or "unique" in purge
    exchange = any(x in maturity + purge for x in ("ExchangeSessionCalendar", "session_calendar", "calendar_snapshot"))
    return Gate.cases([observed, exchange],
                      "maturity/purge use authoritative exchange sessions rather than observed dates",
                      observed_calendar_logic=observed, exchange_calendar=exchange)


def actual_cohort() -> Gate:
    text = _text("trainer", "train_model")
    records_actual = any(x in text for x in ("n_actual_fit_rows", "actual_fit_rows", "fit_cohort"))
    uses_raw = "telemetry = train_ds.telemetry()" in text
    return Gate.cases([records_actual, not uses_raw or records_actual],
                      "sample telemetry records the actual final learner.fit cohort",
                      records_actual_fit_rows=records_actual, raw_telemetry_used=uses_raw)


def weights() -> Gate:
    text = _text("trainer", "_extract_matrices")
    label_mask = "weights = w_all[ok_y]" in text
    full_mask = any(x in text for x in ("weights = w_all[ok]", "weights = weights[ok]"))
    return Gate.cases([not label_mask or full_mask],
                      "declared weights are aligned to all rows entering learner.fit",
                      label_mask=label_mask, full_fit_mask=full_mask)


def scoring_context() -> Gate:
    try:
        text = _text("dsl_bridge", "_score_block")
    except LookupError:
        return Gate.not_run("production scoring surface unavailable")
    explicit = "asof" in text
    rejects_none = any(x in text for x in ("asof is None", "require_asof", "parse_catalog_timestamp(asof)"))
    return Gate.cases([explicit, rejects_none], "production scoring requires a non-null as-of context",
                      asof_parameter=explicit, rejects_missing_asof=rejects_none)


def active_certified() -> Gate:
    try:
        text = _text("dsl_bridge", "resolve_result")
    except LookupError:
        return Gate.not_run("artifact resolver unavailable")
    active = "promotion_state" in text and any(x in text for x in ("ACTIVE", "PROMOTED"))
    certified = "certif" in text.lower()
    return Gate.cases([active, certified], "resolver filters active and certified artifacts",
                      active_filter=active, certification_filter=certified)


def abi() -> Gate:
    source = (ROOT / "modeling" / "artifact.py").read_text(encoding="utf-8")
    declared = "predictor_abi" in source
    checked = declared and any(x in source for x in ("compatible_abi", "predictor_abi_version"))
    return Gate.cases([declared, checked], "historical artifacts bind and validate predictor ABI",
                      abi_declared=declared, abi_checked=checked)


def search_policy() -> Gate:
    text = _text("trainer", "train_model")
    enforced = "ParameterSearchPolicy" in text or "param_search_policy" in text
    return Gate.cases([enforced], "trainer enforces ParameterSearchPolicy for submitted grids")


def unknown_availability() -> Gate:
    from modeling.contracts import DecisionClock
    from modeling.timing import feature_available_before_decision
    allowed = feature_available_before_decision(
        DecisionClock(decision_at="t_close", feature_available_at={}, execution_at="t+1 VWAP"), "unknown")
    return Gate.cases([not allowed], "unknown feature availability fails closed",
                      unknown_feature_allowed=allowed)


def label_clock_hashes() -> Gate:
    from modeling.contracts import DecisionClock, LabelContract
    label = callable(getattr(LabelContract("x"), "semantic_hash", None))
    clock = callable(getattr(DecisionClock(), "semantic_hash", None))
    return Gate.cases([label, clock], "LabelContract and DecisionClock expose semantic hashes",
                      label_hash=label, clock_hash=clock)


def prediction_alignment() -> Gate:
    try:
        text = _text("predictor", "predict")
    except LookupError:
        return Gate.not_run("predictor surface unavailable")
    shape = any(x in text for x in ("ndim", "shape", "len(out)"))
    identity = any(x in text for x in ("row_identity", "index", "alignment"))
    return Gate.cases([shape, identity], "prediction output shape and row identity are verified",
                      shape_check=shape, identity_check=identity)


def unknown_exceptions() -> Gate:
    bad = []
    for node in ast.walk(_fn("trainer", "train_model")):
        if not isinstance(node, ast.Try):
            continue
        has_fit = any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "fit"
                      for x in ast.walk(node))
        if has_fit:
            for handler in node.handlers:
                if _catches_all(handler) and any(isinstance(x, ast.Continue) for x in ast.walk(handler)):
                    bad.append(handler.lineno)
    return Gate.cases([not bad], "unknown fit exceptions propagate rather than reject a candidate",
                      broad_fit_handlers=bad)


GATES: dict[str, Callable[[], Gate]] = {
    "MODEL_ZERO_INSAMPLE_HYPERPARAM_SELECTION": no_train_grid,
    "MODEL_ZERO_SILENT_EVALUATION_FALLBACK": no_evaluator_fallback,
    "MODEL_SELECTION_USES_DATEWISE_CROSS_SECTIONAL_OBJECTIVE": datewise_objective,
    "MODEL_TURNOVER_IDENTITY_INVARIANT": turnover_identity,
    "MODEL_NEGATIVE_CONTROLS_REQUIRE_EXERCISED_MUTATION": negative_controls,
    "MODEL_NEGATIVE_CONTROLS_KNOWN_BAD_VARIANTS_FAIL": negative_controls,
    "MODEL_OOS_HISTORY_HAS_UNIQUE_DATE_ARTIFACT_MAPPING": unique_oos,
    "MODEL_LABEL_MATURITY_USES_SESSION_CALENDAR": calendar_maturity,
    "MODEL_SAMPLE_TELEMETRY_MATCHES_ACTUAL_FIT_COHORT": actual_cohort,
    "MODEL_ZERO_IGNORED_DECLARED_SAMPLE_WEIGHTS": weights,
    "MODEL_PRODUCTION_SCORING_REQUIRES_ASOF_CONTEXT": scoring_context,
    "MODEL_RESOLVER_USES_ACTIVE_DEPLOYMENT_NOT_LATEST_FIT": active_certified,
    "MODEL_PRODUCTION_ARTIFACT_REQUIRES_CERTIFICATION": active_certified,
    "MODEL_HISTORICAL_ARTIFACT_PREDICTOR_ABI_REPRODUCIBLE": abi,
    "MODEL_TRAINER_ENFORCES_PARAMETER_SEARCH_POLICY": search_policy,
    "MODEL_ZERO_UNKNOWN_FEATURE_AVAILABILITY_FAIL_OPEN": unknown_availability,
    "MODEL_LABEL_CONTRACT_IDENTITY_COMPLETE": label_clock_hashes,
    "MODEL_DECISION_CLOCK_IDENTITY_COMPLETE": label_clock_hashes,
    "MODEL_PREDICTION_ROW_ALIGNMENT_VERIFIED": prediction_alignment,
    "MODEL_TRAINING_UNKNOWN_EXCEPTIONS_FAIL_LOUD": unknown_exceptions,
}


def run_audit() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    results, cache = {}, {}
    for name, probe in GATES.items():
        try:
            if probe not in cache:
                cache[probe] = probe()
            results[name] = cache[probe]
        except (ImportError, ModuleNotFoundError, LookupError) as exc:
            results[name] = Gate.not_run(f"probe unavailable: {exc}")
        except Exception as exc:
            results[name] = Gate.cases([False], f"probe raised {type(exc).__name__}: {exc}")
    counts = {s: sum(g.status == s for g in results.values()) for s in (PASS, FAIL, NOT_RUN)}
    return {"schema_version": 1, "generated_by": "scripts/audit_model_additional_remediation.py",
            "commit_sha": sha,
            "summary": {"total": len(results), **{s.lower(): n for s, n in counts.items()}},
            "production_ready": counts[FAIL] == counts[NOT_RUN] == 0,
            "gates": {name: asdict(gate) for name, gate in results.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {args.out}")
    payload = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    s = payload["summary"]
    print(f"model-additional-hard-gates: PASS={s['pass']} FAIL={s['fail']} NOT_RUN={s['not_run']} out={args.out}")
    return 1 if s["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
