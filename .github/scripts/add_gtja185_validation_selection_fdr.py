from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"
DOCS = PROJECT / "docs" / "gtja185_batch_evaluation.md"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_evaluator() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    text = replace_exact(
        text,
        '''    cost_bps: float = 10.0
    universe_dataset: str | None = None
''',
        '''    cost_bps: float = 10.0
    fdr_alpha: float = 0.10
    universe_dataset: str | None = None
''',
        "FDR config field",
    )
    text = replace_exact(
        text,
        '''        if not math.isfinite(float(self.cost_bps)) or self.cost_bps < 0:
            raise ValueError("cost_bps must be a finite non-negative number")
''',
        '''        if not math.isfinite(float(self.cost_bps)) or self.cost_bps < 0:
            raise ValueError("cost_bps must be a finite non-negative number")
        if not 0 < float(self.fdr_alpha) <= 1:
            raise ValueError("fdr_alpha must be in (0, 1]")
''',
        "FDR config validation",
    )
    text = replace_exact(
        text,
        '''    hac_t_stat = mean / hac_se if math.isfinite(hac_se) and hac_se > 0 else float("nan")
    return {
''',
        '''    hac_t_stat = mean / hac_se if math.isfinite(hac_se) and hac_se > 0 else float("nan")
    hac_p_value = (
        math.erfc(abs(hac_t_stat) / math.sqrt(2.0))
        if math.isfinite(hac_t_stat)
        else float("nan")
    )
    return {
''',
        "HAC p-value calculation",
    )
    text = replace_exact(
        text,
        '''        "hac_t_stat": hac_t_stat if math.isfinite(hac_t_stat) else None,
        "positive_ratio": float((clean > 0).mean()),
''',
        '''        "hac_t_stat": hac_t_stat if math.isfinite(hac_t_stat) else None,
        "hac_p_value": hac_p_value if math.isfinite(hac_p_value) else None,
        "positive_ratio": float((clean > 0).mean()),
''',
        "HAC p-value result",
    )
    text = replace_exact(
        text,
        '''            "hac_t_stat": None,
            "positive_ratio": None,
''',
        '''            "hac_t_stat": None,
            "hac_p_value": None,
            "positive_ratio": None,
''',
        "empty HAC p-value result",
    )
    text = replace_exact(
        text,
        '''def _route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    test = primary.get("test", {})
    rank_ic = ((test.get("rank_ic") or {}).get("mean"))
    icir = ((test.get("rank_ic") or {}).get("ir"))
    sharpe = ((test.get("long_short_net") or {}).get("sharpe"))
    positive = ((test.get("rank_ic") or {}).get("positive_ratio"))
''',
        '''def _route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    """Route using validation only; test remains a locked final holdout."""
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    validation = primary.get("valid", {})
    rank_ic = ((validation.get("rank_ic") or {}).get("mean"))
    icir = ((validation.get("rank_ic") or {}).get("ir"))
    sharpe = ((validation.get("long_short_net") or {}).get("sharpe"))
    positive = ((validation.get("rank_ic") or {}).get("positive_ratio"))
''',
        "validation-only routing",
    )

    route_anchor = '''def _ranking_rows(records: Sequence[FactorRunRecord]) -> list[dict[str, Any]]:
'''
    fdr_helpers = '''def _primary_horizon_metrics(record: FactorRunRecord) -> Mapping[str, Any]:
    return record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}


def _apply_validation_fdr(
    records: Sequence[FactorRunRecord],
    *,
    alpha: float,
) -> None:
    """Benjamini-Hochberg correction over validation RankIC HAC p-values."""
    candidates: list[tuple[int, float]] = []
    for index, record in enumerate(records):
        if record.status != "success":
            continue
        validation = _primary_horizon_metrics(record).get("valid", {})
        p_value = ((validation.get("rank_ic") or {}).get("hac_p_value"))
        if p_value is None:
            continue
        p_float = float(p_value)
        if math.isfinite(p_float) and 0 <= p_float <= 1:
            candidates.append((index, p_float))
    ordered = sorted(candidates, key=lambda item: item[1])
    m = len(ordered)
    q_values: dict[int, float] = {}
    running = 1.0
    for reverse_rank, (record_index, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - reverse_rank + 1
        adjusted = min(1.0, p_value * m / max(rank, 1))
        running = min(running, adjusted)
        q_values[record_index] = running
    for index, record in enumerate(records):
        q_value = q_values.get(index)
        record.metrics["multiple_testing"] = {
            "method": "benjamini_hochberg",
            "family_size": m,
            "alpha": float(alpha),
            "validation_rank_ic_q_value": q_value,
            "passed": q_value is not None and q_value <= alpha,
        }
        if (
            record.status == "success"
            and record.route in {"tier3a_core", "tier3b_satellite"}
            and (q_value is None or q_value > alpha)
        ):
            record.route = "tier2_research"


'''
    if route_anchor not in text:
        raise RuntimeError("ranking anchor not found")
    text = text.replace(route_anchor, fdr_helpers + route_anchor, 1)

    text = replace_exact(
        text,
        '''        primary = record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}
        test = primary.get("test", {})
''',
        '''        primary = _primary_horizon_metrics(record)
        validation = primary.get("valid", {})
        test = primary.get("test", {})
        multiple_testing = record.metrics.get("multiple_testing", {})
''',
        "ranking split metrics",
    )
    text = replace_exact(
        text,
        '''                "test_rank_ic": ((test.get("rank_ic") or {}).get("mean")),
''',
        '''                "validation_rank_ic": ((validation.get("rank_ic") or {}).get("mean")),
                "validation_rank_ic_ir": ((validation.get("rank_ic") or {}).get("ir")),
                "validation_rank_ic_hac_t": ((validation.get("rank_ic") or {}).get("hac_t_stat")),
                "validation_rank_ic_q_value": multiple_testing.get("validation_rank_ic_q_value"),
                "validation_long_short_net_sharpe": ((validation.get("long_short_net") or {}).get("sharpe")),
                "test_rank_ic": ((test.get("rank_ic") or {}).get("mean")),
''',
        "ranking validation columns",
    )
    text = replace_exact(
        text,
        '''            row.get("test_rank_ic_ir") if row.get("test_rank_ic_ir") is not None else -999,
            row.get("test_long_short_net_sharpe") if row.get("test_long_short_net_sharpe") is not None else -999,
''',
        '''            -row.get("validation_rank_ic_q_value")
            if row.get("validation_rank_ic_q_value") is not None
            else -999,
            row.get("validation_rank_ic_ir")
            if row.get("validation_rank_ic_ir") is not None
            else -999,
            row.get("validation_long_short_net_sharpe")
            if row.get("validation_long_short_net_sharpe") is not None
            else -999,
''',
        "ranking validation sort",
    )
    text = replace_exact(
        text,
        '''    ranking = _ranking_rows(records)
''',
        '''    _apply_validation_fdr(records, alpha=config.fdr_alpha)
    for record in records:
        _json_dump(
            report_dir / f"{record.factor_name}.json",
            {
                "factor_name": record.factor_name,
                "formula_hash": record.formula_hash,
                "snapshot_id": snapshot_id,
                "config_hash": config_hash,
                "record": asdict(record),
            },
        )
    ranking = _ranking_rows(records)
''',
        "apply FDR before final reports",
    )
    text = replace_exact(
        text,
        '''    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--universe-dataset", default=None)
''',
        '''    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--universe-dataset", default=None)
''',
        "FDR CLI argument",
    )
    text = replace_exact(
        text,
        '''        cost_bps=args.cost_bps,
        universe_dataset=args.universe_dataset,
''',
        '''        cost_bps=args.cost_bps,
        fdr_alpha=args.fdr_alpha,
        universe_dataset=args.universe_dataset,
''',
        "FDR CLI config",
    )
    EVALUATOR.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    text = text.replace(
        '''    _long_short_series,
''',
        '''    FactorRunRecord,
    _apply_validation_fdr,
    _long_short_series,
    _route_factor,
''',
        1,
    )
    text += '''

def _selection_metrics(valid_rank_ic: float, test_rank_ic: float) -> dict:
    def split(rank_ic: float) -> dict:
        return {
            "rank_ic": {
                "mean": rank_ic,
                "ir": 0.6,
                "positive_ratio": 0.6,
                "hac_p_value": 0.01,
            },
            "long_short_net": {"sharpe": 1.0},
        }

    return {"21": {"valid": split(valid_rank_ic), "test": split(test_rank_ic)}}


def test_routing_uses_validation_not_test():
    metrics = _selection_metrics(valid_rank_ic=0.04, test_rank_ic=-0.20)
    assert _route_factor(metrics, coverage=0.8) == "tier3a_core"


def test_benjamini_hochberg_controls_185_factor_family():
    records = [
        FactorRunRecord(
            factor_name=f"f{i}",
            formula_hash=str(i),
            status="success",
            coverage=0.8,
            metrics=_selection_metrics(0.04, 0.04),
            route="tier3a_core",
        )
        for i in range(3)
    ]
    records[0].metrics["21"]["valid"]["rank_ic"]["hac_p_value"] = 0.001
    records[1].metrics["21"]["valid"]["rank_ic"]["hac_p_value"] = 0.04
    records[2].metrics["21"]["valid"]["rank_ic"]["hac_p_value"] = 0.8
    _apply_validation_fdr(records, alpha=0.05)
    q_values = [
        record.metrics["multiple_testing"]["validation_rank_ic_q_value"]
        for record in records
    ]
    assert q_values[0] == pytest.approx(0.003)
    assert q_values[1] == pytest.approx(0.06)
    assert q_values[2] == pytest.approx(0.8)
    assert records[0].route == "tier3a_core"
    assert records[1].route == "tier2_research"
    assert records[2].route == "tier2_research"
'''
    TESTS.write_text(text, encoding="utf-8")


def patch_docs() -> None:
    text = DOCS.read_text(encoding="utf-8")
    old = '''9. Production mode requires a registered point-in-time universe dataset with explicit
   membership and optional tradability flags; missing membership fails closed.
10. Produce deterministic routing recommendations (`tier3a_core`, `tier3b_satellite`,
   `tier2_research`, `rejected`) without silently publishing a factor.
'''
    new = '''9. Production mode requires a registered point-in-time universe dataset with explicit
   membership and optional tradability flags; missing membership fails closed.
10. Route factors using **validation only**. The test sample remains a locked final holdout
    and is never used for direction, thresholds, routing or ranking order.
11. Apply Benjamini-Hochberg false-discovery-rate control across the complete validation
    family before a factor may enter core or satellite tiers.
12. Produce deterministic routing recommendations (`tier3a_core`, `tier3b_satellite`,
   `tier2_research`, `rejected`) without silently publishing a factor.
'''
    if old not in text:
        raise RuntimeError("selection documentation anchor not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        '''11. Optionally upsert factor values to `factor_lake_staging`; publication remains a separate,
''',
        '''13. Optionally upsert factor values to `factor_lake_staging`; publication remains a separate,
''',
        1,
    )
    DOCS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_evaluator()
    patch_tests()
    patch_docs()
    print("GTJA185 validation-only selection and FDR control added")


if __name__ == "__main__":
    main()
