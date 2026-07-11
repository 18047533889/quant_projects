from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    old = '''    engine, parsed, compile_timings = _compile_pack(
        selected,
        market_frame,
        backend=config.backend,
        run_mode=config.run_mode,
    )
    raw_results, execute_timings, execution_errors = _execute_pack(
        engine,
        parsed,
        batch_size=config.batch_size,
        market=config.market,
    )
    forward_returns = _price_forward_returns(
        market_frame,
        config.horizons,
        config.forward_price_field,
        entry_lag=config.entry_lag,
    )

    records: list[FactorRunRecord] = []
    purified_signals: dict[str, pd.Series] = {}
    skipped = 0
    for factor in selected:
        record_path = report_dir / f"{factor.name}.json"
        if config.resume:
            resumed = _resume_record(
                record_path,
                formula_hash=factor.formula_hash,
                snapshot_id=snapshot_id,
                config_hash=config_hash,
            )
            if resumed is not None:
                records.append(resumed)
                skipped += 1
                continue
'''
    new = '''    resumed_records: dict[str, FactorRunRecord] = {}
    pending_factors: list[FactorDefinition] = []
    for factor in selected:
        resumed = None
        if config.resume:
            resumed = _resume_record(
                report_dir / f"{factor.name}.json",
                formula_hash=factor.formula_hash,
                snapshot_id=snapshot_id,
                config_hash=config_hash,
            )
        if resumed is not None:
            resumed_records[factor.name] = resumed
        else:
            pending_factors.append(factor)

    compile_timings: dict[str, float] = {}
    execute_timings: dict[str, float] = {}
    raw_results: dict[str, pd.Series] = {}
    execution_errors: dict[str, str] = {}
    forward_returns: dict[int, pd.Series] = {}
    if pending_factors:
        engine, parsed, compile_timings = _compile_pack(
            pending_factors,
            market_frame,
            backend=config.backend,
            run_mode=config.run_mode,
        )
        raw_results, execute_timings, execution_errors = _execute_pack(
            engine,
            parsed,
            batch_size=config.batch_size,
            market=config.market,
        )
        forward_returns = _price_forward_returns(
            market_frame,
            config.horizons,
            config.forward_price_field,
            entry_lag=config.entry_lag,
        )

    records: list[FactorRunRecord] = []
    skipped = len(resumed_records)
    for factor in selected:
        record_path = report_dir / f"{factor.name}.json"
        if factor.name in resumed_records:
            records.append(resumed_records[factor.name])
            continue
'''
    text = replace_exact(text, old, new, "resume before compile")
    text = text.replace(
        '''            records.append(record)
            purified_signals[factor.name] = purified
''',
        '''            records.append(record)
''',
        1,
    )
    EVALUATOR.write_text(text, encoding="utf-8")

    test_text = TESTS.read_text(encoding="utf-8")
    test_text += '''

def test_resume_skips_factor_compile_and_execute(tmp_path, monkeypatch):
    import evaluation.gtja185_batch as batch

    frame = batch.build_synthetic_market_frame(periods=260, symbols=8, seed=99)
    config = BatchEvaluationConfig(
        horizons=(1,),
        min_assets=6,
        n_quantiles=4,
        limit=1,
        strict=True,
        resume=True,
    )
    first = batch.run_gtja185_evaluation(
        config,
        output_dir=tmp_path,
        market_frame=frame,
        snapshot_id="synthetic:resume-contract",
    )
    assert first.succeeded == 1

    def should_not_compile(*args, **kwargs):
        raise AssertionError("resume should not compile a valid cached factor")

    monkeypatch.setattr(batch, "_compile_pack", should_not_compile)
    second = batch.run_gtja185_evaluation(
        config,
        output_dir=tmp_path,
        market_frame=frame,
        snapshot_id="synthetic:resume-contract",
    )
    assert second.succeeded == 1
    assert second.skipped == 1
'''
    TESTS.write_text(test_text, encoding="utf-8")
    print("GTJA185 resume now skips valid cached factor computation")


if __name__ == "__main__":
    main()
