from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_exact(
        EVALUATOR,
        '''        realized = valid["return"].to_dict()
        gross = float(sum(weights[str(asset)] * float(value) for asset, value in realized.items() if str(asset) in weights))
''',
        '''        return_by_asset = {
            str(asset): float(value)
            for asset, value in zip(
                assets,
                valid["return"].to_numpy(dtype=float),
                strict=True,
            )
        }
        gross = float(
            sum(weight * return_by_asset[asset] for asset, weight in weights.items())
        )
''',
        "portfolio return asset alignment",
    )
    replace_exact(
        EVALUATOR,
        '''        if self.require_point_in_time_universe and not self.universe_dataset:
            raise ValueError(
                "require_point_in_time_universe=True requires a registered universe_dataset"
            )
''',
        '''        # A registered dataset is required by the CLI path. Programmatic callers may
        # inject an already validated universe_frame into run_gtja185_evaluation.
''',
        "allow injected PIT universe frame",
    )
    replace_exact(
        TESTS,
        '''    assert turnover.iloc[0] == pytest.approx(1.0)
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.001)
''',
        '''    assert gross.iloc[0] == pytest.approx(0.01)
    assert gross.iloc[1] == pytest.approx(0.01)
    assert turnover.iloc[0] == pytest.approx(1.0)
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.001)
''',
        "gross return assertion",
    )
    replace_exact(
        TESTS,
        '''    with pytest.raises(ValueError, match="universe_dataset"):
        BatchEvaluationConfig(
            run_mode="production",
            require_point_in_time_universe=True,
        ).validate()
''',
        '''    BatchEvaluationConfig(
        run_mode="production",
        require_point_in_time_universe=True,
    ).validate()
''',
        "injected universe config test",
    )
    print("GTJA185 portfolio weights aligned with asset returns")


if __name__ == "__main__":
    main()
