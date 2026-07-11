from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVAL = AUTO / "evaluation"
GTJA = ROOT / "gtja191"


def move(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        source.replace(target)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def migrate_modules() -> None:
    move(EVAL / "gtja185_models.py", EVAL / "batch_models.py")
    move(EVAL / "gtja185_metrics.py", EVAL / "batch_metrics.py")
    move(EVAL / "gtja185_engine.py", EVAL / "batch_engine.py")
    move(EVAL / "gtja185_runner.py", EVAL / "batch_runner.py")

    models = (EVAL / "batch_models.py").read_text(encoding="utf-8")
    models = models.replace(
        '"""Stable contracts for the GTJA185 evaluation pipeline."""',
        '"""Stable contracts for generic factor-pack evaluation."""',
    )
    models = models.replace('factor_version: str = "gtja185.v1"', 'factor_version: str = "v1"')
    models = models.replace(
        '"production GTJA185 evaluation requires require_point_in_time_universe=True"',
        '"production factor-pack evaluation requires require_point_in_time_universe=True"',
    )
    write(EVAL / "batch_models.py", models)

    metrics = (EVAL / "batch_metrics.py").read_text(encoding="utf-8")
    metrics = metrics.replace(
        '"""Leakage-safe, vectorized metrics for GTJA185 evaluation."""',
        '"""Leakage-safe, vectorized metrics for generic factor-pack evaluation."""',
    )
    metrics = metrics.replace("from .gtja185_models import", "from .batch_models import")
    write(EVAL / "batch_metrics.py", metrics)

    engine = (EVAL / "batch_engine.py").read_text(encoding="utf-8")
    engine = engine.replace(
        '"""Efficient GTJA185 preflight and production-aware FactorEngine execution."""',
        '"""Efficient factor-pack preflight and production-aware FactorEngine execution."""',
    )
    engine = engine.replace(
        "from factor_packs.gtja185 import FactorDefinition",
        "from .factor_pack import FactorDefinition",
    )
    engine = engine.replace('universe="GTJA185"', 'universe=str(definition.metadata.get("universe", "FACTOR_PACK"))')
    write(EVAL / "batch_engine.py", engine)

    runner_path = EVAL / "batch_runner.py"
    runner = runner_path.read_text(encoding="utf-8")
    runner = runner.replace(
        '"""Production runner for the versioned GTJA185 factor pack."""',
        '"""Production runner for any versioned factor pack."""',
    )
    runner = runner.replace(
        "from factor_packs.gtja185 import FactorDefinition, load_gtja185_pack",
        "from .factor_pack import FactorDefinition, FactorPack",
    )
    runner = runner.replace("from .gtja185_metrics import", "from .batch_metrics import")
    runner = runner.replace("from .gtja185_models import", "from .batch_models import")
    runner = replace_once(
        runner,
        "def _compile_pack(\n    factors: Sequence[FactorDefinition],\n    frame: pd.DataFrame,\n    *,\n    backend: str,\n    run_mode: str,\n)",
        "def _compile_pack(\n    factors: Sequence[FactorDefinition],\n    frame: pd.DataFrame,\n    *,\n    backend: str,\n    run_mode: str,\n    universe: str,\n)",
        "compile signature",
    )
    runner = runner.replace('universe="GTJA185"', "universe=universe")
    runner = replace_once(
        runner,
        "def run_gtja185_evaluation(\n    config: BatchEvaluationConfig,",
        "def run_factor_pack_evaluation(\n    pack: FactorPack,\n    config: BatchEvaluationConfig,",
        "runner entry",
    )
    runner = runner.replace("    pack = load_gtja185_pack()\n", "")
    runner = replace_once(
        runner,
        "            run_mode=config.run_mode,\n        )",
        "            run_mode=config.run_mode,\n            universe=pack.name,\n        )",
        "compile call",
    )
    runner = runner.replace('f"gtja185_{started_at.strftime', 'f"{pack.name}_{started_at.strftime')
    runner = runner.replace("GTJA185 evaluation failed", "factor-pack evaluation failed")
    cli_anchor = runner.find("\ndef _parse_csv_ints")
    if cli_anchor < 0:
        raise RuntimeError("runner CLI anchor not found")
    runner = runner[:cli_anchor] + "\n"
    write(runner_path, runner)


def write_factor_contract() -> None:
    write(
        EVAL / "factor_pack.py",
        '''"""Provider-neutral factor-pack contracts used by AutoFactorEvaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    formula: str
    source_formula: str
    description: str
    formula_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FactorPack:
    name: str
    version: str
    source_catalog_hash: str
    pack_hash: str
    factors: tuple[FactorDefinition, ...]
    metadata: Mapping[str, Any]

    def names(self) -> list[str]:
        return [factor.name for factor in self.factors]

    def by_name(self) -> dict[str, FactorDefinition]:
        return {factor.name: factor for factor in self.factors}

    def select(
        self,
        names: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> tuple[FactorDefinition, ...]:
        selected = list(self.factors)
        if names is not None:
            requested = list(dict.fromkeys(str(name) for name in names))
            mapping = self.by_name()
            missing = [name for name in requested if name not in mapping]
            if missing:
                raise KeyError(f"unknown factors in pack {self.name}: {missing[:10]}")
            selected = [mapping[name] for name in requested]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            selected = selected[:limit]
        return tuple(selected)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("factor pack name must be non-empty")
        if not self.version.strip():
            raise ValueError("factor pack version must be non-empty")
        if not self.factors:
            raise ValueError(f"factor pack {self.name} is empty")
        names = self.names()
        if len(set(names)) != len(names):
            raise ValueError(f"factor pack {self.name} contains duplicate names")
        for factor in self.factors:
            if not factor.name or not factor.formula or not factor.formula_hash:
                raise ValueError(f"invalid factor definition in pack {self.name}: {factor}")


@runtime_checkable
class FactorPackProvider(Protocol):
    def __call__(self) -> FactorPack: ...
''',
    )


def write_batch_facade() -> None:
    write(
        EVAL / "batch.py",
        '''"""Public, provider-neutral batch evaluation API and CLI."""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from typing import Sequence

from .batch_models import BatchEvaluationConfig, BatchRunSummary, FactorRunRecord, SplitBoundaries
from .batch_runner import build_synthetic_market_frame, run_factor_pack_evaluation
from .factor_pack import FactorDefinition, FactorPack, FactorPackProvider


def load_factor_pack(provider: str) -> FactorPack:
    module_name, separator, attribute = provider.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("provider must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    pack = factory()
    if not isinstance(pack, FactorPack):
        raise TypeError(f"provider {provider} returned {type(pack).__name__}, expected FactorPack")
    pack.validate()
    return pack


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a versioned external factor pack")
    parser.add_argument("--provider", required=True, help="module:function returning FactorPack")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--universe-id", default="A_SHARE_ALL_A_EX_ST")
    parser.add_argument("--universe-dataset", default=None)
    parser.add_argument("--require-point-in-time-universe", action="store_true")
    parser.add_argument("--backend", default="pandas")
    parser.add_argument("--run-mode", choices=["research", "production"], default="research")
    parser.add_argument("--horizons", default="1,5,21")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--entry-lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", action="append", default=[])
    parser.add_argument("--materialize-staging", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=320)
    parser.add_argument("--synthetic-symbols", type=int, default=8)
    args = parser.parse_args(argv)

    pack = load_factor_pack(args.provider)
    require_universe = args.require_point_in_time_universe or args.run_mode == "production"
    config = BatchEvaluationConfig(
        market=args.market,
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        universe_id=args.universe_id,
        universe_dataset=args.universe_dataset,
        require_point_in_time_universe=require_universe,
        backend=args.backend,
        run_mode=args.run_mode,
        horizons=_parse_csv_ints(args.horizons),
        batch_size=args.batch_size,
        min_assets=args.min_assets,
        n_quantiles=args.n_quantiles,
        entry_lag=args.entry_lag,
        cost_bps=args.cost_bps,
        fdr_alpha=args.fdr_alpha,
        materialize_staging=args.materialize_staging,
        publish=args.publish,
        factor_version=f"{pack.name}.{pack.version}",
        strict=not args.allow_partial,
        resume=not args.no_resume,
        limit=args.limit,
        selected_factors=tuple(args.factor),
    )
    frame = None
    snapshot_id = None
    if args.synthetic:
        frame = build_synthetic_market_frame(
            periods=args.synthetic_periods,
            symbols=args.synthetic_symbols,
        )
        snapshot_id = f"synthetic:{pack.name}"
    try:
        summary = run_factor_pack_evaluation(
            pack,
            config,
            output_dir=args.output_dir,
            market_frame=frame,
            snapshot_id=snapshot_id,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str))
    return 0 if summary.status == "success" else 1


__all__ = [
    "BatchEvaluationConfig",
    "BatchRunSummary",
    "FactorDefinition",
    "FactorPack",
    "FactorPackProvider",
    "FactorRunRecord",
    "SplitBoundaries",
    "build_synthetic_market_frame",
    "load_factor_pack",
    "run_factor_pack_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
''',
    )


def write_provider() -> None:
    write(GTJA / "autofactor" / "__init__.py", 'from .provider import load_pack\n\n__all__ = ["load_pack"]')
    write(
        GTJA / "autofactor" / "provider.py",
        '''"""GTJA191 factor-pack provider for the external AutoFactorEvaluation library."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from evaluation.factor_pack import FactorDefinition, FactorPack
from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog, deliverable_names


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_pack() -> FactorPack:
    catalog = deliverable_catalog()
    names = deliverable_names(catalog)
    rows: list[dict[str, Any]] = []
    factors: list[FactorDefinition] = []
    for name in names:
        item = catalog[name]
        formula = str(item["dsl_formula"]).strip()
        source_formula = str(item.get("source_formula") or "").strip()
        formula_hash = _sha(formula)
        row = {
            "name": name,
            "formula": formula,
            "source_formula": source_formula,
            "formula_hash": formula_hash,
        }
        rows.append(row)
        factors.append(
            FactorDefinition(
                name=name,
                formula=formula,
                source_formula=source_formula,
                description=f"GTJA191 canonical factor {name}",
                formula_hash=formula_hash,
                metadata={"domain": "price_volume", "universe": "GTJA185"},
            )
        )
    if len(factors) != DELIVERABLE_COUNT:
        raise ValueError(f"expected {DELIVERABLE_COUNT} deliverable factors, got {len(factors)}")
    source_catalog_hash = _sha(_canonical(rows))
    pack_hash = _sha(
        _canonical(
            {
                "name": "gtja185",
                "version": "1",
                "source_catalog_hash": source_catalog_hash,
                "factors": rows,
            }
        )
    )
    pack = FactorPack(
        name="gtja185",
        version="1",
        source_catalog_hash=source_catalog_hash,
        pack_hash=pack_hash,
        factors=tuple(factors),
        metadata={
            "source": "gtja191/lib/catalog.py",
            "deliverable_count": DELIVERABLE_COUNT,
            "owner": "gtja191",
        },
    )
    pack.validate()
    return pack
''',
    )
    write(
        GTJA / "scripts" / "evaluate_with_autofactor.py",
        '''#!/usr/bin/env python3
"""Evaluate the external GTJA185 pack through AutoFactorEvaluation."""
from __future__ import annotations

import sys

from evaluation.batch import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--provider",
                "autofactor.provider:load_pack",
                "--output-dir",
                "gtja191/output/autofactor",
                *sys.argv[1:],
            ]
        )
    )
''',
    )


def update_public_api() -> None:
    init_path = EVAL / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    text = text.replace(
        "The GTJA185 batch evaluator is dependency-light and must remain importable without\nloading the legacy timeseries/indicator/label stack.  Legacy public names are\ntherefore resolved lazily; importing ``evaluation.gtja185_batch`` no longer\nimplicitly imports vectorbt or opens legacy evaluation resources.",
        "The provider-neutral batch evaluator is dependency-light and remains importable\nwithout loading the legacy timeseries/indicator/label stack. Factor packs live\noutside this package and implement the FactorPack provider contract.",
    )
    insertion = '''    "BatchEvaluationConfig": ("evaluation.batch", "BatchEvaluationConfig"),
    "FactorDefinition": ("evaluation.batch", "FactorDefinition"),
    "FactorPack": ("evaluation.batch", "FactorPack"),
    "run_factor_pack_evaluation": ("evaluation.batch", "run_factor_pack_evaluation"),
'''
    text = text.replace("_LAZY_EXPORTS: dict[str, tuple[str, str]] = {\n", "_LAZY_EXPORTS: dict[str, tuple[str, str]] = {\n" + insertion)
    write(init_path, text)


def remove_specific_files() -> None:
    shutil.rmtree(AUTO / "factor_packs", ignore_errors=True)
    generator = AUTO / "scripts" / "generate_gtja185_pack.py"
    if generator.exists():
        generator.unlink()
    for path in [
        EVAL / "gtja185_batch.py",
        AUTO / "tests" / "test_gtja185_batch_evaluation.py",
        AUTO / "tests" / "test_gtja185_evaluation_contracts.py",
        AUTO / "docs" / "gtja185_batch_evaluation.md",
    ]:
        if path.exists():
            path.unlink()


def write_tests() -> None:
    write(
        AUTO / "tests" / "test_batch_contracts.py",
        '''from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.batch import BatchEvaluationConfig, FactorDefinition, FactorPack
from evaluation.batch_metrics import (
    apply_point_in_time_universe,
    long_short_series,
    performance_stats,
    price_forward_returns,
    purged_split_mask,
)
from evaluation.batch_models import SplitBoundaries


def test_core_has_provider_neutral_pack_contract():
    factor = FactorDefinition("f1", "close", "close", "demo", "hash", {})
    pack = FactorPack("demo", "1", "source", "pack", (factor,), {})
    pack.validate()
    assert pack.select()[0].name == "f1"


def test_forward_returns_enter_after_signal_bar():
    dates = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame({"datetime": dates, "asset": "A", "vwap": np.arange(100.0, 108.0)})
    result = price_forward_returns(frame, (2,), "vwap", entry_lag=1)[2]
    assert result.iloc[0] == pytest.approx(103.0 / 101.0 - 1.0)


def test_purged_split_does_not_cross_boundaries():
    dates = pd.bdate_range("2024-01-02", periods=10)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "asset"])
    bounds = SplitBoundaries(str(dates[4].date()), str(dates[7].date()), str(dates[0].date()), str(dates[-1].date()))
    mask = purged_split_mask(index, "train", bounds, dates, entry_lag=1, horizon=2)
    assert list(np.flatnonzero(mask)) == [0, 1]


def test_weight_turnover_and_costs_are_explicit():
    index = pd.MultiIndex.from_product([pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C", "D"]], names=["datetime", "asset"])
    signal = pd.Series([-2, -1, 1, 2] * 2, index=index, dtype=float)
    returns = pd.Series([0, 0, 0.01, 0.01] * 2, index=index, dtype=float)
    gross, net, turnover = long_short_series(signal, returns, min_assets=4, n_quantiles=2, cost_bps=10)
    assert gross.iloc[0] == pytest.approx(0.01)
    assert turnover.iloc[0] == pytest.approx(1.0)
    assert net.iloc[0] == pytest.approx(0.009)


def test_production_fails_closed_without_pit_universe():
    with pytest.raises(ValueError, match="point_in_time_universe"):
        BatchEvaluationConfig(run_mode="production").validate()
''',
    )
    write(
        GTJA / "tests" / "test_autofactor_integration.py",
        '''from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from autofactor.provider import load_pack
from evaluation.batch import BatchEvaluationConfig, build_synthetic_market_frame, run_factor_pack_evaluation


def test_gtja_pack_is_external_complete_and_versioned():
    pack = load_pack()
    assert pack.name == "gtja185"
    assert len(pack.factors) == 185
    assert len(set(pack.names())) == 185
    assert len(pack.pack_hash) == 64
    for factor in pack.factors:
        if "VWAP" in factor.source_formula.upper():
            assert "col('vwap')" in factor.formula or 'col("vwap")' in factor.formula


def test_all_185_run_through_generic_evaluator(tmp_path: Path):
    pack = load_pack()
    frame = build_synthetic_market_frame(periods=320, symbols=8, seed=191)
    config = BatchEvaluationConfig(
        market="ashare",
        horizons=(1, 5),
        batch_size=16,
        min_assets=6,
        n_quantiles=4,
        factor_version=f"{pack.name}.{pack.version}",
        strict=True,
        resume=False,
    )
    summary = run_factor_pack_evaluation(
        pack,
        config,
        output_dir=tmp_path,
        market_frame=frame,
        snapshot_id="synthetic:gtja185-external",
    )
    assert summary.status == "success"
    assert summary.factor_count == summary.succeeded == 185
    assert summary.failed == 0
    ranking = pd.read_csv(tmp_path / "ranking.csv")
    assert len(ranking) == 185
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["pack_hash"] == pack.pack_hash
''',
    )


def remove_pipeline_coupling() -> None:
    path = AUTO / "pipeline.py"
    text = path.read_text(encoding="utf-8")
    start = text.find("\ndef run_gtja185_batch_pipeline(")
    end = text.find("\ndef _validate_all_paths", start)
    if start >= 0 and end >= 0:
        text = text[:start] + "\n" + text[end:]
    lines = text.splitlines()
    lines = [line for line in lines if "--gtja" not in line]
    text = "\n".join(lines) + "\n"
    block_start = text.find("    if args.gtja185:")
    block_end = text.find("    # 全流程启动前验证所有路径已就绪", block_start)
    if block_start >= 0 and block_end >= 0:
        text = text[:block_start] + text[block_end:]
    write(path, text)


def write_docs_and_workflow() -> None:
    write(
        AUTO / "docs" / "factor_pack_evaluation.md",
        '''# External factor-pack evaluation

`AutoFactorEvaluation-RECONSTRUCT` is a provider-neutral evaluation library. It does not own GTJA, fundamental, analyst, text, alternative-data, flow, or graph factors.

Each factor project remains a sibling directory and exposes a callable returning `evaluation.factor_pack.FactorPack`.

```text
quant_projects/
├── AutoFactorEvaluation-RECONSTRUCT/
├── gtja191/
├── fundamental_factors/
├── news_text_factors/
└── analyst_revision_factors/
```

Generic invocation:

```bash
python -m evaluation.batch \\
  --provider autofactor.provider:load_pack \\
  --output-dir gtja191/output/autofactor \\
  --synthetic
```

A production provider must supply immutable factor names, formulas, formula hashes, source hash, pack hash, version and metadata. The evaluator owns data snapshots, PIT/tradability filtering, execution, purification, train/validation/test isolation, FDR, costs, reports and staging materialization.
''',
    )
    write(
        GTJA / "docs" / "autofactor_evaluation.md",
        '''# GTJA185 evaluation adapter

GTJA185 is owned by the sibling `gtja191` project. `autofactor.provider:load_pack` converts the canonical GTJA catalog into the provider-neutral `FactorPack` contract without copying formulas into AutoFactorEvaluation.

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT:$PWD/gtja191"
python gtja191/scripts/evaluate_with_autofactor.py --synthetic
```

For production add the registered market dataset, PIT universe dataset, date range and `--run-mode production`.
''',
    )
    old = ROOT / ".github" / "workflows" / "autofactor-gtja185-production.yml"
    if old.exists():
        old.unlink()
    write(
        ROOT / ".github" / "workflows" / "factor-pack-evaluation.yml",
        '''name: Factor Pack Evaluation

on:
  push:
    branches: [main, agent/autofactor-gtja185-evaluation]
    paths:
      - "AutoFactorEvaluation-RECONSTRUCT/**"
      - "gtja191/**"
      - "factor_engine/**"
      - "data_access/**"
      - ".github/workflows/factor-pack-evaluation.yml"
  pull_request:
    paths:
      - "AutoFactorEvaluation-RECONSTRUCT/**"
      - "gtja191/**"
      - "factor_engine/**"
      - "data_access/**"

permissions:
  contents: read

jobs:
  core:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
          python -m pip install -r AutoFactorEvaluation-RECONSTRUCT/requirements.txt
      - name: Enforce evaluator independence
        run: |
          test ! -d AutoFactorEvaluation-RECONSTRUCT/factor_packs
          test -z "$(grep -RIlE 'factor_packs\.gtja|gtja191|gtja185' AutoFactorEvaluation-RECONSTRUCT/evaluation || true)"
      - name: Core evaluator contracts without GTJA on PYTHONPATH
        env:
          PYTHONPATH: ${{ github.workspace }}:${{ github.workspace }}/factor_engine:${{ github.workspace }}/AutoFactorEvaluation-RECONSTRUCT
        run: pytest -q AutoFactorEvaluation-RECONSTRUCT/tests/test_batch_contracts.py AutoFactorEvaluation-RECONSTRUCT/tests/test_quant_platform_integration.py

  gtja185-integration:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
          python -m pip install -r AutoFactorEvaluation-RECONSTRUCT/requirements.txt
      - name: External GTJA185 provider and 185/185 evaluation
        env:
          PYTHONPATH: ${{ github.workspace }}:${{ github.workspace }}/factor_engine:${{ github.workspace }}/AutoFactorEvaluation-RECONSTRUCT:${{ github.workspace }}/gtja191
          FACTOR_ENGINE_DISABLE_NUMBA: "1"
        run: pytest -q gtja191/tests/test_autofactor_integration.py gtja191/tests/test_full_catalog_factor_engine.py gtja191/tests/test_gtja191.py
''',
    )


def main() -> None:
    migrate_modules()
    write_factor_contract()
    write_batch_facade()
    write_provider()
    update_public_api()
    remove_specific_files()
    write_tests()
    remove_pipeline_coupling()
    write_docs_and_workflow()
    print("AutoFactorEvaluation decoupled from external factor packs")


if __name__ == "__main__":
    main()
