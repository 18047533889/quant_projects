# Quant Factor Platform — Project Rules (compact)

## Mission
Pre-model factor stack only: `DataAccess → FactorEngine → QuantEvaluator → (optional FactorOptimizer) → FactorAssets → FactorPreprocess → ModelInput`. No training / portfolio / execution in this phase.

## Source of truth
1. Server working tree is code truth (not GitHub).
2. Data/PIT/calendar/universe/snapshot → `dataaccess`.
3. DSL/AST/IR/operators/materialize → `factor_engine`.
4. New metrics → `quant_evaluator`; mutations → `factor_optimizer`; identity/lifecycle → `factor_assets`; pre-model transforms → `factor_preprocess`.

## Before editing
- Skim this file + repo-root `../CLAUDE.md` (or `/home/shw/quant_projects/CLAUDE.md`).
- Load skill `quant-factor-platform-development` only for blueprint package work.
- Read **only** the `AI_GUIDE/*.md` chapter for the package you touch — never dump the whole guide into the prompt.
- `git status` / `git diff --stat` / search existing DA/FE/legacy before adding modules.

## Do not rebuild
Second Data lake / PIT / calendar / universe / DSL / materializer / cache platform / generic DAG / large backtest-execution-optimizer.

## Engineering
Batch-first; Reference/Fast with parity; no silent Pandas in production; fitted objects record fit window; no `sys.path` hacks; hard gates + evidence; unrun = `NOT_RUN`.

## Acceptance (per new package)
Independent venv + pytest; legacy/boundary audits; parity/benchmarks/corpus where applicable; docs/API/schemas synced. Details: matching `AI_GUIDE/` chapter only.
