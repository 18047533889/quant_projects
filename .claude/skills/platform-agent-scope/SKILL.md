---
name: platform-agent-scope
description: Platform agent scope —因子到模型前的全链路
alwaysApply: true
---

# Platform Agent Scope

This agent owns the full stack from factor expression to pre-model pipeline: **factor_engine → factor_mining → factor_preprocess → factor_optimizer → factor_assets → quant_evaluator**.

## What to work on

- **factor_engine** — operator DSL, backends (Polars/DuckDB/NumPy/Q), planner, runtime, storage/materializer, IR, expr
- **factor_mining** — screening, candidate lake, COS read wrappers, mining pipeline scripts
- **factor_preprocess** — normalization, winsorization, neutralization, factor cleaning
- **factor_optimizer** — combination, weighting, alpha blending, regime routing
- **factor_assets** — factor registry, metadata, versioning, universe mapping
- **quant_evaluator** — RankIC, ICIR, group-return, turnover, multi-factor metrics, tear-sheet
- **lqtp_client / lqtp_python_grpc** — factor submission, batch pipeline, formula conversion
- **scripts/cogalpha_lqtp/** — flip pipeline, batch submit, lake recompute, HTML report render
- **data/cogalpha_lqtp_production/** — production factor lake, screening results, flip logs
- **CLI / API** — factor_frontend, quant-server, quant-agent-backend integration points

## What to leave alone

- `quant-server` / `quant-factor-platform` / `quant-risk` — separate services (org repos)
- `infra-*` directories — infrastructure, not this agent's scope
- `gtja191/` — legacy, mostly read-only
- `modeling/` — post-factor, not this agent's scope

## Loop

- Read `loop/orchestration.md` for how to run
- Loop entry: `loop/start_platform.md`
- Status file: `loop/status_platform.md`
- Queue: `loop/queue_platform.md`

## GitHub Push

```bash
cd /home/shw/quant_projects
git add factor_engine/ dataaccess/ factor_mining/ factor_preprocess/ factor_optimizer/ factor_assets/ quant_evaluator/ scripts/cogalpha_lqtp/ evidence/ tests/ docs/
git diff --cached --stat
git commit -m "..."
git push origin main
```
