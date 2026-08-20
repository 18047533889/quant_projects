# Loop: Platform (因子到模型前全链路)

## What This Loop Does

The Platform loop keeps improving the full stack from factor expression to pre-model evaluation:

- `factor_engine` → output
- `factor_mining` → screening, candidate lake, flip pipeline
- `factor_preprocess` → normalization, winsorization, neutralization
- `factor_optimizer` → combination, weighting, alpha blending
- `factor_assets` → registry, metadata, versioning
- `quant_evaluator` → RankIC, ICIR, group-return, turnover, multi-factor metrics
- `lqtp_client` / `lqtp_python_grpc` → factor submission
- `scripts/cogalpha_lqtp/` → batch pipeline, lake recompute, HTML report render
- `data/cogalpha_lqtp_production/` → production factor lake, screening results

## How to Start

1. Read `loop/queue_platform.md` — what's already queued?
2. Read `loop/status_platform.md` — what's in progress?
3. Read `loop/resume.md` — any notes from the last session?
4. Scan `scripts/cogalpha_lqtp/` and `data/cogalpha_lqtp_production/` for recent activity
5. Scan `quant_evaluator/` for metric gaps or failures
6. Dispatch tasks to `loop/queue_platform.md`
7. Implement directly (no subagents needed for platform work)
8. Push to GitHub after each batch
9. Repeat

## Core Command

From the coordinator, repeatedly:

```
Task = pop next P0 from loop/queue_platform.md
Implement Task
Run tests if applicable
Update queue
Push to GitHub
Loop until queue empty OR user says stop
```

## What to Work On

Ask yourself:

- Are there any factors in `data/cogalpha_lqtp_production/` with wrong sign or missing data?
- Is the lqtp batch submit pipeline running cleanly?
- Are all `quant_evaluator` metrics computed and logged?
- Are the HTML reports rendering correctly?
- Is the flip pipeline identifying all sign-flipped factors?
- Are there any gaps in `factor_preprocess` or `factor_optimizer`?
- Are new factors being registered in `factor_assets`?

Add findings to `loop/queue_platform.md` with priority.

## Weekly Maintenance (Every Loop Tick)

- Run `scripts/cogalpha_lqtp/flip_pipeline.py` — check for new sign-flipped factors
- Run lqtp batch submit for new candidates
- Check `data/cogalpha_lqtp_production/reports/` for HTML rendering issues
- Check `quant_evaluator/` for any failing metric computations
- Update `loop/resume.md` with findings

## GitHub Push After Each Batch

```bash
cd /home/shw/quant_projects
git add factor_engine/ dataaccess/ factor_mining/ factor_preprocess/ factor_optimizer/ factor_assets/ quant_evaluator/ scripts/cogalpha_lqtp/ evidence/ tests/ docs/
git diff --cached --stat
git commit -m "fix: <description>"
git push origin main
```
