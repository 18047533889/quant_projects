# quant_projects — Claude Workspace Rules

## Two Loops

This repo has **two independent self-running loops**. Start them and they keep going until you say stop.

### Loop 1: FactorEngine / DataAccess

- **Scope:** `factor_engine/` + `dataaccess/`
- **Entry:** Read `loop/start_fe.md` and follow it
- **Agents:** writer-fe, finder, dispatcher, tester, reviewer (use subagents)
- **Skills:** `.claude/skills/factor-engine-rules/` (always on), `.claude/skills/verify/`, `.claude/skills/github-push/`

### Loop 2: Platform (因子到模型前全链路)

- **Scope:** `factor_mining/` → `factor_preprocess/` → `factor_optimizer/` → `factor_assets/` → `quant_evaluator/` → lqtp → `scripts/cogalpha_lqtp/`
- **Entry:** Read `loop/start_platform.md` and follow it
- **Agent:** coordinator-platform (direct implementation)
- **Skills:** `.claude/skills/platform-agent-scope/` (always on), `.claude/skills/github-push/`

## Always-Apply Skills

Both loops read these skills automatically:

```
.claude/skills/factor-engine-rules/SKILL.md    # FE rules, always on
.claude/skills/platform-agent-scope/SKILL.md     # Platform scope, always on
```

## GitHub Push

After each approved change, push immediately. See `.claude/skills/github-push/SKILL.md`.

| Repo | Remote |
|------|--------|
| quant_projects | `github.com/18047533889/quant_projects` |
| factor_engine | `github.com/HKUST-QUANT-SOCIETY/factor_engine` |
| data_access | `github.com/HKUST-QUANT-SOCIETY/data_access` |

## Hard Rules

- **Working tree is truth.** No `git checkout/restore/stash/clean/reset/rebase`.
- **No plan mode unless user asks.** Execute immediately.
- **Evidence is mandatory** for operator/backend/planner changes — write YAML in `evidence/r2/`.
- **Never stop looping** unless user explicitly says stop.
- Never load `archives/` or `docs/R2_HISTORY_ARCHIVE.md` wholesale.
- Brief ≤30 lines per loop/status file.
- No full dataset runs unless explicitly requested.

## Bloat

If session grows huge: `bash /home/shw/quant_projects/scripts/cleanup_claude_bloat.sh`
