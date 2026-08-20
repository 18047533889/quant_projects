# Care File

This file is read first by any agent before starting work.

## Quick Rules

1. **Never stop looping unless user says stop.** If queue is empty, find new work or sleep and retry.
2. **Working tree is truth.** No `git checkout/restore/stash/reset/rebase`.
3. **Push after each approved change.** See `.claude/skills/github-push/SKILL.md`.
4. **Evidence is mandatory** for operator/backend/planner changes. See `evidence/r2/` for format.
5. **No full datasets.** Smoke tests and small synthetic data only.

## Read First (in order)

1. `loop/care.md` ← you are here
2. `loop/status.md` or `loop/status_platform.md` ← which loop are you in?
3. `loop/queue.md` or `loop/queue_platform.md` ← what's next?
4. `loop/resume.md` ← any blockers or findings?

## Two Loops

| | FE | Platform |
|---|---|---|
| Entry | `loop/start_fe.md` | `loop/start_platform.md` |
| Status | `loop/status.md` | `loop/status_platform.md` |
| Queue | `loop/queue.md` | `loop/queue_platform.md` |
| Skills | `.claude/skills/factor-engine-rules/` | `.claude/skills/platform-agent-scope/` |

## Bloat

If session grows huge: `bash /home/shw/quant_projects/scripts/cleanup_claude_bloat.sh`
