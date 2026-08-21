# Care File

This file is read first by any agent before starting work.

## Quick Rules

1. **Never stop looping unless user says stop.** If queue is empty, find new work or sleep and retry.
2. **Working tree is truth.** No `git checkout/restore/stash/reset/rebase`.
3. **Push after each approved change.** See `.claude/skills/github-push/SKILL.md`.
4. **Evidence is mandatory** for operator/backend/planner changes. See `evidence/r2/` for format.
5. **No full datasets.** Smoke tests and small synthetic data only.

## Transient API Errors (400 / 524 / gateway) — MANDATORY recovery

Subagents WILL hit upstream API errors (HTTP 400 "上游服务暂时不可用", 524 Cloudflare timeout, 5xx). This is EXPECTED, not a reason to stop.

**When a subagent dies on a transient API error:**
- Resume it via SendMessage with the agent ID + a short "continue where you left off" message. Do this EVERY time, however many times it happens.
- Before resuming, back off ~120 seconds (the gateway's retry_after) so the upstream has recovered.
- The message must re-state the current task step so the agent can continue without losing context.
- Do NOT treat a transient API error as task failure. The task is only done when the agent reports a real result (tests run / files written / evidence produced).

**What subagents themselves should do on 400/524:**
- Do not stop. Sleep ~120s (time.sleep), then retry the operation that failed.
- Prefer Read/Edit/Write tools over shell inspection when the shell returns odd output.
- Keep going through retries until the work completes.

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
