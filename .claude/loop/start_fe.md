# Loop: FactorEngine / DataAccess

## What This Loop Does

The FE loop keeps improving `factor_engine/` and `dataaccess/`:

- New operators and operator fixes
- Backend parity (Polars/DuckDB/NumPy/Q)
- Planner improvements and bug fixes
- Evidence YAML maintenance
- Test coverage
- Performance improvements

## How to Start

1. Read `loop/queue.md` — what's already queued?
2. Read `loop/status.md` — what's in progress?
3. Read `loop/resume.md` — any notes from the last session?
4. Run **finder** — scan for problems (see `.claude/agents/finder.md`)
5. Run **dispatcher** — break findings into tasks (see `.claude/agents/dispatcher.md`)
6. Run **writer-fe** — implement tasks (see `.claude/agents/writer-fe.md`)
7. Run **tester** — write and run tests (see `.claude/agents/tester.md`)
8. Run **reviewer** — final review (see `.claude/agents/reviewer.md`)
9. Push to GitHub after approval
10. Repeat

## Core Command

From the coordinator, repeatedly:

```
Task = pop next P0 from loop/queue.md
Agent = read Task.agent (writer-fe / tester / reviewer)
Run agent with Task
On complete: mark done in queue, push to GitHub
On flag: reassign to writer-fe
On block: write to loop/resume.md, pop next task
Loop until queue empty OR user says stop
```

## What to Work On

Ask yourself:

- Are there any `NOT_RUN` items in `evidence/r2/` YAML files?
- Are there any tests that have been consistently failing?
- Are there any operator stubs (not implemented) that should be implemented?
- Is the planner handling all canonical operator types?
- Are there any backend parity gaps?
- Is the Q compiler passing its test suite?

Add findings to `loop/queue.md` with priority.

## GitHub Push After Each Approved Change

```bash
cd /home/shw/quant_projects
git add factor_engine/ dataaccess/ evidence/ tests/
git commit -m "fix: <description>"
git push origin main

# HKUST org repos if changed:
cd /home/shw/quant_projects/factor_engine && git push origin main
cd /home/shw/quant_projects/dataaccess && git push origin main
```
