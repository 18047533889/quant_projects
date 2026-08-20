# Loop Orchestration

## Two Loops

There are two independent loops. Each is a self-sustaining pipeline that keeps running until the user says stop.

### Loop 1: FactorEngine / DataAccess (FE)

- **Entry:** `loop/start_fe.md`
- **Status:** `loop/status.md`
- **Queue:** `loop/queue.md`
- **Resume:** `loop/resume.md`
- **Coordinator:** `.claude/agents/coordinator-fe.md`
- **Agents:** `writer-fe`, `finder`, `dispatcher`, `tester`, `reviewer`

### Loop 2: Platform (因子到模型前全链路)

- **Entry:** `loop/start_platform.md`
- **Status:** `loop/status_platform.md`
- **Queue:** `loop/queue_platform.md`
- **Resume:** `loop/resume.md`
- **Coordinator:** `.claude/agents/coordinator-platform.md`
- **Agents:** `platform-agent` (direct implementation)

## How Each Loop Works

```
Coordinator reads queue → picks next item → dispatches to agent
Agent implements/tests/reviews → marks done → pushes to GitHub
Coordinator reads next item → repeat until queue empty or user stops
```

## Loop Files

| File | Purpose |
|------|---------|
| `loop/README.md` | This file — how to use the system |
| `loop/orchestration.md` | Mechanical details of loop execution |
| `loop/start_fe.md` | FE loop bootstrap — what to do first |
| `loop/start_platform.md` | Platform loop bootstrap |
| `loop/status.md` | FE loop current state (who is doing what) |
| `loop/status_platform.md` | Platform loop current state |
| `loop/queue.md` | FE pending tasks (checkbox format) |
| `loop/queue_platform.md` | Platform pending tasks |
| `loop/resume.md` | Notes, blockers, findings, next steps |

## Running the Loop

### Start the FE loop

Read `loop/start_fe.md` and follow it. The coordinator reads the queue, dispatches agents, and keeps looping.

### Start the Platform loop

Read `loop/start_platform.md` and follow it.

### User says "don't stop"

Set a background loop:

```bash
# FE loop: every 5 minutes, read queue and work
while true; do
  sleep 300
  echo "LOOP_TICK_FE"
done
```

Monitor the sentinel and keep dispatching.

## GitHub Push in Loop

After each approved change, push immediately:

```bash
cd /home/shw/quant_projects
git add factor_engine/ dataaccess/ evidence/ tests/
git commit -m "fix: <description>"
git push origin main
```

For HKUST org repos (if their files changed):

```bash
cd /home/shw/quant_projects/factor_engine && git push origin main
cd /home/shw/quant_projects/dataaccess && git push origin main
```

## Bloat

If the session grows too large:
```bash
bash /home/shw/quant_projects/scripts/cleanup_claude_bloat.sh
```
