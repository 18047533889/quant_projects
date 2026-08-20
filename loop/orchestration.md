# Continuous multi-subagent loop

**Never stop until `loop/HALT` exists or the user says 停 / stop / `/clear`.**  
Main session = Coordinator only. **No bulk source edits in main.**

Index: [README.md](README.md) · cares: [care.md](care.md)

## Never-idle rule

| Event | What Coordinator does next |
|---|---|
| Finder / Dispatcher / Writer / Tester / Reviewer **returns** | Spawn the **next** role **immediately**. Do not chat. |
| Reviewer PASS or FAIL-bounce done | Status one-liner → **start cycle N+1** (Finder) now. |
| Finder says `NONE` | Not a stop. Dispatcher assigns next open/NOT_RUN ticket, or Finder hunts another path. |
| Queue empty | Finder must still emit a NEW ticket from a different file/package. |
| User silent | Keep cycling. Silence is not permission to stop. |

`touch loop/HALT` (or write that file) is the only machine stop switch.

## Roles (one spawn = one role)

Canonical agents: `.claude/agents/` (do not duplicate under `factor_engine/.claude/agents/`).

| Role | File | Job |
|---|---|---|
| **Finder** | `finder.md` | Hunt defects; one-line tickets on `queue.md` |
| **Dispatcher** | `dispatcher.md` | Deal 1–2 disjoint tickets |
| **Writer** | `writer-platform.md` or `writer-fe.md` | Owned paths + focused pytest |
| **Tester** | `tester.md` | Re-run tests; FAIL → bounce Writer |
| **Reviewer** | `reviewer.md` | `git diff` + evidence; FAIL → bounce |
| **Coordinator** | `coordinator.md` | Main session; keep spawning |

Concurrency: ≤2 Writers (disjoint) + 1 Finder. Then Tester, then Reviewer. ≤15 GiB.

## Cycle

```
Finder → queue.md
Dispatcher → ASSIGN 1–2 tickets
Writers → patch + focused pytest
Tester → FAIL bounce
Reviewer → FAIL bounce
Coordinator → one line on status.md or platform.md → Finder again (no pause)
```

## Lanes

| Lane | Start paste | Status | Writer |
|---|---|---|---|
| Platform | [start_platform.md](start_platform.md) | [platform.md](platform.md) | `writer-platform.md` |
| FE/DA | [start_fe.md](start_fe.md) | [status.md](status.md) + [queue.md](queue.md) | `writer-fe.md` |

## Ticket line

`ID | DISCOVERED\|FIXING\|TESTING\|REVIEW\|CLOSED | paths | pytest | evidence`

Max 8 open tickets.

## Context

Standing cares: [care.md](care.md). Subagent brief ≤30 lines. Never load `archives/`, taskbooks, `docs/R2_HISTORY_ARCHIVE.md`, parent chat, full huge tests. Unrun = `NOT_RUN`. No destructive git. Pytest: `OPENBLAS/OMP/MKL/POLARS=1`.
