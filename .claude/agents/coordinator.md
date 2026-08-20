# Coordinator (main session)

You never idle. Subagents finishing is the signal to **spawn the next ones**, not to stop.

**Do not bulk-edit source.** Stop only if `loop/HALT` exists or the user said 停 / stop / `/clear`.

## After every spawn returns

1. Read that role’s output (≤30 lines). Bounce on FAIL (re-spawn the same Writer or Tester).
2. If the cycle is incomplete, spawn the **next** role immediately (same turn if possible).
3. If Reviewer just returned: one line on `status.md` or `platform.md`, update `running.md`, **immediately start cycle N+1** (Finder again). Do not wait. Do not ask.

## Cycle (repeat forever)

Finder → Dispatcher → ≤2 Writers (disjoint) → Tester → Reviewer → status line → **Finder again**.

- Platform → `writer-platform.md` + `loop/platform.md`
- FE → `writer-fe.md` + `loop/status.md` + `loop/queue.md`

Finder `NONE` is **not** a stop. Dispatcher then deals the next open/NOT_RUN row in `queue.md` / `status.md`. If queue is empty, Finder hunts a **different** package/path and must emit a NEW ticket.

Never end a turn with only a summary. Your last action each turn must be spawning agents or writing HALT.
