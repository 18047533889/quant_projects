# Loop control (single home)

All live loop files live **here**. Do not create new `LOOP_*` / `AGENT_*` / `RESUME_*` files at repo root.

**What the user cares about (read this, not archives):** [care.md](care.md)

| File | What |
|---|---|
| [care.md](care.md) | Standing contract — travels with the repo |
| [orchestration.md](orchestration.md) | Roles + cycle |
| [start_platform.md](start_platform.md) | Paste this into a **platform** Claude session |
| [start_fe.md](start_fe.md) | Paste this into a **FactorEngine/DA** session |
| [status.md](status.md) | FE/DA compact truth + evidence pointers |
| [platform.md](platform.md) | Platform-package lane status |
| [queue.md](queue.md) | Open tickets |
| [running.md](running.md) | Who is spawned this cycle |
| [resume.md](resume.md) | After `/clear` |
| [findings.md](findings.md) | One-line findings |

**Never idle:** subagent 返回立刻派下一个。停：`touch loop/HALT` 或你说停。


**Agents (one copy):** `../.claude/agents/`

**History (do not load):** `../archives/` · `../docs/R2_HISTORY_ARCHIVE.md`

**Repo auto-load:** root `CLAUDE.md` only (short pointer). Package `factor_engine/.claude/skills/` still applies in FE trees.
