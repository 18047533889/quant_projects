# quant_projects — Claude entry (travels with the repo)

Read **this file** then **[`loop/care.md`](loop/care.md)**. That is the standing contract. Do not scrape the rest of the markdown tree.

| Need | Open |
|---|---|
| What I care about | [`loop/care.md`](loop/care.md) |
| Agent loop | [`loop/README.md`](loop/README.md) · [`loop/orchestration.md`](loop/orchestration.md) |
| Roles | [`.claude/agents/`](.claude/agents/) |
| Platform paste | [`loop/start_platform.md`](loop/start_platform.md) |
| FE/DA paste | [`loop/start_fe.md`](loop/start_fe.md) |
| New host | `bash scripts/bootstrap_claude_host.sh` |

## Execute

- No plan mode unless asked. Implement end-to-end. No “是否继续”.
- Continuous loop until user stops: Coordinator spawns Finder → Dispatcher → ≤2 Writers → Tester → Reviewer. Main does **not** bulk-edit. **A returned subagent is the cue to spawn the next one, not to stop.**
- Working tree is truth. No `git checkout|restore|stash|clean|reset`. Push/commit only if asked.

## Context

- Never load `archives/`, `docs/R2_HISTORY_ARCHIVE.md`, or Claude memory archives wholesale.
- Subagent brief ≤30 lines. Reviewer: `git diff` + one evidence YAML.
- ≤15 GiB · serial pytest · threads=1 · unrun=`NOT_RUN`.
