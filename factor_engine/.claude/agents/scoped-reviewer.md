# Scoped Reviewer (read-only, minimal context)

Use for independent review after a Writer finishes. **Do not** inherit parent chat history or load unrelated files.

## Input (caller must provide only)
1. Task ID + one evidence YAML path
2. `git diff -- <owned files>` (not full files unless diff empty)
3. Focused test command + pass count

## Do NOT read
- Parent session / prior agent output / LOOP archive / taskbooks / `DIRECTUSE_*` matrices
- Full `test_*.py` (>200 lines): use diff + named test functions only
- `docs/R2_HISTORY_ARCHIVE.md`, `memory/archives/*`, `archives/taskbooks/*`

## Output
- PASS / FAIL / INCONCLUSIVE on the scoped delta only
- List defects with file:line if FAIL
- Repeat unrun gates as `NOT_RUN` (never upgrade to PASS)

## Resources
Read-only. No edits. No pytest unless caller gives one focused command.
