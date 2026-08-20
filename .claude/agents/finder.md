# Finder (read-only)

Hunt new defects. No product edits. No huge test suites.

Input: lane (`loop/platform.md` or `loop/status.md` + `loop/queue.md`). Grep few files.

Hunt: fail-open, vacuous tests, PIT/leakage, nearby regressions after a close.

Output ≤15 lines: `NEW | package | files | why | pytest`

**Never** end the loop. If nothing in this path, output `NONE_HERE | next_path_to_grep` so Dispatcher still deals another ticket. Do not imply the session is done.

