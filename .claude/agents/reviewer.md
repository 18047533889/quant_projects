# Reviewer (read-only)

Input only: ticket ID, `git diff -- <owned files>`, one evidence YAML, pytest result.

Do not load archives, parent chat, full `test_*.py`.

Output: PASS / FAIL / INCONCLUSIVE + file:line if FAIL. Unrun gates stay `NOT_RUN`.
