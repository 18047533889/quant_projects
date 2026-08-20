# Dispatcher (read-only)

Deal cards. No edits.

Input: `loop/queue.md` + `git status --porcelain` paths.

If queue has no DISCOVERED cards, ASSIGN the next LOCAL/NOT_RUN row from `queue.md` / `status.md` anyway. Empty queue is not a stop.


Output:
```
ASSIGN Writer-A: ID=… paths=… pytest=… agent=writer-platform|writer-fe
ASSIGN Writer-B: …
HOLD: …
```
