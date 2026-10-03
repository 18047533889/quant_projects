# RAW summary ABBA: rejected attempt, 2026-10-04

This is a diagnostic receipt, **not a successful benchmark or speed claim**.
The real manifest-bound F16, 500-date, 256-asset run used one cached warmup
before the planned uncached/cached/cached/uncached measured calls.

- Original observation handle: session 12599; server PID 2609209.
- Confirmed terminal exit: 1; the PID was subsequently absent.
- Rejection: `RuntimeError: benchmark runtime changed during optimizer call`.
- Another window reported a concurrent scoped QE commit; observed HEAD was
  `a34db833f`. The initial captured runtime was not exported, so this receipt
  does not invent its starting HEAD or a complete runtime delta.
- The runtime comparison includes Git HEAD, Python/package versions and the
  recorded thread environment. Selected-source and data checks remain enabled.
- `real_raw_summary_cache_oct04.json` was verified absent after termination.

No measured ABBA comparison completed or qualified. Do not infer cache speed
from this attempt, reuse it as routing evidence, or bypass the drift gate.
Rerun only after a fresh coordinated freeze of source/HEAD and performance slot.
