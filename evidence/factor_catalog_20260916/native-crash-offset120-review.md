# Native crash review: Donchian rerun offset 120

Date: 2026-09-16  
Scope: read-only review of existing `donchian-position-semantic-rerun-r3`
artifacts. No reproduction, scan, core dump, installation, or code change was
performed.

## Conclusions

- **SIGSEGV is established.** The shard ledger records subprocess return code
  `-11` for offset 120, and the corresponding log ends with `EXIT:-11`. On this
  Linux host, signal 11 is `SEGV`.
- **The responsible module is not established.** Existing evidence cannot
  distinguish Python runtime, FactorEngine code, or a native dependency such
  as NumPy, pandas, Polars, PyArrow, DuckDB, or psutil.
- **The 120-second faulthandler output is not a crash stack.** It was emitted by
  `faulthandler.dump_traceback_later(120, repeat=True)` in `smoke_catalog.py`.
  It is a periodic snapshot of thread state near the failure, not a native
  SIGSEGV backtrace.
- Resource-admission and full-history/warmup messages explain the transition
  into singleton retries. They do not establish the cause of SIGSEGV.

## Evidence

1. `donchian-position-semantic-rerun-r3.ledger.jsonl`, entry 7:
   offset `120`, limit `20`, exit `-11`.
2. `donchian-position-semantic-rerun-r3-120.log` ends with `EXIT:-11`.
   Offset 100 is separately recorded as exit `124`; the offset-120 result is
   therefore not being inferred from the watchdog timeout code.
3. Before termination, the log records a 10-factor batch abort caused by
   `ResourceAdmissionError`, followed by singleton retry starts for source rows
   32087, 32088, 32089, 32132, and 32133. The last emitted retry marker does not
   prove that row 32133 caused the crash.
4. The timed Python traceback shows the resource-autopilot thread in
   `_run -> _tick -> ResourceBroker._refresh -> _process_family_rss -> _walk
   -> _one`, while reading process memory data. The main thread is truncated
   after `<frozen posixpath>._joinrealpath`; no complete caller chain or native
   frame is present.
5. The log contains no `Fatal Python error: Segmentation fault`, allocator
   corruption, double-free, or named native-extension failure message.
6. No core was found. The observed shell has `ulimit -c = 0`, and the host
   `core_pattern` routes dumps through apport. The failed shard left only an
   empty temporary output file and no summary.

Observed environment: Python 3.12.3, NumPy 2.2.6, pandas 2.3.3, Polars 1.42.1,
PyArrow 25.0.0, DuckDB 1.5.4, and psutil 7.2.2.

## Unknowns

- The native instruction and shared object that received SIGSEGV.
- Whether the failure is deterministic, data-row-specific, batch-interaction
  specific, or timing/concurrency dependent.
- Whether the truncated `realpath` frame was executing at the fault or merely
  captured by the periodic timeout dump.

## Suggested bounded follow-up (not executed)

Do not resume the catalog-wide scan. If a reproduction is later authorized,
start with a separately supervised subprocess covering only the first offset-120
batch (`offset=120`, `limit=10`, `batch_size=10`), retaining the 2 GiB watchdog
and enabling `PYTHONFAULTHANDLER=1` / `-X faulthandler`. Record source and
dependency hashes, stdout, stderr, and the raw return code. If it reproduces,
