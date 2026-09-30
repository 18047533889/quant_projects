# Process-local source identity checks

`runtime.source_identity.ProcessSourceIdentity` is a bounded source identity
checker, now connected to materialized batch calibration with strict content
checks by default and explicit stat-guarded checks. Default auto is unchanged.
Its receipts identify on-disk source, not already loaded Python semantics.
Do not hot-reload or monkeypatch a worker and treat a disk fingerprint as
certification of its running code.

The default stat-guarded mode hashes sources initially and rehashes changed
files only. Guards include device, inode, mode, size, mtime and ctime. Each
check scans entries to detect added/deleted sources; directory-only changes
are reported separately. `strict_full_content=True` rereads every source.
Metadata guards do not detect edits retaining every guard: on server-c, a
bounded 1000-write temporary-file probe observed 747 same-size writes with
unchanged guards within a filesystem timestamp tick. Privileged restoration
has the same limitation. Strict content checking remains calibration's default.

Bounds cover scanned entries, source bytes, changed source paths and retries.
Directory symlinks and escaping Python symlinks fail closed. Read races retry
boundedly. A weak instance registry resets locks after fork, allowing a child
to establish its own nonce/baseline even when a parent thread held a lock.

## Bounded measurement on server-c

Five repetitions on 537 source files (6,422,803 bytes; 1,315 scanned entries):
existing full source hash median 29.745 ms; unchanged stat-guarded identity
5.890 ms (zero files/bytes reread); strict-content identity 23.611 ms.
Guarded and strict identities matched, with no source drift observed.
This is identity-check timing, not evaluator end-to-end speed evidence.

Validation: 12 source-identity tests, including real fork lock inheritance,
passed. The combined source/calibration suite now passes 38 tests, including
mode switching without false drift, zero reread on unchanged guarded hits,
permanent cache rejection after source drift, and actual admitted CUDA parity.
