# v9 continuation checkpoint — 2026-09-09

Only server-c working tree `/home/sunhaiwei/quant_projects`; no commit, push,
deployment or publication. Preserve concurrent modifications. This checkpoint
supersedes earlier wave-3 resume notes. Historical test evidence is in
`V9-PROGRESS.yaml`; do not add historical test counts as coverage.

## Latest integrated work

- M36 RESOURCE v5 NOW MERGED: operator_cost9f71deb99510b66bb40abea2b415d22b82ece70e6027daba82e61c8592dadd8e,
  new test factor_engine/tests/backend/test_v9_m36_resource_admission.py,
  main16passed/2warnings/24.19s, v9-m36-resource-focused-main.xml. Independent
  scope review approved. Manifest V9-M36-RESOURCE-MERGED.yaml. Mixed legacy
  benchmark suite17pass/2fail/1error because absent baseline JSON and parent
  registry thaw; Huber independently reproduces old-source behavior and fixes
  test isolation if justified, no fake performance measurements.
  Pending-resume candidate98356abf is NOT merged: root flagged missing resume
  eligibility filtering in next/refill admission (only main loop filtered),
  and fork-only test misses default spawn/direct mode. A repairing; wavelet
  independently reviews mixed accepted/done worker cases and deadline/proof.

- Full v9 main after C and non-weakened test alignments: 717passed/60warnings/
  187.84s, v9-wave8-durable-final.xml. Preservation124pass/1existing-xfail and
  auto80 policy75pass remain separate overlapping scopes, not additive coverage.
  New pending-resume implementation is stage-only with v8_protocol: only
  NEVER-DISPATCHED ACCEPTED/attempt0/no generation/NOT_STARTED, and only v1
  indexed evidence + strict complete SQLite EXITED proof + original deadline
  + coordinator lock. Anything ambiguous remains rejected. No gate changed
  in main yet. M36 cost stage also unmerged: caller rows are flattened total
  observations (estimate_plan_rows), DO NOT multiply instruments again. Root
  requested corrected row-unit proof and final admissible rounding guard.

- M11-C preservation FINAL MAIN: 124passed/1existing-xfail/70warnings/138.22s,
  v9-m11c-preservation-final.xml; full test-only patchc7cf4e is merged and
  independently approved (v6+v8 exact fault/slot identity, not weakened gates).
  See V9-M11C-DURABLE-MERGED.yaml and V9-DEFAULT-CALL-AND-FAILURE-EVIDENCE.md.
  Current new task: M36 runtime workspace admission stage at
  /tmp/v9-m36-resource.BrBjqZ. DO NOT APPLY v2 a2dd0997: root rejected max-only
  memory estimate omitting simultaneously-live input/output and uncertainty<1
  reducing actual admissible bytes beneath the purported floor. Huber repairs;
  wavelet independent review. Main operator_cost remains47efcab7.

- Latest main verification: wave8 716pass/1fail (owner forwarding mock omitted
  transport budget). Root amended only that mock and added exact budget/ID
  forwarding assertions; owner suite now10pass, independent review approved.
  Default auto80 policy75pass. Actual main slot suite5pass/1fail/1existing-xfail:
  newer-slot test still expects b CANCELLED but actual is FAILED. A's reported
  overlay41pass used a not-yet-delivered updated test; DO NOT treat it as main
  preservation PASS. A and wavelet now complete/review missing test-only updates.
  v6 two-test patch956b2c is reviewed2pass but not yet merged; await combined
  source-verified patch. Full pending execution gate remains intact.

- CORRECTION TO PENDING-LEGACY HYPOTHESIS BELOW: independent synthetic probe
  reaches the pre-existing fail-closed pending execution gate first (bounded
  pipeline about line1449). Actual same-run pending execution requires persisted
  worker-exit proof and is NOT yet enabled. Do not remove this gate to make an
  evidence test pass. The suspected marker write is not currently reachable
  through that public path; no proven pending-resume regression is claimed.
  Complete pending execution remains OPEN, separate from terminal-artifact resume
  and missing-receipt evidence validation. Default example corrected accordingly.

- NEW ROOT REVIEW FINDING: completed-only legacy resume tests do not exercise
  new compute. A legacy run with pending work has no v1 marker, yet newly added
  evidence writes require that marker. Agents v8_protocol and v9_wavelet are
  reproducing/repairing this genuine continuation gap, stage-only follow-up.
  Keep legacy provenance unavailable, do not fake a complete history or weaken
  new-run marker gates. Main v3 remains current while follow-up is tested.

- M11-C runtime v3 IS NOW MERGED: patch 5596a4870ad45fcbd54ca4e359540a176705125884a3ebeb492adc69fbce3dc4.
  Main focused 66 passed/35 warnings/34.76s (v9-m11c-durable-main.xml).
  Broad preservation 115 passed/2 failed/59 warnings/126.05s
  (v9-m11c-preservation-main.xml). Both failures in test_v6_bounded_pipeline.py:
  quarantine injection targets obsolete _compute_wave instead of _compute_wave_worker;
  spawn-exit expected CANCELLED3 but new actual failing-slot authority yields
  FAILED1/CANCELLED2. Agent v8_protocol preparing narrow tested alignment with
  exact injected-fault/ordinal/commit-state assertions, stage only. Do not remove
  old assertions without equivalent-or-stronger semantic proof. Current runtime
  hashes: batch7bae6f96, engine6d2a0707, boundeda07194d0, state3397822c,
  resume57435fea. These supersede all historical stage-only notes below.

- Main wave7 explicit v9 files plus generic snapshot suite: 691 passed,
  26 warnings, 180.83s, evidence/r2/v9-wave7-targeted.xml. This excludes the
  still-staged runtime durability candidate. Earlier broad-directory collection
  was interrupted before test execution and is not a pass. Wavelet independently
  approved latest C candidate including byte-bounded pages, orphan-link rejection,
  strict cursor types and missing-receipt hard-crash recovery. Await A final full
  server-overlay tests (including real engine) and hash-consistent patch.

- Latest continuation: M33/M36 relational export v3 and R19 fixes are merged;
  main 22 passed, 3 warnings, 81.78s; see V9-RELATIONAL-R19-MERGED.yaml.
  Registry SHA 2dad4938a702dff291d553c14bb43724b3d5250c90b3d512f553c2fa8e4159e2.
  Canonical/backend inventory preserved (1756/4072); live export refresh in progress.
  M11-C runtime is STILL STAGED on Mac /tmp/v9-m11c-stage, server overlay
  /tmp/v9-m11c-runtime-overlay. Ordinal-link streaming storage has 22 scoped
  tests passing; newer-slot fault proof and byte-bounded evidence pagination,
  orphan-link rejection and strict next_seq type are under review. Do not apply
  any old runtime draft. Agents v8_protocol and v9_wavelet review it together.

- Snapshot follow-up is merged too: module now
  `6447e867beef9cc2fc7f30e040d19033946a45b0dad8ea58546698ee88244ec1`.
  Incremental UTF-8 wire validation and duplicate/sequence checks: main40pass,
  `v9-m11c-snapshot-hardening-main.xml`. Previous 52bb is historical.
  Additional unmerged C review blockers: newer/refill failing-slot recovery must
  retire/reconcile the actual failed slot; every pre-decode failure must record
  unavailable without overwriting already-observed evidence; metadata limits
  (512 assignments/64KiB) must not introduce post-compute failure for previously
  legal large-lookahead/long-name waves. Registry relational preflight must reject
  before backend publication, and final full-bootstrap inventory must preserve
  canonical/backend coverage. Agents are addressing these; do not apply stale
  runtime fca21 or registry v2 patches.

- Latest M11-C update: standalone wire snapshot module is now merged, SHA
  `52bb6f5414237c9f49112dc0c5c422d4c5592232ae07448241912c26edc7b826`.
  Main snapshot plus receipt integration suite: 52 pass, 2 warnings, 26.15s;
  see `V9-M11C-SNAPSHOT-MERGED.yaml`. Runtime transport/durable table/resume
  changes are NOT merged: draft `/tmp/v9-m11c-runtime-durability.patch` hash
  `fca21c30...` is under review and must not be blindly applied. Root flagged
  payload length checks after fetch, assignment scalar bounds, missing new-table
  evidence masquerading as legacy, repeated legacy resume losing unavailable
  provenance, and lack of non-vacuous persistence/tamper integration tests.
  Agent v8_protocol is repairing these; v9_wavelet independently reviews runtime
  slot/adapter/failure protocol. Huber agent is on canonical relational export
  drift (Agent Polars parameter schemas lost M33/M36 joint relations), stage only.

- Newest checkpoint (supersedes historical hashes below): M11-A owner plumbing,
  M11-B real producer, integer bounds and M35 scale hardening are all merged.
  Core `3341881a4f013f67ea08f2e4783cd46bb05c8542b6dd6e07c9cb4935072bcbd5`;
  research `0af43f7f0fb45f2c01b2a990e25b01e0f4b069e6bbadca951d1745aa4e2156b3`.
  Full v9 at these sources: **643 passed, 21 warnings, 122.25 seconds**,
  `v9-wave6-owner-scale.xml`. M11-focused: 33 pass; M35-focused: 41 pass;
  v6/v8 runtime, receipt, resume and DataAccess preservation: 108 pass.
  Public planner probe passed serially: `V9-PUBLIC-PLANNER-POST-M11AB.json`.
  Scoped compileall and diff-check passed. These are not full-repository gates.
  Main live dependency export is now durable: 4,072 rows / 1,756 canonicals,
  every row PARTIAL / NOT_CLAIMED; `V9-LIVE-BINDINGS-POST-M11AB.yaml` gives hashes.
  Current synthetic CPU/RSS ledger: `V9-M35-SCALE-MERGED.yaml`; no auto/GPU claim.
  M31/M33/M36 independent review approved only scoped numerical/runtime domains.
  Active next work: M11-C wire snapshot and durable evidence (agents v9_wavelet
  and v8_protocol, stage only), plus actual catalog joint-domain drift check
  (v9_huber_redesign). All agents remain GPT-5.6-sol. No main C changes yet.

- Continuation update: M11-B public dynamic producer and ordinary date index
  support are merged. Main core is now
  `a6c424245c6389a7386fafa3ea363cb172287ce8b7f77c1ecbab2a23a04fbc0f`;
  dynamic source is `6d94d101f5e1ede4935d09ad9d92d7d298c7f22e099b3b4b7610da608099e129`.
  Focused main receipts: 18 passed, 2 warnings; evidence `V9-M11B-MERGED.yaml`.
  All-v9 suite at this source state: 611 passed, 21 warnings, 121.28 seconds,
  `v9-wave5-before-owner.xml`. It excludes the still-staged runtime owner,
  integer-diagnostic bound and spectral scale-hardening patches.
  Older core hashes and smaller test totals below are historical, not current.

- Shared Huber / sparse LAD / sparse pinball quantile regression and numerical
  scale/restoration checks are integrated. M08/M09 plus Huber/Ridge/Hill: 57 pass,
  12 warnings, `/tmp/v9-m08-m09-merged.xml`. Seven actual affected quantile
  canonicals now version 2; alternate backend parity remains open.
- M10 relative Mahalanobis covariance geometry, M22 blockwise distance
  correlation/covariance, M23 explicit ordered-category/quantile TE policy are
  integrated. TE actual public and existing tests: 44 pass.
- Wavelet five CPU bridges and authoritative window domain, Butterworth cycles
  per bar and full-history declaration, Bessel finite-event-clock SOS are
  integrated. Filter final public CPU bridge tests: 57 pass. No native claim.
- Q01 inventory now reads actual research winners, records SQL declaration
  entrypoints as PARTIAL (runtime emitter not traced), and never calls an empty
  dependency graph resolved. Independent all-export observed 4,072 backend rows,
  1,756 canonicals, all PARTIAL / NOT_CLAIMED, no empty entrypoints. Fresh durable
  main export should follow the final source changes.
- Q02 adds 21 non-vacuous public CPU tests across five panel models: immature
  label boundary, prefix, 60-bar replay, future-only members, and independent
  PCR OLS reference. Window replay is not runtime checkpoint restoration.
- Q01/Q02/identity combined actual main: 100 pass, 2 warnings,
  `/tmp/v9-q01-q02-identity-merged.xml`.
- M26 BDS common-center probability, conditioned integral and stable
  original-deviation geometry merged from patch
  `04d6a3d5e46ad081e1b2392973a38a52d0b123e9ef08b895dda34765e5e1c318`.
  Source `22c1be078f77f93c65864fa9904db952fc5761984ba7cbf1467b50efa83d3c76`.
- M43 stable episode-first anchors and local missing-coverage masks merged from
  patch `349122f6a081945f0abe818867665f9660b93b80779610726316c19920773ec8`.
  Event source `00335d6866a89ea47bfc90257e596247dd63718c9b15a8090a752936138ba4d5`;
  history source `30d46c843a07aed822b164073231fc89f1a9341098d838d8918a84c7021421b7`.
  Immediate-predecessor gap determines first status, so refractory-1 preceding
  bars suffice before an eligible anchor; this is not an arbitrary warmup claim.
  Response outputs exclude unproven anchors; diagnostics mask unknown raw cohorts.
  All five affected canonicals version 2. M26/M43 actual merged test:
  125 passed, 2 warnings, `/tmp/v9-m26-m43-merged.xml`.

## Active follow-up work (all subagents GPT-5.6-sol)

Latest main coherent run: 593 passed, 21 warnings, 121.67 seconds,
`evidence/r2/v9-wave4-combined.xml`. It includes M35 intercepts (research source
`01bda330e59781ede40a8165f9f157982f12754d720a0fc2c8ea42b238a4729e`)
and current M08/Huber core ea3f72b6, but excludes M11 and scale-hardening candidates.
M35/KKT/BDS/nonlinear combined separately: 52 pass, 2 warnings.
The original 9 M34 no-intercept-oracle failures are preserved as historical;
the new augmented-system KKT oracle is independently reviewed and retains the
two-RHS allocation check plus public infeasible-fold rejection.

- v9_huber_redesign: M11 bounded immutable per-window outcome/scope propagation,
  compatible interfaces, explicit kernel-only when factor/run identity absent;
  avoid inventing a second runtime context authority. Narrow stage only.
  HISTORY: first candidate was mistakenly written directly to main (core SHA
  `4f38a22dd962c4e59cb4e2dfc6d765fd5aecdb4cf0477687306ae0ce232acec7`
  and test_v9_m11_fit_receipts.py). Root review rejected unpublished legacy-callable status,
  incomplete scope coordinates, shallow immutability and shared-sink concurrency.
  Root checked exact hashes, archived this candidate in the stage as
  `_rolling_core.unreviewed-main.py`, moved its new test to
  `test_v9_m11_fit_receipts.unreviewed-main.py`, and restored only these
  unreviewed edits. Main now has the verified M08/M09 core SHA
  `ea3f72b6dc53856fe89160ad26da88d99a2a9328346815108c4102ae8f1f3dcb`,
  not HEAD. Never apply a HEAD-relative patch over prior Huber/quantile work.
  Correction: built-in ols_fit already publishes status; the unknown-status
  problem concerns custom/legacy callables that do not publish one, not OLS.
  FINAL PRIMITIVE NOW MERGED: core SHA
  `49356426a8a254fc0fda150da1d561954a847402f2958b6d6782e7a40b4ae217`,
  patch `94e7028f5d172c0c5980c0f1ecd03e2ec0117442330460be96b0245507296494`.
  Main receipt+pinball suite: 34 pass, 2 warnings,
  `/tmp/v9-m11-primitive-merged.xml`. It bounds/detaches details, sanitizes
  invalid scopes, separates unknown status, locks shared sinks, records
  pre-solver failures, and exposes `fit_failure_receipts`, `fit_receipt_scope`,
  `current_fit_scope`, `record_current_fit`. Still kernel opt-in only.
  Huber agent now implements phase B only in dynamic_regression public producer;
  v8_protocol handles phase A context/root/bridge plumbing in separate files.
- v9_wavelet: M35 unpenalized intercept and train-only centering. Preserve
  explicitly fixed trace-normalized additive model, describe blocked OOS loss
  comparison, NOT equal nested penalties or causality. Preserve M26/M18/M34/M36.
  Also repair malformed legacy nonlinear-dependence tests without swallowing errors.
  These are now merged. Agent is on additional extreme-unit hardening in
  `/tmp/v9-m35-scale-stage`. Latest candidate source 8062349f is NOT approved:
  root flagged constant train+constant test target log(0/0) incorrectly returned0,
  and dropping one constant feature column changes a multi-feature cross kernel.
  Await corrected finite-loss and constant-coordinate policy/tests. No main edits.
- v8_protocol: repair malformed legacy event tests (one argument for binary
  operators, swallowed exceptions) with real fixtures and non-vacuous assertions.
  Completed/merged; root extended all9 nonfinite roles (68 tests). Q03 real
  DuckDB 4-test oracle also merged, including execute/fetchdf counters in XML,
  but transfer bytes explicitly NOT_INSTRUMENTED. Latest task is M11 phase-A
  existing ExecutionContext/per-root owner/bridge transport wiring, stage only.
  IDs come from profile/artifact_plan/task; shared CSE remains factor-unowned.
  Avoid putting unpickleable locked sink across process spawn. Coordinate core
  transport API with huber agent; no worker-envelope or persistence changes yet.

## Still not closed

M11 full factor/window receipts, Q01 transitive complete certification, Q02 full
PIT and restart integration, Q03 real native/SQL/GPU execution, broker admission
for expensive kernels, v8 runtime/recovery gaps, approved-data auto80/100k
throughput and transfer ledger. Do not claim all 51 tasks or all 1,756 operators
closed. Existing production outputs have not been recomputed or deleted.

## Disk safety

Root filesystem has only about 6.5 GB free. Old `/tmp/v9-m37-stage` unexpectedly
contains a 339-GB whole-repo copy. Never create another broad staging copy.
Only one byte-identical 6,177,468,564-byte temporary dataset duplicate was removed
after comparison; the original lightgbm dataset remains intact. No other data
deletion authorized/performed. Use narrow candidate files and bounded evidence.
ENOSPC-interrupted historical XML attempts are not passes; later completed reruns
are recorded separately. Never print `.claude/loop/resume.md` (contains a secret).
