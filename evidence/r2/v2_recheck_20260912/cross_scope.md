# V2 cross-scope acceptance evidence (2026-09-12)

## Evidence rules and this-run identity

- Formal tree: `/home/sunhaiwei/quant_projects`; observed base HEAD
  `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`; Python `.venv/bin/python`.
- Existing source and tests prove only that an implementation/test surface
  exists. Only the command below is counted as this recheck's execution.
  Existing markdown and prior logs were used for routing, never promoted to a
  fresh pass.
- No live database, private market snapshot, production write/publication,
  large benchmark, branch, worktree, commit, or push was used.
- Hardware discovery was read-only: one NVIDIA L20, driver 580.126.20,
  46068 MiB. The selected suite includes a bounded synthetic strict-CUDA
  capability-evidence test; this is not a full GPU parity or scale claim.

## Current per-item map

Status terms: **RUN-CODE** means relevant public/contract code was exercised on
small synthetic/local fixtures this run; **SOURCE-ONLY** means located but not
executed this run; **PARTIAL/BLOCKED** means a required cross-authority, real
data, scale, migration, or production condition remains.

| Item | Current status | Existing source/test surface located | This-run observation | Still required for full acceptance |
|---|---|---|---|---|
| QA-02 | RUN-CODE / PARTIAL | `quant_evaluator/scripts/generate_capability_matrix.py`; `tests/test_generate_capability_matrix.py` | declared-only, invalid, CPU bundle and bounded strict-CUDA evidence cases passed; unrun adapter/profile/admission/update/publication remain `NOT_RUN` | Generate and retain a matrix from the final stable public stack, real required profiles and per-metric CPU/GPU runs; FE bootstrap is still in flight |
| V-01 | SOURCE-ONLY / BLOCKED | package entrypoints, adapters and many boundary tests exist; current V2 owner evidence files enumerate scoped work | no exhaustive public entry→authority call graph was executed | Full CLI/HTTP/job/legacy/wheel/monorepo scan and owner-reviewed bypass disposition |
| V-02 | RUN-CODE / PARTIAL | FA taxonomy/production adapter; DA field-domain taxonomy | taxonomy/PIT fixture tests passed | Live candidate-ingestion regeneration, canonical catalog snapshot and re-evaluation update |
| V-03 | RUN-CODE / PARTIAL | FA metric→dimension→use-case policy and evidence invalidation | selected FA/FO policy-dependent tests passed | Historical forward calibration and production admission using trusted QE artifacts |
| V-04 | SOURCE-ONLY / PARTIAL | FO diagnosis routing/repair registry; FE compositional operators; FA lineage | bounded routing was covered indirectly, but no real parent/child search was run | Non-mock train/validation materialization and independent parent/child/recombined QE evaluation |
| V-05 | RUN-CODE / PARTIAL | `factor_optimizer/tests/search/test_v05_cheap_prefilter.py` | nonlinear/sparse preservation and bounded routing fixtures passed | Real sparse event/U-shaped populations and false-kill/multiplicity calibration |
| V-06 | RUN-CODE / PARTIAL | FP neutralization specs/diagnostics/output properties; QE exposure contracts; FO paired comparison | selected QE exposure contracts passed; existing FP focused run separately established local math | Paired OOS non-inferiority with common masks, real exposure availability times and portfolio exposure |
| V-07 | RUN-CODE / PARTIAL | FP frozen EWMA replay; FE stateful runtime surfaces | future-poison, daily/chunk/restart/new-asset replay tests passed | Enumerate every production filter and prove FE public incremental/checkpoint parity after FE registry stabilizes |
| V-08 | RUN-CODE / PARTIAL | FP four representation profiles and fitted-state apply; platform modeling-manifest bridge exists | representation profile suite passed | Real modeling consumers/folds, schema drift and tree/NN/linear training validation |
| V-09 | RUN-CODE / PARTIAL | FO diagnosis search budgets, stage trace and ledgers | bounded stage/cheap-prefilter tests passed | Atomic distributed reservation and a real non-mock search campaign |
| V-10 | RUN-CODE / PARTIAL | FO multi-fidelity identity/stage trace; QE evidence refs | stage trace and capability identity fixtures passed | Common formal sample recomputation plus full GPU tier telemetry |
| V-11 | RUN-CODE / PARTIAL | FO sealed-test broker, durable campaign/ledger and split contracts | sealed authority fixtures passed | External durable authority across workers/sessions and retained production campaign audit |
| V-12 | RUN-CODE / PARTIAL | FO split safety; DA PIT fail-closed contracts; QE sealed split | split and DA PIT fixtures passed | Revised filings, exchange calendars, suspensions, time zones and mature labels from versioned real snapshots |
| V-13 | RUN-CODE / PARTIAL | QE execution-trajectory adapter, turnover/cost/portfolio contracts | selected trajectory artifact tests passed | Actual holdings/cash/PnL, A-share constraints, borrow/capacity and cost data; absent inputs must stay `NO_CAPACITY_EVIDENCE` |
| V-14 | RUN-CODE / PARTIAL | FA exact/similarity evidence and incremental clustering | incremental clustering fixture tests passed | Full-corpus ANN recall audit and QE correlation evidence on versioned real windows/masks/costs |
| V-15 | RUN-CODE / PARTIAL | FA cluster lifecycle/lineage/refresh service | incremental assignment and lifecycle fixtures passed | Persistent production corpus refresh, stability calibration and modeling consequence validation |
| V-16 | SOURCE-ONLY / PARTIAL | FO winner sets/Pareto and FA selection decision exist | not separately targeted in this run | Trusted OOS incremental utility and real-library crowding/cost comparison |
| V-17 | SOURCE-ONLY / PARTIAL | FA composite specs/identity and platform feature-version tests exist | not separately targeted in this run | Materialize/evaluate a cancelling composite and prove production model input is unchanged until approved |
| V-18 | RUN-CODE / PARTIAL | FP recipe/fitted identities, QE artifacts, FA evidence refs, platform feature generation | selected evidence invalidation, representation and feature-generation fixtures passed | One resolver-backed model-column→definition→recipe→state→value→evaluation→cluster chain over a real snapshot |
| V-19 | RUN-CODE / PARTIAL | FP frozen EWMA replay; FE incremental scheduler/materialization surfaces | FP full-vs-daily/chunk/restart replay passed | Production daily service proving no FO call, revision generations, old-snapshot retention and atomic pointer rollback |
| V-20 | RUN-CODE / PARTIAL | platform GC root snapshot/dry-run/tombstone/reference authority; QE maturity GC | selected platform storage-GC and QE maturity-GC fixtures passed | Authorized DA/COS orphan deletion receipt on a bounded non-production fixture store, crash recovery, and retained research ledger proof |
| V-21 | RUN-CODE / PARTIAL | QE maturity streaming/window; FA immutable evidence/health refs | maturity streaming fixtures passed | Live post-release monitor over newly matured samples, data-quality alert routing and promotion freeze behavior |
| V-22 | RUN-CODE / PARTIAL | FO deterministic validator/whitelist/budget and research-only production capability | selected bounded-routing/sealed tests passed without an LLM dependency | Adversarial proposal suite through the actual platform Agent adapter; Agent remains non-authoritative |
| V-23 | RUN-CODE / BLOCKED | numerous package integration tests and QA capability generator exist | 267 selected boundary tests passed, including one small strict-CUDA evidence case | Stable FE registry, one real public FE→FP→QE→FO→FA→platform→model chain, private snapshot run and end-to-end CPU/GPU/I/O/transfer/resource benchmark |
| V-24 | RUN-CODE / PARTIAL | FA targeted metric-version invalidation plan; platform migration/GC tests; per-owner V2 evidence ledgers | targeted invalidation fixture passed | Final 97-item state ledger, complete affected-artifact inventory, migration dry-run/rollback and a green stable combined CI run |

## This-run command and exact result

Selected files covered QE capability generation/public artifacts/maturity/GC/
execution trajectory/exposure; FO cheap prefilter/stage trace/sealed split/split
safety; FA evidence invalidation/incremental clusters/lifecycle; FP frozen EWMA
replay/representations; platform storage GC/feature generation; and DA taxonomy/
PIT fail-closed behavior.

Result: **267 passed, 2 warnings in 5.21s**. Both warnings are pytest collection
warnings because production classes named `TestAuthorityBroker` and
`TestStoreRef` have constructors; they are not test failures, but should not be
confused with evidence that every class behavior was exercised.

## Global blockers that must stay visible

1. FactorEngine registry bootstrap was under independent repair during this
   recheck; FE-dependent all-stack execution is not certified here.
2. Synthetic/local fixtures do not establish private-data PIT correctness,
   statistical performance, production availability, capacity, or operating
   scale.
3. Presence of GPU hardware and one bounded strict-CUDA evidence test does not
   establish metric-wide parity, fallback absence, memory bounds, or throughput.
4. No production artifacts were invalidated, migrated, deleted, published, or
   repointed. V-20/V-24 remain partial until authorized dry-run and rollback
   evidence exists against the actual stores.
