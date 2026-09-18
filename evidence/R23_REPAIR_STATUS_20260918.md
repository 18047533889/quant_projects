# R23 repair evidence — 2026-09-18

Worktree: server-c /home/sunhaiwei/quant_projects, main. No branch, repository copy, deployment or production-factor publication. User explicitly authorized reviewed commits and pushes.

## Current results

- DataAccess budget tightening validates original values before coercion. Commit 0f2a8648 was pushed to origin/main.
- Exact physical scope costing now conservatively prunes Parquet row groups only when footer statistics prove actual predicates disjoint. Full physical object scope and selected scan work remain separate. Unknown resident peak is explicitly unknown, not a fabricated hard RSS bound.
- Proven empty filtered scope can pass FactorEngine admission even when the snapshot contains files. Missing proof remains fail-closed.
- Nine group operators handle aligned string labels, finite support, ranks, integer inputs and declared fallback policies consistently.
- Normalize and group_normalize now avoid overflow for opposite-sign finite extrema in Pandas, native Polars, the Polars expression emitter and SQL; finite-span/subnormal arithmetic and their different singleton policies remain unchanged.
- Staged Polars correlation and regression R2 no longer multiply squared moments before normalization, fixing erroneous zero/NaN results at finite scales 1e100 and 1e-100. Regression slope's retval=r2 shares this repair; semantic versions updated.
- Persistence diagram and Fisher information Polars placeholders/proxies are replaced by explicit canonical reference delegates; semantic versions updated. These paths convert Polars → Pandas → Polars and are not native GPU/fusion speedups.
- R21/R22 pairwise staged kernels, finite-only numerical repairs, composite scope costs, held-buffer budget refinement and run_many controls are included in the combined acceptance.
- Auto Polars panel/long workspace admission now covers ts_corr, ts_cov and four regression outputs, including aliases/custom nodes and literal/keyword windows. Existing more-conservative Polars Long covariance allowance remains intact. These are planning bounds, not measured RSS guarantees.
- Registry freeze now creates one recursively immutable catalog shared by its live frozen state and snapshot, instead of deep-freezing twice. Public-read isolation and snapshot stability after thaw are retained.

## Tests (overlap; do not sum)

| Scope | Result | Evidence |
| --- | --- | --- |
| Budget tightening | 107 passed | r23-budget-validation.log |
| Combined acceptance | 572 passed, 20 skipped | r23-combined-acceptance.log |
| Streaming/CSE/sink/snapshot regression | 164 passed | r23-stream-cse-acceptance.log |
| Pairwise scale invariance and staged regression | 42 passed | r23-pairwise-scale-final.log |
| Final numerical backend parity shard | 233 passed, 20 skipped | r23-parity-release.log |
| Actual run_many extreme normalization with shared roots (Pandas/Polars/auto) | 3 passed | r23-extreme-numeric-batch-scope.log |
| Normalize extreme/subnormal/null boundaries | 9 passed | r23-normalize-extreme-expanded.log |
| Group normalize four-path numerical boundaries | 18 passed | r23-group-normalize-extreme-final.log |
| Group boundary regression after extreme fix | 41 passed | r23-group-normalize-related.log |
| Ordered legacy-surface regression | 64 passed | r23_surface_ordered_clean_watchdog.json |
| Markov/filter/group/topology shard excluding isolated cold-bootstrap check | 79 passed, 1 deselected | r23-operator-shard-no-bootstrap.log |
| Isolated dual-bootstrap consistency check | 1 passed | r23-operator-bootstrap-alone.log |
| Final footer safety including empty scope and BINARY predicates | 44 passed | r23-footer-ownership-final.log |
| Group identity/dtype boundary suite | 41 passed | r23_group_identity_dtype_pass_watchdog.json |
| Topology + Markov regression | 20 passed | r23-topology-markov-regression.log |
| Strict reviewed operator campaign | 132 of 1756 canonical operators; 264 backend slots with finite/parity/future-prefix pass across three reviewed ledgers | r23_operator_campaign/REVIEWED_EXECUTION_SUMMARY.json |
| Campaign evidence-cache correctness | 6 passed | r23_operator_campaign/test_campaign_evidence.py |
| Catalog single-freeze and existing isolation/governance regressions | 58 passed | r23-freeze-single-pass-regression-final.log |
| Final six-file workspace/auto/numeric acceptance with coherent numeric-test broker | 62 passed | r23-final-workspace-auto-broker-fix-6files.log |
| Workspace related acceptance | 72 passed | r23_operator_campaign/workspace_related_72.log |

The first six-file run had 61 passes and one CPU-admission denial (two requested CPU tokens, not a memory failure). The numeric test inherited a shared host-coordinator broker affected by soft-budget history/live pressure. Its fixture now injects an independent coherent broker while retaining real admission and exact source scope; production gates remain unchanged. The final rerun passed all 62 tests (82.141 seconds end-to-end, peak sampled family RSS 808534016 bytes).

A later combined rerun (r23-final-reviewed-source.log) reached the 240-second watchdog wall-time guard, peak 1253548032 bytes. It is not a pass; bounded shards are recorded separately. Diagnosis isolated repeated cold bootstrap (closure hashing / catalog freezing), not a demonstrated numerical-kernel deadlock. The 79-test non-bootstrap shard and isolated dual-bootstrap consistency check both passed; the latter took 221.671 seconds end-to-end with unchanged 90-second child timeouts. The first new auto numerical fixture lacked source scope metadata and was correctly rejected for unknown row count; its corrected bounded scope passed without weakening admission.

The campaign binds cached successes to recipe, fixture bytes, seed, backend class/source, semantic hashes and the campaign protocol source hash. Cache reuse requires both backend fingerprints, passing paired parity and explicit future-prefix invariance; single-sided changes invalidate the pair. Failures and untested entries remain retryable. The strict 110-recipe rerun supersedes the earlier 80-operator checkpoint: all four shards used zero cache hits, with 0 failed, 0 warmup-only, and current paired fingerprints. Shard elapsed seconds: 87.408 / 86.157 / 83.585 / 105.421; peak family RSS bytes: 762494976 / 762847232 / 762093568 / 762351616.

Twenty further market/OHLC/share-ratio operators passed with positive, domain-valid inputs and role-specific future mutations (76.943 seconds, peak 762019840 bytes). cs_regression and cs_resid also passed on 48 dates by 12 assets, with independent intercept OLS residual checks (66.199 seconds, peak 748019712 bytes). Their historical failures used only two assets, below the required three valid pairs; those failed records remain preserved. Combined reviewed coverage is 132, leaving 1624 manifest canonicals unverified, not implicitly certified.

The catalog-only 128-entry, 200-round microbenchmark measured duplicate freezing at 3.653432 seconds versus single freezing at 1.498718 seconds (2.438x); this is not an end-to-end startup speedup claim.

## Real run_many(auto) small-data checks

Window 2026-04-20..2026-04-24, eight securities, diagnostic sink only:

| Batch | Result | Batch seconds | Whole-command seconds | Sampled family peak bytes |
| --- | --- | --- | --- | --- |
| Shareholder | 3/3 EXECUTED | 152.294154 | 209.666361 | 776212480 |
| Minute relations | 16/16 EXECUTED | 80.920274 | 138.702469 | 2909380608 |

Evidence: r23-holder-output.summary.json / r23-holder-output.jsonl.gz / r23-holder-auto3.log; r23-minute-output.summary.json / r23-minute-output.jsonl.gz / r23-minute-auto16.log.

This supersedes R22's shareholder admission blocker for this window/sample. Footer metadata for the TopTen scope retains full scope 3890 files / 6287746016 bytes, while the tested date predicates select 5 files / 5 row groups / 432962 rows. Selected compressed bytes 9348172; two-column projection estimate 6927392. No predicate-free row discount or removal of ancestor workspace is used.

Watchdogs are sampled process-family guards, not kernel-enforced hard caps. Concurrent load and different samples prevent a claim of fastest performance or a controlled speedup.

## Remaining boundaries

- Registration/compilation is not numerical execution. Full 113893-factor execution and all canonical operators/backends/parameter domains are not established.
- The 20 skips are not passes. No blanket certification promotion was performed.
- Four late-loaded legacy-only kernels are explicitly classified only when their compatibility modules load; none were promoted to daily production authoring: ts_returns, ts_deviation_from_mean, ts_lag1_autocorr, ts_jump_bipower.
- Legacy cs_normalize remains outside reviewed canonical execution: no registered Pandas logical owner was found; Q contains an L1 formula whereas old SQL branches use min-max. It was not silently redefined or counted as fixed.
- Explicit reference delegates preserve correctness with conversion cost; no universal GPU coverage is claimed.
- No formal backend execute_multi_roots implementation was found at this checkpoint. native_fusion=True permits certified capability; it does not prove native multi-root execution.
- Bounded waves provide per-wave DAG/CSE and eligibility for bounded cross-wave cache under snapshot/budget proof, not unlimited global intermediate retention.
- Updated CSV was not regenerated from these partial tests. Existing R20 CSV remains a compilation artifact, not full execution certification.
- Default and optional calling examples: factor_engine/docs/R21_RUN_MANY_CONTROLS.md.
