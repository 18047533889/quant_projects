# R6 operator repair checkpoint — 2026-09-14

Status: IN PROGRESS. This is not an all-operators or production certification.

## Scope and preservation

All source edits are in server-c /home/sunhaiwei/quant_projects. Existing user/other-agent edits are preserved. No branches, worktrees, whole-repository copies, commits, pushes, deployments or production-factor publication were performed for this work. Subagents are GPT-5.6-sol.

Latest resource check: 772 GiB filesystem space available; evidence/r6 approximately 5–6 MiB. Test watchdogs sample process-family RSS and enforce timeout; their 2 GiB setting is NOT a kernel-enforced memory limit.

## Inventory checkpoint

remaining-current5.json: 1,756 canonicals; 287 aliases; 248 public recipes; 4,057 bindings. 1,464 research_callable, 292 remaining at that checkpoint. More changes may land later; refresh before using as a current count.

This inventory checks declared callable contracts, NOT numerical certification. production_callable=0 in the default snapshot means trusted linked production evidence has not been supplied; it does not mean all numerical implementations fail.

## Recent verified numerical changes

- elementwise-six-final2.log: 35 passed. Six scalar-bearing elementwise operators and envelope3. Real finite-overflow protection for round/truncate, negative-decimal native support, consistent finite-only cross-sectional winsorization, trim bounds, labeled dates, positional/keyword calls.
- rolling-ddof-final.log: 54 passed. Fifteen foundational rolling operators and consistent ts_std ddof=0/1 across backends. Invalid policies rejected.
- geometry-delegate-final.log: 6 passed. Numerical, scalar-binding, prefix and time-axis parity for final geometry delegates; explicit Pandas delegation, not native acceleration.
- daily-finite-history.log: 8 passed. Arg-extreme ages exclude Inf without changing physical bar ages; bucket finite support; exactly window prior bars for historical bucket; invalid quantile order/support rejected.
- extra-patterns-final5.log: 21 passed (18 pattern checks plus 3 bound-kernel identity tests).
- default-pattern-probe.log: 18 checks PASS after fresh default load_all, without pytest bootstrap hooks. Covers 12 supplemental patterns, confirmed triple-top/1-2-3 independent examples, quadratic score and unknown warmup. Same-name native placeholder methods replaced with actual sequential Polars/NumPy kernels.
- shared-regression6.log: 157 passed, 2 skipped. Optional independent bidask/diptest packages are absent; skipped tests are not passes.
- alpha47-final-registry.log: 47/47 frozen-registry Alpha-language numerical/keyword/prefix/invalid-parameter checks passed.
- turnover-survival.log: 63 passed after restoring real second-panel inputs and adding independent survival-weight reference.
- models33-final2.log: 35 passed for 33 model canonicals; models33-regressions-final2.log: 49 passed; models33-full-load-final.log: LOAD_OK.
- intraday_agg_repair.log: 7 passed covering 25 actual unique canonicals, including timezone/daily axes and full-day prefix.
- event_seasonal_repair.log: 8 passed covering 28 event/seasonal canonicals; fixed negative-slice look-ahead and NaN-as-event hazards.
- intraday_legacy_regressions.log: 25 passed with explicit session IDs/calendar and current ts_sharpe signature.
- state_ops_finalize.log: 9/9 remaining target schemas verified after full load; state_ops_schema.log: 78 passed.

Do not sum these counts as unique operators; suites overlap. Consult individual watchdog JSON files for commands, return codes, duration and sampled RSS.

## Performance and conversions

envelope-performance-ledger2.json uses synthetic 512 rows x 8 instruments, median of 3 repetitions. Input conversion is separately timed; output comparison tolerance is 1e-10. Not a 100,000-factor throughput measurement.

| Operator | Kernel speedup | Including input conversion | Max absolute difference |
| --- | ---: | ---: | ---: |
| ts_envelope_compression | 4.04x | 3.48x | 0 |
| ts_envelope_pressure | 22.42x | 21.20x | 4.2e-15 |
| ts_envelope_boundary_dwell | 57.62x | 47.59x | 0 |

Envelope tests prohibit internal Pandas/NumPy conversions during the native call. Generic geometry and several repaired model/relation backends remain explicit delegates. Polars-facing API, numerical success, native execution, GPU support and production certification are distinct.

Implementation identities include wrapper source and direct bound-kernel files. This is not a universal transitive dependency hash.

## Additional verified checkpoints

- direction-risk-existing2.log: 92 passed. Fourteen direction/risk operators; finite-support ratios, scale-stable entropy/RMS, and actual same-name Polars kernels.
- default-risk-probe4.log: 14 operators plus two nanosecond/timezone/slice checks PASS after fresh default load_all. Corrected hidden generic delegate selection and final misc-utils owner. Identity-aware kernels explicitly retain __fe_time__; legacy kernels retain existing stripping behavior.
- safe-six-final.log: 24 passed including risk coordinate and carry-forward policy regression. default-safe-probe.log: six operators, 12 backend checks PASS. Group median uses actual cross-sectional groups/minimum finite support; no arbitrary lineage forward filling; Polars NaN fill matches Pandas; extreme positions preserve physical bars/ties.
- cleaning-four-final2.log: 20 passed. default-cleaning-probe.log: four operators, 20 checks PASS. Fractional EWM alpha/span, missing/nonfinite observations and unbiased alpha=1 undefined output agree; finite fill scalar contracts and real NaN handling.
- interval-six-final2.log: 14 passed (eight new checks plus six existing delegate checks). Six actual interval canonical contracts; removed fake volatility/spread proxy methods from native-class slots. Mode-distance profile now uses exactly W prior bars, not W-1. These backends are explicit Pandas delegates, not native acceleration.
- shared-regression7.log: 201 passed, 2 skipped (missing optional bidask/diptest independent checks); discovery-regression3.log: 68 passed.
- pattern-all-final.log: 64 passed; default-pattern-probe-r6.log: 18/18 PASS. fallback-parity-final2.log: relation_hhi/group_peer_information_diffusion two checks passed.
- alpha_index_numeric_final.log: seven tests covering 14 final-registry operators; alpha_index_full_regression.log: 151 passed. Fixed genuine entry_index_a/b/c and listing_age keyword mismatches.
- daily3-argext-final.log: ten independent final-registry checks PASS, daily3-argext-pytest.log: two passed. daily3-topksum-final3.log: seven independent checks PASS. Native window/min_periods handling and final layer_topk_compat owner synchronized; strict guard retained.
- Intraday final five and A-share limit six source groups eliminated from latest targeted inventories. Watchdog evidence for these batches records commands and actual coverage; real market/price-grid context still required.

Risk-safe-performance.json is a synthetic 128 x 4, median-three conversion ledger. All six tested kernels match within 2.3e-16 and prohibit internal Pandas panel conversion. At this SMALL shape, including input conversion, Polars speedup is 0.12–0.89x (slower than Pandas). Do not claim that native execution always accelerates, or extrapolate this to 100k factors.

## Latest structured-parameter and backend checkpoint

- remaining-current6.json: 1,756 canonicals; 1,522 research_callable; 234 remaining. Contracts only, NOT numerical or production certification.
- structured-parameter-shared.log: 39 passed. Real recursive sequence and union ParamSpec validation; same planning/runtime/hash gate; legacy unspecified containers remain unchanged.
- structured-discovery-final.log: 132 passed, including discovery/r30/r40, structured contracts and semantic-version checks.
- regression-model-existing2.log: 65 passed; default-regression-model-probe.log: seven models / 13 checks PASS. True pinball slope backend and finite-scale dimensionless models; original level-score minimum window 20 retained. Neighboring unregistered classes preserved/restored and full bootstrap succeeds. Six newly corrected models have semantic version 2; quantile slope already had version 2.
- structure-five-final3.log: 55 passed; structure-five-default.log: 11 passed with --noconftest and fresh default load_all. Replaced Bures all-NaN and KM/group variance proxies with explicit reference delegates; corrected scale-dependent correlation rejection and genuine per-bin KM support. Shared KM maturity remains expanding max-lookback with 10 finite historical observations; no fabricated full-window claim.
- daily19-cs-bucket-historical-final3.log: 19 checks PASS. Real tuple/list quantile contracts; exact W prior bars and strict increasing interior quantiles. Final Polars interface still honestly delegates where selected.
- relation-distribution-all-backends2.log: 6 passed; relation-distribution-all-backends-regression.log: 89 passed. All six final Pandas/Polars bindings plus actual DuckDB SQL for group_quantile_spread. Fixed lost variadic topology and recursive list/tuple panel conversion.
- state-space6-target.log: 3 passed; state-space6-regression.log: 49 passed. Six final Kalman bindings, genuine distinct defaults, full-history state semantics, independent recurrence/beta checks.

## Latest verified numerical and default-loading batches

- rotation5-final2.log: 5 operators / 50 checks; rotation5-pytest.log and rotation5-bootstrap.log pass. Real defaults, optional group panel and strict lag/quantile validation.
- extreme-tail5-final5.log: 5 operators / 53 checks; extreme-tail5-pytest-final2.log: 32 passed. Research beta is explicitly research-only; production exclusion assertion remains.
- layer-topk-weighted10-final3.log: 10 operators / 90 checks; topk-pytest-normal.log and topk-pytest-noconftest.log pass. True defaults/topology and top-k support on both bootstrap paths.
- cs4-final3.log: 35 passed; cs4-default.log: 13 passed (--noconftest). Actual isolation/bucket/Bayes/group estimators replace fake backend formulas. Zero standard error is valid; finite scale and missing groups handled.
- weighted5-final2.log: 27 passed; weighted5-default.log: 12 passed. Final polars_chip_tail NumPy kernels now use exact floor strata, positive weight mass without absolute epsilon, scale-stable downside and physical price-gap drawdown reset. No internal Pandas-panel conversion in the tested five native paths.
- dc4-fastpath-regression3.log: 46 passed; dc-eventv2-default-final.log: 18 passed (--noconftest). DC four now have actual per-column NumPy state kernels, stable threshold clocks/asymmetry, monotone completed-leg bounds and requested-event-rate fastpath. Whole-history replay semantics retained.
- eventv2-five-final.log: 147 passed. Event/state-v2 five have correct required panels and NumPy Polars kernels; economic windows remain required, never invented.
- state-event5.log and state-event5-noconftest.log: 11 passed each; state-event5-regression.log: 35 passed / 89 deselected. Source-class metadata removes bootstrap-order dependence.
- flow-impact5.log: 11 passed. flow-impact5-regression-fixed.log: 61 passed / 4597 deselected / no skips. Repaired true panel fixtures and explicit research/legacy-reference test paths; tests of a deliberately unregistered legacy kernel are NOT new Agent availability.
- sequence-complexity-final4.log: 5 passed; sequence-complexity-defaults-final.log: 1 passed. Corrected weighted entropy, DFA amplitude dependence, Higuchi normalization and broken final backend slots.
- wavelet5-ordinary-focused-final2.log: 47 passed; wavelet5-noconftest-final2.log: 7 passed. Haar/FFT scale-normalization retains exact windows, full support, frequency weights and prefix causality.
- shared-regression8.log: 173 passed / 2 skipped. Missing independent optional checks remain visible, not counted as successes.
- feature-geometry4.log and feature-geometry4-noconftest.log: 13 passed each. Four final feature-geometry contracts/backends repaired; robust correlation inputs are scale-normalized before subtraction so near-float-max finite data do not overflow. Beta estimator and rotation eigen-gap semantics remain explicit.
- nonlinear-dependence4.log and nonlinear-dependence4-noconftest.log: 16 passed each before the MI identity split. Fixed-mass tail membership now exactly allocates q*n mass with fractional boundary ties; tiny finite scales no longer fail an absolute-epsilon standard-deviation gate.
- r41-mi-spectral-authority-fixed-final.log: 69 passed. MI is split into `ts_mutual_information_nats` and `ts_normalized_mutual_information`; legacy `ts_mutual_information` resolves to normalized and rejects the removed unit-changing `normalized` switch. Spectral legacy spelling remains a live typed canonical and direct calls require explicit input kind.

weighted-tail-performance.json measures 160 rows x 4 symbols, median-three. Including input conversions speedup is 0.92–1.03x, maximum difference 0. This is parity/small-shape evidence, NOT 100k-factor/GPU throughput.

Semantic identities were bumped for changed numerical definitions (DC v2; wavelet five v3; weighted downside/semivariance/drawdown v3; other repaired definitions as recorded in operator_semantic_version.py). This prevents silent reuse under an unchanged declared semantic version; it is not production certification.

## Latest memory, return, robust and rule-language evidence

- memory4-final.log: 46 passed; memory4-default-final2.log: 14 passed. Exact fractional discarded absolute mass replaces the finite 50,000-term approximation; finite-scale convolution and huge-cutoff/no-support early return are tested. ACF/FD history requirements are explicit. Final Polars memory slots honestly delegate to Pandas.
- returns4-final2.log: 46 passed; returns4-default-final3.log: 13 passed. Four return-decomposition operators use the actual two price panels, explicit price-basis validation and per-column NumPy. Pre-close is not shifted twice and supplied VWAP is not replaced with a volume proxy. Tests prohibit Pandas-panel conversion on native calls.
- robust4-broad-final3.log: 115 passed. robust4-default-allpaths-final2.log: 24 passed after fresh default bootstrap, including actual DuckDB SQL and Polars planned execution. Quantile planned-path support now respects min_periods and original integer literal types.
- robust4-engine-strict-final.log: 3 passed, 12 deselected. Real engine Pandas/Polars/SQL parity is enforced; failures cannot be hidden by Python fallback. DuckDB robust std normalizes each trailing window before variance. DataAccess intentionally rejects convenience macros; the emitter now uses equivalent built-in list_aggregate without weakening the SQL guard. MAD remains honest SQL fallback. No large-N/window memory guarantee follows from these small tests.
- stateful-rule4-ordinary-focused-final2.log: 43 passed; stateful-rule4-noconftest-final2.log: 11 passed. Latch/hold/slew/deadband expose scalar defaults and actual panel controls; recursive full-history declarations are explicit. Invalid scalar controls reject; dynamic invalid panel cells retain break/reset semantics.
- gather-ext5-broader-final.log: 91 passed; gather-ext5-noconftest-final3.log: 9 passed. Real topology, weighted fractional top-k/rank scale handling, and per-cohort permanent unknown-path censoring. These final Polars slots are explicit Pandas delegates.
- event-interval4-operators-final.log and event-interval4-operators-noconftest-final.log: 30 passed each. Finite invalid EventBool is rejected even in immature/discarded blocks; bounded interval history is explicit. Legacy Inf-as-unknown policy is preserved.
- shared-regression9.log: 205 passed, 2 skipped (optional independent packages absent). Suites overlap; do not add test counts as unique operators.

return-robust-performance.log measures 128 rows x 4 symbols, warmup one, median three, including input AND output conversions. All eight outputs have maximum difference zero. End-to-end speedups: returns4 0.13–0.14x; quantile range 1.01x; trimmed mean 0.80x; inclusive/prior robust score 0.92x/0.89x. This is a CPU microbenchmark, not a claim of universal acceleration, GPU support or 100,000-factor throughput.

## Latest vector, dependence and state/model checkpoint

- remaining-current11.log: 1,757 canonicals, 288 aliases, 4,059 bindings; 1,653 research_callable, 104 remaining. This is declared-contract coverage, not universal numerical or production certification.
- vector4-default-final.log: 14 passed; vector4-ordinary-final.log: 45 passed. Exact two-panel geometry replaces one-panel scalar proxies, with scale-normalized coordinates, correct proper intersections and no Pandas-panel conversion on final native paths. Full contiguous-window and degeneracy rules remain. Test collection uses importlib because the tree has two same-basename operator test modules.
- vector-performance.log: 128 x 4, median three including input/output conversions, speedups 0.88–0.99x and max difference zero. Native CPU is not universally faster; no GPU/100k-throughput claim.
- dependence-cte-broad-final.log: 93 passed; dependence-cte-default.log: 15 passed fresh/default. Real Xi/HSIC/CMI/partial-distance-proxy backends and conditional TE replace unrelated or empty placeholder kernels. Scalar grids/domains, actual panels, finite-scale normalization, sample support and original physical lag are tested. Conditional TE remains research-only; Pearson-form partial distance correlation remains explicitly a proxy, not strict pdCor.
- hsic-performance.log: independent dense-matrix reference vs algebraic mean-centering at windows 60/120/240, one BLAS thread, median three. Kernel speedups 1.13/2.07/3.11x; max difference 2.26e-17. Centering changes O(W³) compute to O(W²); both retain O(W²) scratch. This excludes panel conversions and is not a full run_many benchmark.
- robust4-final32.log: 32 passed including original integer-literal preservation and explicit bad-parameter plan rejection. identity-discovery10-final.log: 85 passed; historical merge versions are minimum floors so subsequent valid semantic bumps do not fail old equality fixtures.
- exself4-existing-final.log and exself4-noconftest-final.log: 50 passed each; exself4-oracle3.log adds independent dual-backend scale/default/topology proof. Missing groups and singleton peer sets are handled; scale-dependent absolute floors and overflowing totals are removed.
- stateful-sequential4-ordinary-final.log: 42 passed; stateful-sequential4-noconftest.log: 12 passed. CUSUM/EWM have explicit recursive full-history contracts; rank/peak-lag have finite/compound histories. Real final Polars slots are labelled delegates, and CUSUM is dimensionless.
- stateful-survival3-ordinary.log: 43 passed; stateful-survival3-noconftest.log: 8 passed. Explicit episode/full-history contracts, true scalar defaults and strict domains. Gap fixture now starts confirmed inactive so its two completed runs really exist; original gap NaN/post-gap 0/1 assertions remain. Left/gap-censored runs never enter completed-episode history.
- ar-meanrev4-final.log and ar-meanrev4-final-noconftest.log: 13 passed each; ar-meanrev4-regression-focused-final.log: 12 passed. Independent AR/OU/variance-ratio oracles and 1e±300 scale checks; true defaults, prior history and feasible parameter relations. Final delegated backends remain honestly labelled.
- binned3-existing2.log and binned3-noconftest.log: 20 passed, 16 deselected each. binned3-oracle.log adds independent dual-backend scale proof. Remaining inventory excludes all three.
- New semantic versions for corrected vector, dependence/CTE, sequential, survival, AR, binned and ex-self/group definitions are recorded in operator_semantic_version.py; this is cache identity, not production admission.

## Fresh-load, structural, group, jump and expectile checkpoint

- remaining-current12.log: 1,757 canonical names, 1,665 research_callable, 92 remaining contract gaps. Rough-vol and expectile completed afterwards; refresh before quoting a newer total.
- common-cs3-existing.log: 2 passed, 3 legacy rollout skips; common-cs3-noconftest-final.log: 2 passed; common-cs3-oracle.log verifies independent scale/default behavior. Skips remain skips.
- common-group3-existing-final.log: 30 passed, no skips; common-group3-loadall.log and common-group3-oracle.log verify final contracts and numerical behavior. Correct Polars fallback forwarding and strict two-panel alignment. Duplicate-window tests now accept the central typed rejection without weakening it.
- evt-allan3-final.log and evt-allan3-final-noconftest.log: 11 passed each. Real binary event topology, dimensionless scaling and feasible default windows.
- multiscale-trend3.log and multiscale-trend3-noconftest.log: 13 passed each; multiscale-trend3-focused-regressions.log: 135 passed, 2 optional skips. Retired untyped DMD stays internal; the test uses the live return-typed replacement.
- rough-vol2.log and rough-vol2-noconftest.log: 12 passed each; rough-vol2-existing.log: 18 passed. Log-domain power means prevent high-p overflow/underflow, preserve physical gaps and existing partial-valid-scale semantics.
- structural3-history-default.log: 12 passed with fresh load_all; structural3-broad.log: 66 passed. Nominal-window coverage, safe log ratios and explicit full-history pivot state. pivot-query-performance.log: 1,000 queries / 120-window at history 1,000 and 5,000; indexed queries 2.66x/16.29x faster, exact event parity. Excludes ledger construction/conversions; full ledger still O(N) storage.
- moments3-fresh-load-noconftest.log: 8 passed after repairing final registration. This SUPERSEDES the earlier target-module-only 5-second run, which was not full-load proof. Both moments and jump final owners are explicit Pandas delegates.
- jump-robust3-fresh-load-noconftest.log: 10 passed; jump-robust3-ordinary-focused.log: 27 passed. Default 240, real return panels, stable degree-two MedRV/MinRV scaling and BNS invariance. Broader stale session-recovery fixture now explicitly sets min_events=1 for its one-event geometry check; default min_events=3 remains. session-recovery-round3.log: 25 passed.
- repaired-batch-integration-first.log: 2 passed, actual run_many on Pandas/Polars-long, three repaired vector/dependence/structural factors delivered once to an in-memory sink and numerically matched. Resource snapshot is a coherent test double; not live-host 80% admission or durable COS/100k throughput evidence.
- expectile2-default-final.log: 8 passed with fresh load_all / noconftest; expectile2-broad-final.log: 62 passed. Exact per-column CPU kernels use canonical tau=.1/n_min=3/window=60, proper y/x topology and axis checks; no Pandas-panel conversion. Independent score-root / asymmetric-loss oracles, scale/prefix and invalid-parameter tests. Old 1e-12 predictor perturbation is representable, not singular: the stale test now checks exact two-group OLS slope and retains true constant-design rejection.
- Integer semantic versions updated for real contract/backend/definition changes, including group3/multiscale3=2, jump statistic=3, MedRV/MinRV=2, rough-vol2=3 and expectile2=3. Artifact contract versions are distinct; no double-bump just because another worker saw this newly updated ledger.

## Fiscal, filters, spreads and weighted moments checkpoint

- remaining-current14.log: 1,757 canonicals, 1,692 research_callable, 65 remaining declared-contract gaps. This is NOT full numerical certification of 1,692 operators.
- shared-runtime12.log: 153 passed across identity, schema, true run_many, vector/dependence/structural/expectile/despike integration. Peak sampled process-family RSS 492,761,088 bytes.
- fiscal3-ordinary-focused-final.log and fiscal3-noconftest-focused-final2.log: 12 passed, 18 deselected each. fiscal3-order-consistency.log independently proves common-first vs full-load owners/contracts match. Fixed a real load-order problem: shortened common x/window registration previously prevented authoritative value/period_id contracts from registering. No conftest change; related missing-index comparisons only reflect Polars conversion dropping the original index.
- multifractal3-final.log and multifractal3-final-noconftest.log: 9 passed each; multifractal3-existing.log: 13 passed. Normalization preserves the original non-compressed gap cohort and partial-q semantics.
- multifractal-asym1.log and its noconftest run: 7 passed each; threshold-cycle2.log and its noconftest run: 9 passed each. Thresholds remain required economic inputs, not invented defaults. Exact cycle asymmetry no longer has an epsilon bias.
- first-passage2-final-ordinary-focused.log: 16 passed; first-passage2-final-noconftest.log: 7 passed. Only hit probability/conditional time repaired; the separate bias canonical is not covered by this batch.
- despike3-default-first.log: 13 passed. Exact past-only Hampel, inclusive rolling median and finite seed behavior replace late placeholder formulas. Long-double intermediates avoid overflow of finite median/MAD/covariance calculations on the server. scale_floor retains units of x; rescaling x requires rescaling this explicit floor.
- filter2-oracle.log and filter2-loadall-noconftest.log: independent recursive filter and final-owner proof; filter2-ordinary-focused-final.log: 23 passed. KAMA/Super Smoother remain full-history and not checkpointable. Mirrored IIR tests now respect actual overshoot and absent checkpoint adapters, not fictitious monotone step responses.
- spread2-default-final.log: 8 passed; spread-despike-broad-final.log: 321 passed across both test mirrors and related market/filter suites. Safe log-ratio/tanh CS arithmetic, exact log-price covariance Roll, strict pair support, real high/low inputs; old Polars mean-absolute-return proxy replaced. Roll retains atomic invalid-price column semantics and covariance>=0 -> NaN. Existing DirectUse promotion can label it extended; surface visibility is not production numerical certification, which was not added.
- weighted-moment2-default-final.log: 8 passed; weighted-moment2-broad-final.log: 122 passed. Real weighted panels and conditional sample covariance replace unrelated backend formulas; strict ConditionBool now rejects Inf. Complete feasibility/default contracts; covariance uses wide intermediate precision and returns NaN for unrepresentable final values. Final Polars paths are honest Pandas delegates; no native/GPU claim.
- cross-local2-ordinary.log: 14 passed; cross-local2-oracle.log and cross-local2-contract.log pass for the two global copula summaries. Full-load/global-state review is being checked separately before closing that scope.
- intraday-session-activity-existing.log: 40 passed; intraday-three-noconftest-focused.log: 8 passed; intraday-three-final-backends.log proves final full-load panel/scalar contracts. Shared calendar-schema changes and a separate min_events stale fixture require follow-up integration review.
- expectile-despike-performance.log: 128x4 CPU, median three with both input/output conversions, max difference zero. End-to-end speedups 0.80–0.96x; these small panels are slower via Polars conversions, not universal acceleration or evidence of 100k-factor/GPU throughput.
- Integer semantic versions reflect actual prior ledger entries: first-passage hit/time 2->3, Super Smoother 2->3, weighted standardized moment 2->3; newly unversioned fiscal/multifractal/despike/spread/copula/intraday contracts and covariance are version 2. No duplicate bump for changes another worker has already seen.

## Crossing, distribution, calendars and additional exact backend checkpoint

- remaining-current15.log: 1,757 canonicals / 1,705 research_callable / 52 remaining. Later calendar, path, slice and other repairs supersede this count only after a new inventory; contract completeness is NOT numerical certification.
- crossing2-default-first.log: 10 passed; crossing2-broad-final.log: 50 passed. True x/y panels, wide arithmetic, unknown adjacency -> NaN rather than invented no-crossing zero. Dedicated per-column NumPy CPU implementation, not a Pandas delegate.
- distribution3-default-first.log: 7 passed; distribution3-broad-final2.log: 51 passed. Actual three energy features / two copula panels, independent clipped-U distance and empirical-copula oracles. Strictly-past reference windows, feasible estimator support, scale-safe long-double distances and robust-z. Final Polars implementations honestly delegate to Pandas; energy scratch is quadratic in window length. Both legacy test mirrors now use actual panel topology and no longer swallow deterministic-test errors.
- cross-local2-loadall-noconftest.log: fresh-load proof of global_state role; same-date values are identical across securities, not terminal cross-sectional alpha.
- shareholder3-ordinary.log: 26 passed; shareholder3-loadall-noconftest-final.log: final contracts/numerical/Polars proof. Real required/optional holder panels and report-date availability; deprecated operators retain their status.
- intraday-limit2-focused.log and intraday-limit2-noconftest.log: 13 passed each; final-backends log covers real final slots. intraday-round3-noconftest-final.log: 15 passed. Single-event private geometry may use min_events=1; public recovery operator retains min_events>=3.
- event-response2-physical-final.log: 9 passed; event-response2-engine-only2.log: 1 passed through actual FactorEngine run/run_many Polars-long. These two event kernels are NumPy CPU, not GPU; no lazy/streaming capability claimed. Peak sampled engine test RSS 790,228,992 B.
- session-recovery1-fresh-load-final7.log: 6 passed; session-recovery1-ordinary-final2.log: 20 passed. Exact daily Pandas delegate, strict public support, session-close availability.
- calendar-concrete-type.log and remaining-after-calendar-type.log: three intraday calendar contracts now use actual SessionCalendar, not arbitrary object. Dict/string/opaque substitutes rejected; real calendar accepted; actual inventory excludes these three.
- snapshot-calendar-integration16-final.log: 103 passed. Snapshot, calendar, bound-kernel and public semantic identity integration. Earlier 15 failures were a stale version==2 test; it now checks the established minimum AND exact current public version propagation, retaining implementation fingerprint checks.
- extrema2-state-density-final.log / noconftest: 5 passed each; existing: 18 passed. Exact two-panel extrema, support relations and scale-invariant state density (including 1e-200). state-density-physical.log identifies genuine NumPy CPU kernel, not a GPU claim.
- marked2-default-final.log: 6 passed; marked2-broad-first.log: 77 passed before the final unknown-event-gap addition. Event indicators reject Inf, real mark/panel defaults and strict controls, stable correlations. Unknown event observations now break event-index adjacency for both mark policies; final fresh tests cover this. Final Polars is an honest Pandas reference delegate.
- information-rank2-default-first.log: 5 passed. Replaced false EMA/linear-rank backend formulas with true two-panel exponential average-tie ranking. Stable tied weights with subnormal decay and finite large targets; Decimal first-digit Benford oracle covers float64 subnormals. Both final CPU paths avoid Pandas panel conversion. Benford remains research-only and is not a fraud probability.
- drawdown-path2-fresh-final.log: 5 passed; drawdown-path2-ordinary.log: 37 passed. Returning to the old peak now resets drawdown area. Strict windows and exact Pandas delegates, no native/GPU claim.
- path-signature3-ordinary.log: 28 passed; path-signature3-strict-final.log: 18 passed; path-signature3-loadall-noconftest-final.log verifies true final loading/Polars/oracles. The temporary R5-02 registration error was fixed; target inventory excludes all three.
- slice3-focused.log and slice3-noconftest.log: 23 passed each; slice3-polars-parity.log verifies daily axis and numerical equality for all three. Corrected round-price clustering minute-to-daily route; actual inventory excludes targets.
- Integer versions updated once against the actual ledger, including event-response 2->3 and new crossing/distribution/session/shareholder/extrema/marked/rank/drawdown/path/slice entries at 2.
- Latest disk check: 776 GiB available; evidence/r6 17 MiB. No whole-tree copies, branches, commit, push or production-factor publication.

## Default/schema, mixed inputs and final CPU kernels checkpoint

- remaining-current23.log: 1,757 canonicals, 1,748 research_callable, 9 remaining contract gaps. This is contract completeness only; executable_unverified remains 1,748 and production_callable remains 0.
- semantic3-default-final2.log: 6 passed. semantic3-full-regression23.log: 52 passed, including envelope native conversion-forbidden tests. Product uses exponent/mantissa multiplication; MAD uses wide median/deviation; nullable min_periods relation is valid. Group percentile honestly uses a Pandas delegate.
- information-rank2-broad-final.log: 117 passed; fiscal-period2-default-first.log: 4 passed; fiscal-period2-broad-final.log: 34 passed. Fiscal revisions preserve exact period/availability semantics with explicit full-history replay, no fictitious finite row bound.
- component1-default-first.log: 5 passed; component1-broad-first.log: 21 passed. Real optional component slots, direction/weight sequences, cancellation-safe sums. Native CPU Polars preserves axes; Pandas effective-count attrs are NOT transported as a Polars receipt.
- mixed power/coalesce: elementwise-mixed2-loadall.log fresh schema/DSL run_many pass; elementwise-mixed2-pytest-final.log 19 passed. Runtime now distinguishes panel/scalar/nullable variadic arguments and hashes their real contracts; this is not generic object acceptance.
- shared-runtime20.log: 150 passed after mixed infrastructure stabilized.
- MACD/RSI focused 81 passed, 20 explicit legacy surface skips, not missing GPU/DuckDB dependencies. Actual checkpoint follow-up macd-rsi-checkpoint-recheck.log: 1 passed. ts_quantile fixes integer truncation, late-owner window alias and physical NaN handling; quantile-fresh-load-tests-recheck.log: 1 passed.
- group-state3: fresh 7 passed / ordinary 53 passed; candle-state2: fresh 7 passed / ordinary 33 passed; stateful-events2: fresh 5 passed / ordinary 47 passed. Exact delegates and real panel topology, including candle prior-window off-by-one repair.
- Gaussian/days: gaussian-days3-final.log 56 passed / 1 ADX frozen-contract integration failure under investigation. New Gaussian/days focused cases all passed: true NumPy average-tie normal scores; strict ConditionBool with timestamp columns excluded from validation; unlimited days-since remains full history.
- kurt23-fresh-final.log: 3 passed. Removes contradictory constant-window -3.0 backend overrides and overflow-prone moments; shared scale-stable Fisher estimator, constant NaN, full finite physical windows, default 20 and strict window>=4. Fresh tests cover independent SciPy oracle, 1e-300/1e300 scales, default/final backends and both actual duplicate class definitions without side-effect imports. Broad SQL/engine retest pending.
- StaticAdjacency graph phase A: 12 focused tests passed. Real N-by-N directed adjacency, own-node exclusion, threshold filtering and exact aggregates, typed content-hash identity. Direct Python call only; dynamic time-varying graphs and DSL matrix literals are NOT implemented. Related older tests still being repaired against required panel contracts.
- impulse-dominant-contract.log: 7 passed, including independent sinusoid period oracle and real minute-to-daily event aggregation. Full fresh suite rerun pending after delegate default repair; dedicated daily-axis fresh case already passed.
- rank-semantic-performance20.log: 96x4 CPU, median3 with input+output conversion, maxdiff0. Polars end-to-end ratios score-rank 0.771x / product 0.718x / MAD 0.843x; no universal speedup claim.
- Integer versions updated once against current ledger: protected_div and ts_kurt 2->3, new static graph/Gaussian/days/impulse/dominant-cycle/LQTP/mixed divisions version2. Previous component3, group_percentile4 and MACD3 remain.

## Full callable-contract checkpoint and genuine batch bug repair

- remaining-current25.log: canonical1,757 / research_callable1,757 / executable_unverified1,757 / production_callable0 / known_failed_bindings0. The original remaining-contract list is empty. This is NOT all-operator numerical certification.
- lqtp23-fresh-final.log: 9 passed including Gaussian/days, fresh load_all without conftest. lqtp23-broad.log: 11 passed. Fixed late CVaR backend date-axis loss, actual default SMA/CVaR, strict domains and wide arithmetic. Physical SMA correctly marks stateful and still requires full-history replay.
- kurt23-broad-final.log: 22 passed across actual engine pandas/polars_long and SQL moment tests. Additional SQL review found overflow/underflow of powers: normalized offsets plus opposite-sign overflow recovery now match independent normalized SciPy oracle. kurt24-sql-scale.log: 15 passed including extreme scale/constant cases.
- shared-runtime24.log: 167 passed across repaired runtime contracts, numerical kernels and earlier batch integration.
- Extending the true batch test to eight default operators exposed a genuine planner bug: lower_root_plan added consumer-only root edges to all earlier barriers, although root inputs named only the final stage. Fixed reciprocal chain construction and removed insertion-order fallback for invalid DAGs in both critical-path helpers. dag-batch25-final2.log: 16 passed, including both real batch sink backends, nested-barrier edge/critical-path/cycle tests and shard/lease regression. Test broker now uses a coherent NORMAL snapshot with CPU capacity matching the immutable Polars pool; this does NOT bypass production admission or test live-host pressure.
- tail-contracts-ordinary-final.log: 132 passed. Cross-sectional tests now use actual panels with independent analytical oracles rather than ignored missing-input failures. tail2-static-fresh-noconftest-final.log: 1 passed through full load and real graph/tail calls. StaticAdjacency strictly normalizes integer/string symbols and rejects bool/string weights; typed content hashes distinguish int1 from str1.
- last3-final-ordinary.log and last3-final-noconftest.log: 7 passed each, 7 deselected. True row_sum variadic mixed inputs, attraction/diffusion independent oracles and actual polars_long run/run_many. Physical row sum is NumPy CPU; attraction/diffusion are honest Pandas delegates.
- hump-trade-corrected-ordinary-final.log and hump-trade-corrected-noconftest.log: 5 passed each. trade_when missing condition selects fallback consistently with if_else; nonzero including Inf selects signal. This remains pure select, not WQ position-state semantics. hump permits nonnegative finite threshold including0 and has explicit stateful/full-history physical semantics.
- default-kernel-performance25.log: 96x4 CPU, median3 including both conversions; maxdiff0 for kurt/SMA/CVaR/Gaussian/days. End-to-end Polars ratios 0.512 / 0.435 / 0.785 / 0.574 / 0.166 respectively. No universal speed/GPU/100k claim.
- The seven final signal/cross-section/row/group contracts received integer semantic version2, without production certification changes.
- operators-wide26.log: 470 passed, 6 failed across two mirrored SQL-delay tests. SQL ts_delay invalid window rejection is being repaired, not skipped.
- runtime-wide26-first-error.log found a collect-time import-order conflict: early time_semantic_ops same_clock_lag registered clock_time/lag_minutes versus final canonical lag/clock_unit. The repeated-failure broad run was intentionally interrupted after identifying this shared blocker; alpha owns the root-cause repair. Passing targeted snapshots are not claimed to supersede this broader failure.
- Cleaned 15 completed temporary patch files on each endpoint. Current evidence/r6 is26MiB, free disk776GiB. No branches, tree copies, commit, push or deployment.
- R6_DEFAULT_USAGE.md provides a tested research-call template and concrete input/state/graph/performance limits.

## Final verification checkpoint 29

- remaining-current29.log: canonical 1,757, research_callable 1,757, implementation_available 1,757, known_failed_bindings 0, remaining groups empty. executable_unverified remains 1,757; production_callable remains 0. The contract-gap inventory is closed, NOT independent numerical certification of every operator.
- runtime-wide29-final.log: 802 passed, 0 skipped, 2 warnings. operators-wide28-final.log: 499 passed, 0 skipped, 6 warnings. These are test-case counts, not distinct-operator certification counts. Operator-wide ran before the final dynamic-regression bootstrap repair; that final repair is covered by the runtime-wide and fresh tests below.
- dynamic-regression-polars-final.log and dynamic-regression-polars-final-noconftest.log each pass the actual 24-backend positional/keyword numerical comparison. dynamic-regression-full-final.log: 43 passed including independent least-squares oracles. Final gap-coverage bootstrap now replaces exactly 24 legacy single-series pseudo-native slots with real multi-panel Pandas delegates; source-derived identity and truthful CPU/conversion declarations retained. The old module-name skip is removed.
- SQL ts_delay alias validation and actual kernel argument consumption are repaired for d/window/lag/periods; invalid aliases fail and valid aliases no longer silently select default lag1. The final operator-wide suite includes these regressions.
- The same_clock_lag early conflicting registration is removed; final authority retains lag/clock_unit. ADX complete frozen-contract comparison and per-operator true physical-kind expectations are repaired. Earlier broad SQL/bootstrap failures are superseded by the passing final suites, not ignored.
- temporal-physical27-fresh.log: 16 passed. Stateful/ordered-row requirements for days/SMA/CVaR/kurt are explicit. repaired-batch-pandas3-final.log: 2 passed with warmup NaNs preserved via future_stack=True.
- Scoped git diff --check passes for cleaned_operators, backend, planner, runtime, factor_engine/tests, tests and evidence/r6. Whole-tree whitespace check has an existing unrelated R30 CSV CRLF/trailing-whitespace issue; no unrelated rewrite performed.
- Final disk check: 776 GiB free; evidence/r6 27 MiB. No branches, repository copies, commits, pushes, deployment or production-factor publication.

## Remaining verification and usage limits

- All 1,757 research-call contracts are available in the inventory, but complete independent numerical certification across every operator/backend/input shape has not been performed. A passing inventory must not promote unverified kernels to production certification.
- Static graph direct calls are implemented; dynamic date-by-node-by-node routing and DSL matrix literals remain unsupported.
- Some repaired Polars backends deliberately delegate to the correct Pandas CPU implementation. They are not GPU/native/lazy/streaming implementations.
- Real multi-panel inputs, session calendars and fiscal availability data remain required. Stateful/full-history operators cannot be treated as independent finite-history shards without valid checkpoints.
- Universal GPU support, universal transitive dependency hashes, real COS batch persistence, live 80%-memory pressure and 100,000-factor throughput have NOT been established by this R6 test set.
- Failure logs are retained as audit history. Subsequent passing logs supersede only their corrected scope. Tiny owned patch files are removed after application; code, tests, evidence and user data are retained.
