# v9 continuation checkpoint

Only edit `/home/sunhaiwei/quant_projects` through SSH `qs-server-c`; user calls this local. Preserve dirty worktree. No GitHub, commit, push, deployment, restart, production factor publication or history deletion.

Audit: `/tmp/fe-v9-review.7Sqnlq`, v9 third-addendum, 51 tasks. `verified_sources` are bad baselines, not replacement code. Selective lookup only.

Current progress authority: `evidence/r2/V9-PROGRESS.yaml` (intermediate; update after current tests/agents). Local authored files: `work/v9-root/`. Root has merged substantial v9 changes; do not restart from audit baselines.

Newest continuation (after wave2 checkpoint):

LATEST OVERRIDES (03:24 local):
- M19/M20 cross and bicoherence now BOTH MERGED after independent sol review. Versions2 added four canonicals + identity tests. `/tmp/v9-m20-bico-merged.xml`51pass; `/tmp/v9-m19-m20-merged.xml`running session3347 (cross+bico+identity).
- Wavelet final stage patch73ee2c68... merged source/tests into main. Actual full startup run `/tmp/v9-wavelet-merged.xml`1failed76passed: canonical spectral metadata empty param_specs overwrites Polars window spec/default. v9_wavelet agent fixing authoritative pandas default while implementing remaining3wavelet CPU bridges in isolate. Do not delete failed assertion or force-register native. Main now newer than initial e658/c1a hashes mentioned below.
- Lyapunov stage `/tmp/v9-m16-m17.patch`bc70e6... NOT MERGED; root found physical successor dk<=EPS returnsNone then AssertionError, zero-initial-distance argmin blocks next distinct neighbor, and compressed/physical/oracle numerical trajectory policy mismatch. Agent v8_protocol fixing before M10. 58/38pass old stage not certification.
- CPU bridge payload provenance is now durable: `evidence/r2/V9-CPU-BRIDGE-IDENTITY-PAYLOADS.json` with a reproducibility test in test_v9_cpu_bridge_identity.py. Its four hashes were verified.
- Next after active review fixes: rerun coherent wave3 combined, update source/binding snapshots and evidence. All user prohibitions unchanged.

- M25 O(N) exact Decimal fallback merged; Fraction/pandas/Polars tests 16 passed (`/tmp/v9-m25-linear-merged.xml`).
- Peer/expectile mandatory PhysicalImplementationSpec identity fields merged; both classify unsupported/non-native in research and production, not certified. Source hashes bind wrapper only, not all transitive kernels. Original identity payload texts are in v8_data messages; a durable payload file is still required.
- M37 final patch `ad13869d...` merged after baseline comparisons. Important: final owned signature file is `backend/operator_signatures_phase1.py`, NOT operator_types.py. Production price+ts_std(price) passes; typed price/ratio rejects; unknown production units fail closed. 57 merged M37/bridge/identity tests pass. First preliminary candidate signature was overwritten at startup and must not be restored.
- M46 `dynamic_knn.py` deque(lag+1) merged after independent review; 15 tests pass. `/tmp/v9-m46-lifetime.json` records exact output equivalence and graph arrays 1290240→64512 bytes for40x64x3,k3,lag1. Broker/RSS/auto80 not closed.
- Public planner evidence `/tmp/v9-public-planner.json` passes six stages, legacy equivalence, typed mismatch, dropped-leaf invariant and production numeric policy rejection. `Optimizer.compile()` does NOT exist; actual API optimize_with_pass_trace tested. Probe authored `work/v9-root/v9_public_planner_probe.py`.
- Scoped compileall cleaned_operators/ir/identity/planner/backend and git diff --check passed; not full imports.
- M19/M20 cross_spectrum.py + bicoherence changes in research_spectral.py are STAGED ONLY. Local source/tests in work/v9-root; executable stage `/tmp/v9-m37-stage` (root reused completed stage). 35 independent Welch/direct DFT/extreme scale/public pandas-Polars tests pass (`/tmp/v9-m19-m20-stage-public.xml`). Bicoherence independent review approved; cross review pending. Production cross/research have NOT been overwritten with these candidates. M18/RBF and M34/HSIC changes within research_spectral are already merged baseline and must be preserved.
- Wavelet agent initially wrote M04/M05/M06 into main worktree against instructions; do not revert. Root reviewed and requested strict scalar/default/public-wide bridge fixes in `/tmp/v9-wavelet-stage`, NOT YET MERGED. Main still first candidate (wavelet sha e658c06b..., batch4 c1a708ce...). Root added semantic versions2 for all four wavelet names + spectral_low_frequency_ratio, and identity tests. The stage prefix layout is `cleaned_operators/...` (no factor_engine prefix). Final follow-up still pending defaults128, x keyword contract, leftover int(window), source-identity payload provenance. Existing other wavelet batch4 placeholders remain OPEN. Agent's standalone stage9pass is not full registry proof.
- M01 first candidate `v9_m01/` was REJECTED for uncentered scale/exact gate causing translation dependence, stale scale score, missing status errors, unsafe restoration. Never merge it. Replaced agent assigned redesign; see below.

Active agents (GPT-5.6-sol only):

- `v8_protocol`: M16/M17 local_lyapunov isolated redesign; also root M19/M20 independent review (bico approved, cross pending).
- `v9_wavelet`: finishing stage review fixes above; must not write main again.
- `v9_huber_redesign`: NEW sol agent owns isolated M01/M07 shared Huber redesign and optional M11 ContextVar status. Reads actual baseline, not withdrawn candidate. Root requested x/y translation, actual MAD/std score, exact/rank/invalid/cap/extreme restore/public tests and Ridge regression preservation.
- `v8_data`: completed; old M01 candidate withdrawn. Do not reuse `layout_review` (model unconfirmed).

Newest verification:

- `/tmp/v9-wave2-combined.xml` and `.log`: 292 passed,13warnings (predates newest above).
- `/tmp/v9-m25-m32-backends-merged.xml`: 13 passed,12warnings.
- `/tmp/v9-m02-m03-polars-green.xml`: 15 passed.
- `/tmp/v9-m39-identity-merged.xml`: 136 passed.
- `/tmp/v9-m18-m34-m36-extreme-green.xml`: 19 passed.
- `/tmp/v9-existing-m33-green.xml`: 100 passed.
- `/tmp/v9-wave1-combined.xml`: earlier 224 passed, not current full certification.

Important merged-source distinctions:

- Bessel uses stable prewarped phase-normalized SOS. It now declares recursive/full-history via `declare_stateful`; runtime SOS checkpoint not implemented. M30 missing-as-zero remains OPEN.
- RBF now uses 32-row normalized direct-difference tiles, not the intermediate cdist version; preserves EPS bandwidth through hypot and handles ±1e308/sigma1e308. Pool admission still OPEN.
- HSIC M34 2-RHS and double-centering optimization is merged; `/tmp/v9-hsic-*-perf.json` timing predates M18's later RBF changes. Do not attribute that timing to the newest hash.
- M32 Hill pandas stable log ratios merged. Import order can select `polars_dynamics` or batch5. Batch5 placeholder has been replaced with explicit CPU delegation and rejects multi-stock long frames. Do not claim native Polars.
- M25 `polars_peer.py` now uses shared CPU row kernel, strict `align_cols`, explicit delegate kind, and rejects stock-code long panels. Agent's old metadata fixture allowed ambiguous stock_code; root test now checks date-only preservation plus unsupported-long rejection.
- M03 batch1 expectile beta placeholder was found by real registry test and replaced with shared asymmetric regression (not ordinary OLS). Defaults 60/.1/3, strict aligned frames, CPU delegation, multi-stock long rejection.
- M39 merged eight source files after comparing `/tmp/v9-m39-baseline`: base, registry, group_spectrum, contract_hardening, analyzer, identity serializer/api, expr canonical. FieldRef catalog_hash now round-trips and participates in identity. Root preserves all semantic version map additions (did not copy agent map).

Remaining: M01, M04–11, M16–17, M19–23, M26–27, M30, M35, M43 and Q01–03 are not closed (some staged as above). M18/M34/M46 resource admission is not closed. Broader actual backend/100k/real approved GPU performance, complete descendants and pending resume remain unproven. Never report all 1756 operators fixed/certified.

Next: finish current agent and root staged reviews, merge narrowly against hashes, run fresh combined tests, regenerate binding/source/test snapshots (WAVE1 snapshots remain historical and stale), then continue remaining audit packages. Missing business approval/profile and cgroup delegation are prior genuine limitations; do not invent approval or repeatedly ask the same missing-profile question.
