# V8 FO remediation ledger

Baseline: `6562886acf1bff40cfc58cb2cc138c130f9d18e42750aceca2700bcbcd39a7ca`.

- A02: existing evaluation artifacts are verified and exactly rebound to trial, objective, split, fidelity, and evidence identity before scores are read.
- A10: scalar plateau detection is objective-direction aware; multi-objective callers can record fixed-reference hypervolume instead of frontier count.
- A44: multiplicity counts terminal proposal outcomes, not PROPOSED/SELECTED lifecycle events; PRUNED is explicit.
- A45: OPEN/CLOSED/SEALED state, next sequence, head and count survive restore. Authorized continuation retains prior epoch seals.
- A46: new entries use versioned canonical JSON hashing; legacy entries remain verifiable as encoding v1.
- A69/A70: reservations bind exact attempt IDs; started failures conservatively charge the reserved envelope; duplicate/mismatched release cannot consume another attempt; restore validates aggregates and limits.
- A03/A04-A09/A15 consumer boundary: FO exposes a structural `DecisionProvider.decide(request)` port, imports no FA implementation, and verifies exact request/policy/context/candidate/fidelity receipt binding. WAIT/INCOMPARABLE utility leakage is rejected.

Verification: focused suites passed. The final full FO suite passes **885 tests** with 12 collection/deprecation warnings and 0 failures; the expanded cross-package integration files contribute 7 passing tests. Adversarial coverage includes a high-score COMPLETE artifact without a decision provider being pruned at the cheap tier, a research-mode provisional winner being denied sealed-test freeze, and both legacy local treatment entry points rejecting calls unless the caller explicitly declares screening-only semantics. Screening results carry `authoritative=False` and `purpose=SCREENING_DIAGNOSTIC`.

Cross-package synthetic integration executes QE's public `compute_joint_block_bootstrap` on one common `ResamplingPlan`, binds its plan/replicate/sample provenance into FA `RawJointMetricEvidence`, and uses the canonical `DecisionProvider` through FO final/tier paths and FP's public representation adapter. Direct FA, FO and FP decision/content hashes are identical. It also verifies the raw callback score is not final fitness, missing qualification remains unselectable with `None` utility, an INCONCLUSIVE higher-point variant cannot replace legal RAW, and altered context/candidate-set receipt bindings are rejected. The `NumericalQualificationReceipt` is explicitly scoped to the synthetic test domain and is not evidence of production numerical certification.

Same-device batching evidence compares one batch with repeated differently sized batches while retaining the exact same preregistered family ledger head/count. P-values are aggregated by hypothesis identity and corrected once over the complete family; the family and correction hashes are identical. Persisted orphan reservations remain charged across both SQLite restart and `BudgetTracker` checkpoint restore; a checkpoint that drops the attempt identity while retaining reserved aggregates is rejected.

Cross-layer invariants use production APIs throughout. Candidate-column permutation, independent single-column QE calls, and differently grouped factor-shard QE calls share one `ResamplingPlan`; canonical candidate alignment produces the same FA candidate-set hash and the same FO-verified decision identity. A production frozen fitted FP recipe is also applied to two inputs differing only in their future tail; their earlier treated prefix, QE raw evidence, candidates, and FO-verified FA decision remain byte-identical under the same frozen fit and comparison context. Labels are never provided to the causal FP or QE computation.

Lineage/retry/resource invariants are exercised through the runner and durable store. Parent, split-left, split-right, and recombined candidates each complete evaluation before one sealed complete family is bound without resetting the campaign. A second worker retry adds an operation but retains one effective hypothesis and the first worker's validation p-value. A callback failing after 9 of 10 synthetic work units receives the conservative full reservation charge and leaves no available budget; this proves safe accounting, not measured 90% wall-clock/resource metering.

A01/A03: `SearchConfig.screening_only` explicitly separates compatibility diagnostics from selectable search. Screening-only runs may retain diagnostics but cannot promote through quality tiers or freeze. Selectable runs require final FA provider/request construction (and a stage request factory for tiered search); runner receipts replace provisional score incumbents, and every freeze requires a persisted authoritative decision identity regardless of deployment mode.

History impact: old multiplicity totals derived from event count and old unsealed finished ledgers require reclassification. V1 hash chains remain readable but do not gain V2 canonical-encoding assurance. Failed historical compute with absent worker metering remains unknown spend, never inferred as zero.

Rollback: revert the listed FO source files as one unit. Do not roll back only the runner while retaining attempt-bound tracker serialization, or vice versa.

A43: `SQLiteCampaignStore.bind_hypothesis_family` accepts the structural QE artifact only, verifies the sealed ledger head/count and full recorded member set/identities/statuses, persists it immutably, and returns the verified artifact content hash for FA's `hypothesis_family_ref`.
