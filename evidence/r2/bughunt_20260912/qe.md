# QE bounded bug hunt — 2026-09-12

## Scope and protections

- Formal tree: `/home/sunhaiwei/quant_projects`, baseline HEAD `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`.
- Read and preserved concurrent changes in `contracts/evaluation_refs.py`, `runtime/metric_instances.py`, and `tests/test_r3_fe_reference_metadata.py`.
- No branch, commit, push, deployment, production data access, dependency install, or large market computation.

## Reproduced and fixed defects

1. **Unnamed/misaligned probe axes were constructible.** `ProbePortfolioArtifact(values=(T,F))` accepted an empty time axis, an empty or wrong-length factor axis, and duplicate factor IDs. The contract now requires a complete time axis plus unique factor IDs exactly matching F.
2. **Probe and executable identities could collide across serialization/config hashing.** Their serialized dictionaries omitted the runtime artifact kind, and the executable subclass inherited the probe hash tag. Serialization now records `artifact_type`, rejects cross-kind loading, and executable hashing has a distinct domain tag.
3. **Batched executable evidence retained only the first factor's ledger ref.** Batch assembly now records a factor-keyed execution-ledger mapping. Multi-factor `ExecutablePortfolioArtifact` requires that mapping to exactly cover its factor axis and rejects a scalar/first-column ref masquerading as batch evidence.

## Code and counterexamples

- `quant_evaluator/contracts/artifact_types.py`
  - SHA-256 `2317f984dd351627cdc9eac8b7b3dabf81c664ffa6cd1604641ec800824300ee`
- `quant_evaluator/adapters/execution_trajectory.py`
  - SHA-256 `e6c37602d1f83c6f3b76969f5e2f730c917e7cfbc1f2aeefafd64728002ef5c1`
- `quant_evaluator/tests/test_v5_execution_trajectory_adapter.py`
  - SHA-256 `68db97ade565f9916028b3a9421b9de2718519bc5d01aad17061278da3949602`
  - New counterexamples cover missing/wrong/duplicate axes, executable-to-probe downgrade, kind-specific hashes, exact deserialization, and two ledgers across two factors.
- `quant_evaluator/tests/test_qe_p0_03_hash_stability.py`
  - SHA-256 `5c6250869b4b58ac1026311ff0bb8a9344b35a5ca6554be5ad924a142aa9bd17`
  - Existing cross-process hash fixture now supplies the mandatory named axes.

## Actual execution

- Focused artifact/adapter/serialization/public-path run: **67 passed**, 1 warning, 4.58s.
- Full `quant_evaluator/tests` plus `quant_evaluator/adapters/tests`: **1260 passed, 2 skipped**, 13 warnings, 146.59s.
- The two skips are the existing documented missing zscore/standardize and row-drop kernels; no skip was added or converted to a pass.
- `git diff --check -- quant_evaluator`: PASS.

## Remaining boundary

`ExecutablePortfolioArtifact` validates a complete typed execution-evidence envelope but does not itself resolve external ledger bytes. Authenticity/existence remains the execution-domain resolver's responsibility; QE does not infer certification from research metadata or a bare string reference.
