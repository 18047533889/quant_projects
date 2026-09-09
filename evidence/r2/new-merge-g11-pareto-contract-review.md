# G11 FA/FO Pareto contract read-only review

Date: 2026-09-09. Formal tree: `/home/sunhaiwei/quant_projects`.

## Result

`integration_tests/test_contract_roundtrip_xpkg.py` now calls the real FA API consistently. The positive Pareto fixture supplies finite decision metadata under `objectives` and a versioned `AssemblyPolicy` with frozen `required_objectives=("quality",)`. It reaches and passes the original assertion that changing `selection_policy` from manual to Pareto changes `assembly_hash`. A separate negative fixture proves that a candidate missing the frozen objective is rejected.

FA fails closed when `pareto_front` receives no assembly policy or an empty objective tuple, filters candidates missing any required objective, and ranks on that same frozen tuple. `AssemblyPolicy.to_dict()` includes `required_objectives`, so its content hash binds the objective space.

A production-code search found no `FactorSetAssembler`, `AssemblyPolicy`, or `required_objectives` call in `factor_optimizer`, and no non-test business caller of `FactorSetAssembler`. FO's `factor_optimizer.search.pareto` and treatment-decision frontier are a separate trial-search API. The fixture is consistent with the current FA interface, but it cannot certify a production FO-to-FA assembly consumer because that call chain is not present.

## Executed tests

- Cross-package roundtrip: 29 passed, 3 deprecation warnings.
- FA Pareto/objective assembly selection: 4 passed, 43 deselected, 3 deprecation warnings.
- FO Pareto suite: 31 passed.
- FA optimizer Pareto suite: 10 passed, 3 deprecation warnings.

Relevant SHA-256 values were unchanged across the review:

- `factor_assets/assembly/engine.py`: `974d5b394a64ad026ceb94c08b822fcbfa541b25c7ffe22ac5946a47042a6b4b`
- `factor_assets/contracts/assembly_evidence.py`: `5bc559202d97ba19736efa629a492a4d3a664be08f1ff60ddc25071771262a60`
- `integration_tests/test_contract_roundtrip_xpkg.py`: `8e418ab8d14391b318f6c83c631721e363489e6acf5a5a5993ba98b2a16e498c`
- `factor_optimizer/factor_optimizer/search/pareto.py`: `b368befd9d0d3d46498a43cbd1b399c474d795457de2af27cdd5e9eebbafebe6`

Boundary: current-tree contract evidence only; no remote CI, deployed consumer, or old-worker epoch was certified. No source edit, dependency install, branch, commit, push, or deployment was performed.
