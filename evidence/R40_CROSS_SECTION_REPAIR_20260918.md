# Cross-sectional call-contract and regression repair

Before: evidence/r40-cross-section-after-audit-once.log — 4 failed, 145 passed. Three tangent tests omitted two required panels; the peer-deviation test supplied two of its five declared inputs. Tangent metadata also failed to declare panel/scalar topology, leaking a raw Python missing-argument error.

Repair:
- Declare f1/f2/f3 panels and scalar k for cs_knn_tangent_residual, retaining k>=20 and rank/breadth fail-closed behavior.
- Real 32-asset three-feature tests, independent pandas ranks + covariance eigenvector normal-distance oracle, missing/insufficient breadth, prefix, permutation and parameter validation.
- Five-component peer-deviation fixture and independent ddof=0 z-score sum expectation.
- Build the registry catalog once per audit rather than once per canonical. Keep the k>=2 rejection test compatible with the current clear error message.

Root after: evidence/r40-cross-section-repaired-root.log — 149 passed, no skips. Watchdog sampled family peak 803774464 bytes, 82.9s; not a hard memory cap. These are suite tests, not 149 newly certified operators or full factor-catalog execution.
