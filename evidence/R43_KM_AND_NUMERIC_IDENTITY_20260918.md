# KM nearest-attractor repair and numeric result identities

Effective Polars KM equilibrium selected the first stable downward crossing. Canonical semantics select the attractor nearest the current state center, including properly bracketed zero plateaus. On the diagnosis fixture, 22 finite points differed, maximum absolute error 1.9100906279, all with multiple stable roots; this was not floating-point tolerance or ill-conditioning.

The repaired root selection follows canonical semantics. Independent two-root, plateau, one-root, missing/no-root checks and paired prefix tests are added. Eager materialized Polars-to-NumPy execution is declared truthfully, not native expressions or GPU.

Root evidence/r43-km-runmany-root-attempt2.log: 59 passed including the entire deepening suite, KM focused tests, cross-wave cache, sink backpressure, wave reuse and real AR run_many regression. First root attempt used a nonexistent test filename, exited 4 before execution; it is not evidence of a product failure or a pass.

Semantic versions now invalidate pre-repair identities for ts_poly2_resid (v2), ts_extremal_index (v2), ts_gpd_shape_pwm (v2), ts_deviation_from_mean (v2), ts_km_equilibrium_distance (v3). Root evidence/r43-repair-identity-attempt3.log: 61 passed. The new test first incorrectly simulated a live registry version change and then tried mutating the immutable catalog; both failed test attempts are retained. Final test simulates an old registration-time catalog snapshot and checks changed public contract hashes.

Historical outputs affected by quadratic residual math, GPD placeholders or KM root choice must be recomputed. This patch does not delete outputs, invalidate external storage destructively, deploy, or publish production factors.
