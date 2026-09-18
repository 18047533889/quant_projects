# Group-spectrum topology and legacy contract checks

Seven group-spectrum operators now explicitly declare four panels (f1/f2/f3/group) and their actual scalar controls. Missing input fails at the formal call boundary; numeric kernels are unchanged.

Legacy fixtures supplied only one feature. They now supply independent companion features and aligned groups. Added full-breadth four-panel finite execution and prefix checks for all seven operators, plus an independent covariance-eigenvalue oracle for member count, coverage ratio, mode share, effective rank and spectral gap.

Other stale tests now expect formal OperatorParameterError for rejected missing companions/non-finite weights, retaining rejection and valid-negative-weight assertions. Super-smoother v2 regression now requires semantic version >=2 so the existing v3 bump remains valid; formula assertions are unchanged.

Root evidence/r42-group-spectrum-final.log: 173 passed, no skips, sampled family peak 476196864 bytes. Earlier broad G/H scan had missing optional bidask/diptest golden-oracle dependencies (2 skips); those are not claimed as passed here, and the later targeted run does not cover those two optional oracle tests.
