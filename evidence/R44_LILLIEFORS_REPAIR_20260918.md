# Expanding Lilliefors p-value repair

The operator called scipy.stats.lilliefors, which does not exist; the expanding evaluator swallowed that AttributeError and returned only missing values. It now calls statsmodels.stats.diagnostic.lilliefors(dist="norm", pvalmethod="table"). There is no fallback to another test.

The import is local to the operator and outside the exception-swallowing expanding callback. Missing statsmodels raises an explicit ImportError without preventing unrelated statistics operators from registering. factor_engine/pyproject.toml now declares statsmodels>=0.14; the root team production lock already pins statsmodels==0.14.6 with a hash and is unchanged. No large lock or environment rebuild occurred.

Root evidence/r44-lilliefors-root.log: 5 passed: known reference p-values, finite-only expanding support, prefix causality/degenerate constant samples, dependency isolation, TOML and root lock declaration. Numeric semantic version is bumped to v2. This does not count expected dependency rejection or all-NaN as a usable finite factor.
