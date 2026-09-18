# R39 conditional operator contracts

Production fix: ts_modwt_band_corr now declares both required input panels
and authored window/level/band defaults (120/3/1). Missing input is rejected
by the shared parameter contract on both backends rather than leaking a raw
Python missing-positional-argument exception.

Legacy conditional tests now supply valid masks and required panels, with
independent rolling min/max oracles, missing-mask and future-prefix checks.
Composite tests retain the active-set checks and record RSI's actual Polars
source instead of requiring an obsolete source label.

Root combined suite: evidence/r39-conditional-root.log — **46 passed**.
No operator restored from a removed surface; no production promotion.
