# Effective Polars extremal-index repair

Root verification: evidence/r40-extremal-root.log — 7 passed; sampled family peak 797888512 bytes, no watchdog termination. This is sampled RSS, not a hard memory cap.

The effective ts_advanced_batch1 registration had a pandas-style placeholder and incorrect defaults. It now invokes the canonical runs estimator over Polars columns, with canonical defaults (120, upper, 0.9, 3, 1), NaN cluster breaks and preserved time identity. Independent runs-count oracle covers both tails, date/timestamp, missing data, prefix causality, defaults, empty input and invalid parameters.

Execution is eager CPU Polars-to-NumPy-to-Polars, not native lazy expressions or GPU. This does not certify all operators, all backends, or the 110k catalog. Full deepening parity advanced past this repair but exposed a separate GPD placeholder; that repair is ongoing.
