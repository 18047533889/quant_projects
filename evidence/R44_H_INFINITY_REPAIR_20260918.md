# H-infinity information-form repair

The old attenuation recurrence could amplify a small smooth fixture to about 2.9e27. It incorrectly placed gamma squared inside a covariance-subtraction denominator and then clamped covariance, rather than enforcing the H-infinity Riccati feasibility condition.

Now the scalar posterior information is 1/P_pred + 1/r - 1/gamma², gain P_post/r. Gamma is the attenuation level (smaller feasible values impose stronger constraints; the large-gamma limit recovers Kalman). Nonpositive/nonfinite gamma/q/r reject. Nonpositive/nonfinite information makes the remaining path unavailable, not a clipped finite substitute. Tiny positive gamma fails closed without a division-by-zero underflow exception.

Independent scalar recurrences cover H-infinity and alpha-beta; real Polars registry execution replaces references to a removed private bridge table. The obsolete assertion that every constant-velocity tracker must reduce total variance of a strongly trending/sinusoidal series is replaced by its actual recurrence.

Root evidence/r44-kalman-root.log: 78 passed, no skips, sampled family peak 466169856 bytes. Numeric version ts_h_infinity_level_filter is bumped to v2. Old outputs from the incorrect recurrence must be recomputed; no stored production outputs were modified here.
