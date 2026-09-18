# Effective GPD PWM and deviation-from-mean Polars repair

GPD had a TODO placeholder computing an absolute-value quantile, with unsupported DataFrame methods and incompatible defaults. It now wraps the canonical probability-weighted-moment kernel with defaults 120/upper/0.2/10, canonical parameter metadata and retained time columns. Eager CPU NumPy execution is explicitly declared; no native-expression/GPU claim or research eligibility promotion.

The Polars-only ts_deviation_from_mean attempted DataFrame.rolling_mean, which does not exist. It now evaluates abs(current - trailing rolling mean) using real Polars expressions over value columns, retains metadata columns and rejects pandas input clearly. It still requires an explicit window>=20 and a full window; this change does not add a pandas backend.

Root evidence/r42-tail-deviation-root.log: 18 passed (7 GPD, 4 deviation, 7 extremal-index regression). Independent PWM formula, both tails, defaults, time identities, empty/NaN, future-prefix and invalid domains are exercised. Sampled family peak 449019904 bytes, not a hard cap.

Full deepening parity now passes GPD/extremal but exposes a separate KM equilibrium numerical discrepancy under investigation. No blanket all-backend/110k certification.
