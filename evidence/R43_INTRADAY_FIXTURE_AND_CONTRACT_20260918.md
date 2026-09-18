# Intraday finite-value fixtures and explicit impact inputs

Root reviewed minute tests:
- Impact decay uses ret+amount on two days/two assets with analytically known exponential log-price decay rates; compares exact finite daily values, invalid amount, required input and completed-day prefix.
- Activity-duration curvature uses an explicit official ASHARE bar-start calendar and 240 bars per session. Independent bucket-completion oracle, finite values, missing calendar/grid/NaN fail-closed and completed-day prefix are tested.
- Reset the one-time missing-calendar warning in that test so prior test order cannot create a false failure.

Impact metadata had names but lacked explicit panel/scalar topology. Added ret/amount panel parameters and horizon/shock_quantile scalar parameters; missing amount now raises the formal parameter error. Numeric kernels and session-grid policies were not changed. Availability remains session_close and not same-session usable.

Root evidence/r43-intraday-formal-contract-root.log: 9 passed, no skips; sampled family RSS 807346176 bytes. Earlier evidence/r43-intraday-real-fixtures-root.log had 9 passes before formal topology hardening. Small fixture validation is not a full-day data-quality certification or production materialization.
