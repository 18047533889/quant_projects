# R40 event default binding

Live event_response_decay_rate Polars registration comes from polars_dynamics,
not the obsolete common/polars_event candidate. The generated variadic wrapper
hid its authored callable's defaults from the common parameter binder.

_mk now exposes _contract_callable=staticmethod(fn). Omitted history_window,
horizon and min_events bind to the authored 120/10/3 without callers manually
filling them in. Explicit/default calls match pandas and preserve future-prefix
behavior; no eligibility or native/GPU claims changed.

Root regression: evidence/r40-event-default-root.log:
**74 passed,1 deselected**. New default tests plus event-response and deepening
registration/shape/domain tests. The deselected full deepening parity test
passed this repaired event failure but exposed a separate ts_extremal_index
Polars implementation bug, currently being repaired. It is not reported as
passing and its earlier failure log is retained.
