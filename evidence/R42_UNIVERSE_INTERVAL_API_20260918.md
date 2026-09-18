# Auditable universe-snapshot interval API

DataAccess UniverseSnapshot now supports paired optional effective_start/effective_end, explicitly provided from trusted membership-version provenance. ISO dates and timezone-aware datetimes are validated; reversed/malformed/naive datetime intervals reject. Interval fields participate in the digest and to_dict. Legacy snapshots without intervals retain their old digest.

from_store deliberately does not infer effective bounds from request time_range. Historical set membership is not proof that it applies throughout a requested interval.

FactorEngine's production scope gate rebuilds the interval-bound digest, checks exact members/market/universe and that the request lies inside the audited interval. Reversed requests, NaT, naive datetimes, forged identity and absent intervals reject. Date-only bounds use UTC calendar boundaries; aware datetimes normalize to UTC. This is a conservative interface, not an automatic reconstruction of changing historical membership.

Root evidence:
- r42-universe-interval-root.log: 38 passed (new interval/gate tests plus legacy scope rejection).
- r42-da-snapshot-regression.log: 25 passed (legacy DataAccess snapshot regression).
- Tests use official build/from_store; no mutation of frozen snapshots to manufacture missing fields.

Real historical membership providers still need interval provenance wired into their snapshots. This patch makes that expressible and verifiable; it does not certify existing set-valued snapshots or permit production CS factors to bypass historical scope checks.
