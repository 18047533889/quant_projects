from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from data_access.core.exceptions import ResourceAdmissionError
from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation


@pytest.mark.parametrize('mode', ['global_limit', 'principal_limit', 'duplicate'])
@pytest.mark.parametrize('host_backed', [True, False])
def test_admission_rechecks_after_unlocked_host_callback(mode, host_backed):
    governor = GlobalResourceGovernor(
        max_active_queries=1 if mode == 'global_limit' else 4,
        per_principal_active=1 if mode == 'principal_limit' else None,
        max_total_reserved_memory=1000,
        max_total_scan_bytes_inflight=1000,
    )
    barrier = threading.Barrier(2)
    leases = []

    class Lease:
        released = 0

        def release(self):
            # Rollback callbacks must run outside the governor lock.
            governor.active_count()
            self.released += 1

    def request_lease(*args):
        lease = Lease()
        leases.append(lease)
        barrier.wait(timeout=5)
        return lease if host_backed else None

    governor.set_host_lease_request(request_lease)

    def admit(i):
        query_id = 'same' if mode == 'duplicate' else str(i)
        try:
            return governor.admit(ResourceReservation(query_id, 'principal', 10, 10))
        except ResourceAdmissionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(admit, range(2)))
    assert sum(result is not None for result in results) == 1
    assert governor.active_count() == 1
    assert sum(lease.released for lease in leases) == int(host_backed)
    for result in results:
        if result is not None:
            governor.release(result.query_id)
    assert governor.active_count() == 0
    assert sum(lease.released for lease in leases) == 2 * int(host_backed)
