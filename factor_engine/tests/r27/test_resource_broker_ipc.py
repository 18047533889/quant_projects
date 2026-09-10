import multiprocessing as mp

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.resource_broker_ipc import (
    BrokerRPCError, BrokerRPCTimeout, ParentBrokerIPC, ResourceBrokerProxy,
)


def _acquire_in_child(proxy, nbytes, output, release_event):
    lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, nbytes)
    output.put(lease is not None)
    if lease is not None:
        release_event.wait(3)
        lease.release()


def _broker():
    from factor_engine.runtime.auto_memory_budget import AutoMemoryBudget
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3, cpu_slots=4,
        min_host_reserve_gb=0, min_host_reserve_fraction=0,
    )
    budget = 1024**3
    broker._refresh_auto_budget = lambda: AutoMemoryBudget(
        hard_memory_limit=8 * 1024**3, emergency_reserve=0,
        safe_live_budget=budget, execution_budget=budget,
        safety_factor=.8, measurement_state="TEST_INJECTED",
    )
    return broker


def test_spawn_children_compete_for_one_parent_budget():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    budget = broker.execution_budget()
    ordinary_budget = budget - broker.current_sink_budget()
    assert ordinary_budget > 2
    ctx = mp.get_context("spawn")
    output = ctx.Queue()
    release = ctx.Event()
    p1 = ctx.Process(target=_acquire_in_child, args=(ipc.create_proxy(), ordinary_budget, output, release))
    p2 = ctx.Process(target=_acquire_in_child, args=(ipc.create_proxy(), 2, output, release))
    p1.start(); assert output.get(timeout=5) is True
    p2.start(); assert output.get(timeout=5) is False
    release.set(); p1.join(5); p2.join(5)
    assert p1.exitcode == p2.exitcode == 0
    assert ipc.active_tokens() == 0
    ipc.close()


def test_disconnect_does_not_reclaim_without_exit_proof():
    import subprocess, sys
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    proxy = ipc.create_proxy()
    lease = proxy.acquire_memory(MemoryLeaseKind.COMPUTE, 1024)
    assert lease is not None and ipc.active_tokens() == 1
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2)"])
    ipc.bind_client_process(proxy.client_id, process.pid)
    assert ipc.reclaim_client(proxy.client_id) == 0
    assert ipc.active_tokens() == 1
    process.wait(timeout=2)
    assert ipc.reclaim_client(proxy.client_id) == 1
    assert ipc.active_tokens() == 0
    ipc.close()


def test_protected_egress_real_leases_stay_in_parent():
    broker = _broker()
    ipc = ParentBrokerIPC(broker)
    proxy = ipc.create_proxy()
    leases = proxy.acquire_protected_egress(1024, 2048, lease_id="rpc-egress")
    assert leases is not None
    assert ipc.active_tokens() == 2
    assert broker._lease_sum_bytes() == 3072
    leases[0].release(); leases[1].release()
    assert ipc.active_tokens() == 0
    assert broker._lease_sum_bytes() == 0
    ipc.close()


def test_transport_timeout_permanently_fences_proxy():
    import time
    class Blocked:
        def send(self, value): time.sleep(1)
        def close(self): pass
    proxy = ResourceBrokerProxy(Blocked(), "blocked", 0.05)
    try:
        proxy.execution_budget()
    except BrokerRPCTimeout:
        pass
    else:
        raise AssertionError("blocked send must time out")
    try:
        proxy.execution_budget()
    except BrokerRPCError as exc:
        assert "fenced" in str(exc)
    else:
        raise AssertionError("timed-out proxy must remain fenced")
