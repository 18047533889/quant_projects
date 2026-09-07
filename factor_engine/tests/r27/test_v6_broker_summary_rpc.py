from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC


def test_hard_limit_rpc_is_not_temporary_available_headroom():
    class Parent:
        hard_memory_limit = 8192

        def current_read_budget(self):
            return 0

    broker = Parent()
    ipc = ParentBrokerIPC(broker)
    try:
        proxy = ipc.create_proxy()
        assert proxy.current_read_budget() == 0
        assert proxy.hard_memory_limit == 8192
        broker.hard_memory_limit = 16384
        assert proxy.hard_memory_limit == 16384
        assert ipc.active_tokens() == 0
    finally:
        ipc.close()


def test_envelope_rpc_uses_parent_authority():
    from types import SimpleNamespace

    class Parent:
        calls = 0

        def resource_envelope(self):
            self.calls += 1
            return SimpleNamespace(safe_memory_bytes=8192, measurement_status="AVAILABLE")

    broker = Parent()
    ipc = ParentBrokerIPC(broker)
    try:
        envelope = ipc.create_proxy().resource_envelope()
        assert envelope.safe_memory_bytes == 8192
        assert envelope.measurement_status == "AVAILABLE"
        assert broker.calls == 1 and ipc.active_tokens() == 0
    finally:
        ipc.close()


def test_summary_rpc_uses_parent_without_allocating_leases():
    class Parent:
        calls = 0

        def summary(self):
            self.calls += 1
            return {"authority": "parent", "running_tasks": 7, "lease_bytes": 123}

    broker = Parent()
    ipc = ParentBrokerIPC(broker)
    try:
        proxy = ipc.create_proxy()
        assert proxy.summary() == {"authority": "parent", "running_tasks": 7, "lease_bytes": 123}
        assert broker.calls == 1
        assert ipc.active_tokens() == 0
    finally:
        ipc.close()
