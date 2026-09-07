"""Spawn-safe parent-mediated access to the single ResourceBroker authority."""
from __future__ import annotations

import multiprocessing as mp
import threading
import uuid
from typing import Any

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind


class BrokerRPCError(RuntimeError):
    pass


class BrokerRPCTimeout(TimeoutError):
    pass


class RemoteLease:
    """Opaque child-side lease token; the real lease never leaves the parent."""

    def __init__(self, proxy: "ResourceBrokerProxy", token: str) -> None:
        self._proxy = proxy
        self.token = token
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._proxy._rpc("release", token=self.token)
            self._released = True


class ResourceBrokerProxy:
    def __init__(self, connection: Any, client_id: str, timeout_seconds: float) -> None:
        self._connection = connection
        self.client_id = client_id
        self.timeout_seconds = float(timeout_seconds)
        self._lock = threading.Lock()
        self._broken = False

    def __getstate__(self) -> dict[str, Any]:
        return {"_connection": self._connection, "client_id": self.client_id,
                "timeout_seconds": self.timeout_seconds}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._broken = False

    def _rpc(self, operation: str, **payload: Any) -> Any:
        if self._broken:
            raise BrokerRPCError("broker RPC proxy is fenced after transport failure")
        request_id = uuid.uuid4().hex
        if not self._lock.acquire(timeout=self.timeout_seconds):
            raise BrokerRPCTimeout(f"broker RPC {operation} lock wait timed out")
        try:
            sent: list[BaseException | None] = []
            def send_request():
                try: self._connection.send((request_id, operation, payload)); sent.append(None)
                except BaseException as exc: sent.append(exc)
            sender = threading.Thread(target=send_request, daemon=True); sender.start()
            sender.join(self.timeout_seconds)
            if sender.is_alive() or not sent or sent[0] is not None:
                self._fence_transport()
                raise BrokerRPCTimeout(f"broker RPC {operation} send failed or timed out")
            if not self._connection.poll(self.timeout_seconds):
                self._fence_transport()
                raise BrokerRPCTimeout(f"broker RPC {operation} timed out")
            received: list[Any] = []
            receiver = threading.Thread(target=lambda: received.append(self._connection.recv()), daemon=True)
            receiver.start(); receiver.join(self.timeout_seconds)
            if receiver.is_alive() or not received:
                self._fence_transport()
                raise BrokerRPCTimeout(f"broker RPC {operation} receive timed out")
            response_id, ok, value = received[0]
        finally:
            self._lock.release()
        if response_id != request_id:
            raise BrokerRPCError("broker RPC response identity mismatch")
        if not ok:
            raise BrokerRPCError(str(value))
        return value

    def _fence_transport(self) -> None:
        self._broken = True
        try: self._connection.close()
        except OSError: pass

    def acquire_memory(self, kind: MemoryLeaseKind, nbytes: int, *, lease_id: str = "") -> RemoteLease | None:
        token = self._rpc("acquire_memory", kind=str(kind), nbytes=int(nbytes), lease_id=lease_id)
        return None if token is None else RemoteLease(self, token)

    def try_reserve(self, task: Any, *, task_id: str = "") -> RemoteLease | None:
        token = self._rpc("try_reserve", task=task, task_id=task_id)
        return None if token is None else RemoteLease(self, token)

    def automatic_result_queue_budget(self) -> int:
        return int(self._rpc("automatic_result_queue_budget"))

    def current_read_budget(self) -> int:
        return int(self._rpc("current_read_budget"))

    def current_sink_budget(self) -> int:
        return int(self._rpc("current_sink_budget"))

    def execution_budget(self) -> int:
        return int(self._rpc("execution_budget"))

    def cpu_budget(self) -> int:
        return int(self._rpc("cpu_budget"))

    def summary(self) -> dict[str, Any]:
        """Read the parent's ledger, never construct worker-local authority."""
        return self._rpc("summary")

    def resource_envelope(self) -> Any:
        """Read the same effective envelope used by the parent authority."""
        return self._rpc("resource_envelope")

    @property
    def hard_memory_limit(self) -> int:
        """Stable parent admission ceiling, distinct from temporary headroom."""
        return int(self._rpc("hard_memory_limit"))

    @property
    def hard_cpu_slots(self) -> int:
        return int(self._rpc("hard_cpu_slots"))

    def resource_decision(self, *, sink_backpressure: float = 0.0,
                          job_memory_lease_bytes: int | None = None) -> Any:
        return self._rpc("resource_decision", sink_backpressure=float(sink_backpressure),
                         job_memory_lease_bytes=job_memory_lease_bytes)

    def acquire_protected_egress(self, result_queue_bytes: int, writer_workspace_bytes: int,
                                 *, lease_id: str) -> tuple[RemoteLease, RemoteLease] | None:
        tokens = self._rpc("acquire_protected_egress", result_queue_bytes=int(result_queue_bytes),
                           writer_workspace_bytes=int(writer_workspace_bytes), lease_id=lease_id)
        if tokens is None:
            return None
        return RemoteLease(self, tokens[0]), RemoteLease(self, tokens[1])


class ParentBrokerIPC:
    """Own real leases in the parent and serve spawn-safe child proxies."""

    def __init__(self, broker: Any, *, rpc_timeout_seconds: float = 5.0) -> None:
        self.broker = broker
        self.rpc_timeout_seconds = max(0.05, float(rpc_timeout_seconds))
        self._lock = threading.RLock()
        self._leases: dict[str, tuple[str, Any]] = {}
        self._connections: dict[str, Any] = {}
        self._client_processes: dict[str, tuple[int, str]] = {}
        self._threads: list[threading.Thread] = []
        self._closing_clients: set[str] = set()
        self._closed = False

    def create_proxy(self) -> ResourceBrokerProxy:
        if self._closed:
            raise RuntimeError("broker IPC is closed")
        parent, child = mp.get_context("spawn").Pipe(duplex=True)
        client_id = uuid.uuid4().hex
        with self._lock:
            self._connections[client_id] = parent
        thread = threading.Thread(target=self._serve, args=(client_id, parent), daemon=True)
        thread.start()
        self._threads.append(thread)
        return ResourceBrokerProxy(child, client_id, self.rpc_timeout_seconds)

    def _serve(self, client_id: str, connection: Any) -> None:
        while not self._closed:
            try:
                if not connection.poll(0.1):
                    continue
                request_id, operation, payload = connection.recv()
            except (EOFError, OSError):
                return  # disconnect is not exit proof; leases remain charged
            try:
                value = self._dispatch(client_id, operation, payload)
                connection.send((request_id, True, value))
            except Exception as exc:
                try:
                    connection.send((request_id, False, f"{type(exc).__name__}: {exc}"))
                except OSError:
                    return

    def _dispatch(self, client_id: str, operation: str, payload: dict[str, Any]) -> Any:
        with self._lock:
            if client_id in self._closing_clients:
                raise RuntimeError("broker IPC client is fenced")
        if operation == "release":
            token = str(payload["token"])
            with self._lock:
                owned = self._leases.get(token)
                if owned is None or owned[0] != client_id:
                    return False
            owned[1].release()
            with self._lock:
                self._leases.pop(token, None)
            return True
        if operation == "acquire_memory":
            lease = self.broker.acquire_memory(
                MemoryLeaseKind(payload["kind"]), int(payload["nbytes"]),
                lease_id=str(payload.get("lease_id") or ""),
            )
        elif operation == "try_reserve":
            lease = self.broker.try_reserve(payload["task"], task_id=str(payload.get("task_id") or ""))
        elif operation == "acquire_protected_egress":
            leases = self.broker.acquire_protected_egress(
                int(payload["result_queue_bytes"]), int(payload["writer_workspace_bytes"]),
                lease_id=str(payload["lease_id"]),
            )
            if leases is None:
                return None
            tokens = []
            with self._lock:
                if client_id in self._closing_clients:
                    for real_lease in leases:
                        real_lease.release()
                    raise RuntimeError("broker IPC client fenced during egress admission")
                for real_lease in leases:
                    token = uuid.uuid4().hex
                    self._leases[token] = (client_id, real_lease)
                    tokens.append(token)
            return tuple(tokens)
        elif operation == "resource_decision":
            return self.broker.resource_decision(
                sink_backpressure=payload["sink_backpressure"],
                job_memory_lease_bytes=payload["job_memory_lease_bytes"],
            )
        elif operation in {"hard_cpu_slots", "hard_memory_limit"}:
            return int(getattr(self.broker, operation))
        elif operation in {"automatic_result_queue_budget", "current_read_budget", "current_sink_budget", "execution_budget", "cpu_budget", "summary", "resource_envelope"}:
            return getattr(self.broker, operation)()
        else:
            raise ValueError(f"unsupported broker RPC operation {operation!r}")
        if lease is None:
            return None
        token = uuid.uuid4().hex
        with self._lock:
            if client_id in self._closing_clients:
                lease.release()
                raise RuntimeError("broker IPC client fenced during admission")
            self._leases[token] = (client_id, lease)
        return token

    def bind_client_process(self, client_id: str, pid: int) -> None:
        """Bind a client to the parent's observed PID/starttime identity."""
        from factor_engine.runtime.resource_broker import _process_identity

        identity = _process_identity(int(pid))
        if identity is None:
            raise ValueError("client process identity is unavailable")
        with self._lock:
            if client_id not in self._connections:
                raise KeyError("unknown broker IPC client")
            previous = self._client_processes.get(client_id)
            if previous is not None and previous != identity:
                raise RuntimeError("broker IPC client is already bound to another process")
            self._client_processes[client_id] = identity

    def reclaim_client(self, client_id: str) -> int:
        """Release orphan leases only when the parent verifies process exit."""
        from factor_engine.runtime.resource_broker import _process_identity

        with self._lock:
            identity = self._client_processes.get(client_id)
        if identity is None:
            raise RuntimeError("client has no parent-verified process identity")
        if _process_identity(identity[0]) == identity:
            return 0
        with self._lock:
            self._closing_clients.add(client_id)
            items = [(token, lease) for token, (owner, lease) in self._leases.items() if owner == client_id]
            for token, _ in items:
                self._leases.pop(token, None)
            self._client_processes.pop(client_id, None)
        for _, lease in items:
            lease.release()
        return len(items)

    def active_tokens(self) -> int:
        with self._lock:
            return len(self._leases)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=self.rpc_timeout_seconds)
