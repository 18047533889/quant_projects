"""Adapter from FA transactional commits to the platform's durable Outbox."""
from factor_assets.registry.serialization import event_to_json
import json

class _ConnectionDb:
    def __init__(self, connection): self.connection=connection
    def execute(self, sql, params=()): return self.connection.execute(sql, params)

class PlatformLifecycleOutboxAdapter:
    """Stages platform outbox rows on the exact FA SQLite transaction."""
    def stage(self, *, conn, event, revision, idempotency_key):
        # Import at the adapter edge only; FA contracts/repository remain free
        # of a platform dependency and the actual Outbox stays authoritative.
        from quant_platform.app.outbox import Outbox
        outbox=Outbox(_ConnectionDb(conn), publisher=None)
        return outbox.emit(event_type="factor.lifecycle.changed",
            aggregate_type="FactorAsset", aggregate_id=event.factor_id,
            correlation_id=event.decision_id or idempotency_key,
            idempotency_key=idempotency_key,
            payload={"revision":revision,"event":json.loads(event_to_json(event))})
