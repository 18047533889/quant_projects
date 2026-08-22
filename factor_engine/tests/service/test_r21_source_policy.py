"""R21-019..027 + R21-089: approved-source policy and secret redaction."""

from __future__ import annotations

import pytest

from service.errors import ServiceError, sanitize_message
from service.security import ApprovedSourcePolicy


def _policy(**kwargs):
    base = dict(
        approved_profiles={"prod_daily": {"dataset": "ashare_stock_daily"}},
        approved_datasets=frozenset({"ashare_stock_daily", "us_stock_daily"}),
        allowed_clickhouse_hosts=frozenset({"ch.internal.example"}),
        allowed_clickhouse_tables=frozenset({"ch.internal.example:prod:factors"}),
        approved_roots=frozenset({"/data/factors/approved"}),
    )
    base.update(kwargs)
    return ApprovedSourcePolicy(**base)


class TestApprovedSourcePolicy:
    def test_data_access_allowlist(self):
        p = _policy()
        p.validate_source({"type": "data_access", "dataset": "ashare_stock_daily"}, production=True)
        with pytest.raises(ServiceError):
            p.validate_source({"type": "data_access", "dataset": "secret_db"}, production=True)

    def test_clickhouse_host_allowlist(self):
        p = _policy()
        p.validate_source(
            {"type": "clickhouse", "host": "ch.internal.example", "database": "prod", "table": "factors"},
            production=True,
        )
        with pytest.raises(ServiceError):
            p.validate_source(
                {"type": "clickhouse", "host": "evil.example", "database": "prod", "table": "factors"},
                production=True,
            )

    def test_parquet_root_allowlist(self):
        p = _policy()
        p.validate_source(
            {"type": "parquet", "root": "/data/factors/approved/daily", "timestamp_col": "t", "instrument_col": "i"},
            production=True,
        )
        with pytest.raises(ServiceError):
            p.validate_source(
                {"type": "parquet", "root": "/etc/passwd", "timestamp_col": "t", "instrument_col": "i"},
                production=True,
            )

    def test_unknown_source_type_rejected_for_production(self):
        p = _policy()
        with pytest.raises(ServiceError):
            p.validate_source({"type": "mystery_remote"}, production=True)

    def test_research_never_checked(self):
        p = _policy()
        p.validate_source({"type": "data_access", "dataset": "anything"}, production=False)

    def test_profile_resolves_dataset(self):
        from service.security import resolve_source_profile

        src = resolve_source_profile({"approved_source_profile_id": "prod_daily"}, policy=_policy())
        assert src == {"type": "data_access", "dataset": "ashare_stock_daily"}

    def test_unknown_profile_rejected(self):
        from service.security import resolve_source_profile

        with pytest.raises(ServiceError):
            resolve_source_profile({"approved_source_profile_id": "nope"}, policy=_policy())


class TestRedaction:
    def test_password_redacted(self):
        assert "hunter2" not in sanitize_message("connect failed password=hunter2 host=x")
        assert "password=<redacted>" in sanitize_message("password=hunter2")

    def test_dsn_redacted(self):
        assert "secret" not in sanitize_message("postgres://user:secret@db:5432/factors")

    def test_authorization_header_redacted(self):
        assert sanitize_message("Authorization: Bearer abc.def.ghi") == "Authorization: <redacted>"
