import pytest

from factor_engine.runtime.lineage import _canonicalize_config_value, hash_data_source_config


@pytest.mark.parametrize("key", ["password", "access_key", "secret_key", "token", "authorization", "private_key"])
@pytest.mark.parametrize("value", ["synthetic-secret", "not:a:url", "https://synthetic-secret.example/token"])
def test_secret_fields_never_echo_plain_or_uri_shaped_values(key, value):
    sanitized = _canonicalize_config_value({key: value})
    assert sanitized == {key: "<redacted>"}
    assert value not in repr(sanitized)


def test_dsn_preserves_endpoint_but_not_password():
    left = {"dsn": "postgresql://user:synthetic-one@db.example/a"}
    rotated = {"dsn": "postgresql://user:synthetic-two@db.example/a"}
    other = {"dsn": "postgresql://user:synthetic-one@db.example/b"}
    assert "synthetic-one" not in repr(_canonicalize_config_value(left))
    assert hash_data_source_config(left) == hash_data_source_config(rotated)
    assert hash_data_source_config(left) != hash_data_source_config(other)
