"""Auth tests — login success/fail, session validate, logout invalidates, protected route."""

from __future__ import annotations

from starlette.testclient import TestClient

from quant_platform.app.api.app import create_app
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.security.passwords import hash_password


def _seed_user(db, username="alice", password="s3cret", principal_id="p1"):
    db.execute(
        "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
        (principal_id, "HUMAN", "Alice"),
    )
    db.execute(
        "INSERT INTO human_users (principal_id, username, password_hash) VALUES (?, ?, ?)",
        (principal_id, username, hash_password(password)),
    )


def _client():
    db = SqliteDb(":memory:")
    _seed_user(db)
    app = create_app(db=db, secret=b"test-secret")
    return TestClient(app)


def test_login_success_sets_cookie():
    client = _client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "s3cret"})
    assert resp.status_code == 200
    assert "qrp_session" in resp.cookies


def test_login_wrong_password_fails():
    client = _client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_fails():
    client = _client()
    resp = client.post("/auth/login", json={"username": "nobody", "password": "x"})
    assert resp.status_code == 401


def test_me_authenticated():
    client = _client()
    client.post("/auth/login", json={"username": "alice", "password": "s3cret"})
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["principal_id"] == "p1"
    assert body["username"] == "alice"


def test_me_unauthenticated_401():
    client = _client()
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_logout_invalidates_session():
    client = _client()
    client.post("/auth/login", json={"username": "alice", "password": "s3cret"})
    assert client.get("/auth/me").status_code == 200
    client.post("/auth/logout")
    # Cookie cleared => subsequent /auth/me is 401.
    assert client.get("/auth/me").status_code == 401


def test_health_protected_401_without_session():
    client = _client()
    assert client.get("/health").status_code == 401


def test_health_200_with_session():
    client = _client()
    client.post("/auth/login", json={"username": "alice", "password": "s3cret"})
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
