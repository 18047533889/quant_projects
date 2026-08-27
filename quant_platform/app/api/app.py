"""FastAPI app skeleton.

QRP-P1. Thin HTTP layer over the metadata DB + security layer. Mounts:

- ``POST /auth/login`` — username+password → sets an HTTP-only session cookie.
- ``POST /auth/logout`` — revokes the session and clears the cookie.
- ``GET /auth/me`` — returns the authenticated principal.
- ``GET /health`` — protected health route (requires a valid session).

The app is built by ``create_app`` with an injected ``Db`` and server secret, so
tests can wire an in-memory SQLite DB. No domain logic lives here.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from quant_platform.app.db.sqlite_backend import Db, SqliteDb
from quant_platform.app.security.auth import SessionManager
from quant_platform.app.security.passwords import verify_password
from quant_platform.app.security.rbac import RBAC

SESSION_COOKIE = "qrp_session"

_bearer = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class MeResponse(BaseModel):
    principal_id: str
    username: str
    display_name: str


def _get_session_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str | None:
    """Extract the session token from the Authorization header or the cookie."""
    if credentials is not None:
        return credentials.credentials
    return request.cookies.get(SESSION_COOKIE)


def create_app(
    db: Db | None = None,
    secret: bytes | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``db`` defaults to a fresh in-memory ``SqliteDb``. ``secret`` defaults to a
    random per-process secret (fine for dev; production must inject a stable
    secret).
    """
    if db is None:
        db = SqliteDb(":memory:")
    if secret is None:
        secret = secrets.token_bytes(32)

    sessions = SessionManager(db, secret)
    rbac = RBAC(db)

    app = FastAPI(title="Quant Research Platform", version="0.1.0")

    def _current_principal(token: str) -> dict[str, Any]:
        st = sessions.validate(token)
        if st is None:
            raise HTTPException(status_code=401, detail="invalid or expired session")
        rows = db.query(
            "SELECT p.principal_id, p.display_name, h.username "
            "FROM principals p LEFT JOIN human_users h ON h.principal_id = p.principal_id "
            "WHERE p.principal_id = ?",
            (st.principal_id,),
        )
        if not rows:
            raise HTTPException(status_code=401, detail="principal not found")
        return rows[0]

    @app.post("/auth/login", status_code=200)
    def login(req: LoginRequest, response: Response) -> dict[str, str]:
        rows = db.query(
            "SELECT p.principal_id, p.display_name, p.enabled, h.password_hash "
            "FROM principals p JOIN human_users h ON h.principal_id = p.principal_id "
            "WHERE h.username = ?",
            (req.username,),
        )
        if not rows:
            raise HTTPException(status_code=401, detail="invalid credentials")
        row = rows[0]
        if not row["enabled"]:
            raise HTTPException(status_code=403, detail="account disabled")
        if not verify_password(req.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid credentials")
        token = sessions.issue(row["principal_id"])
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=8 * 60 * 60,
        )
        return {"status": "ok"}

    @app.post("/auth/logout", status_code=200)
    def logout(response: Response, token: str | None = Depends(_get_session_token)) -> dict[str, str]:
        if token:
            sessions.revoke(token)
        response.delete_cookie(SESSION_COOKIE)
        return {"status": "ok"}

    @app.get("/auth/me", response_model=MeResponse)
    def me(token: str | None = Depends(_get_session_token)) -> MeResponse:
        if token is None:
            raise HTTPException(status_code=401, detail="missing session")
        principal = _current_principal(token)
        return MeResponse(
            principal_id=principal["principal_id"],
            username=principal["username"] or "",
            display_name=principal["display_name"],
        )

    @app.get("/health", status_code=200)
    def health(token: str | None = Depends(_get_session_token)) -> dict[str, str]:
        if token is None:
            raise HTTPException(status_code=401, detail="missing session")
        _current_principal(token)
        return {"status": "ok"}

    return app
