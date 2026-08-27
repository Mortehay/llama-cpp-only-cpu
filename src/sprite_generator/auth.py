"""API keys for the generation API.

WHY THIS REPLACES THE OLD TOKEN

`jobs.py` authenticated with a single shared bearer token read from
`SPRITE_API_TOKEN`, and its check began:

    if not API_TOKEN:
        return

Forgetting one environment variable disabled authentication for the whole API,
silently, while every endpoint kept answering exactly as before. There is a
`scripts/lan-expose.ps1` in this repo, so "it is only on localhost" was never a
safe assumption either.

The replacement keeps the failure mode visible instead of silent:

  * **Open mode** - no keys minted and no legacy token set. The API answers
    unauthenticated, `describe_mode()` says so, and the UI shows it. This is
    the fresh-install state and it is honest about being open.
  * **Enforced mode** - the moment the first key exists (or the legacy token is
    set), every scoped endpoint requires a valid credential. Turning auth ON is
    an action the user takes, and it cannot be turned off by an unset variable.

Tokens are never stored. `key_hash` is SHA-256 of the token; `key_prefix` is
the first 11 characters, kept in clear ONLY so a human can tell two keys apart
in a list. A token is shown exactly once, at creation.

SHA-256 rather than bcrypt/argon2 is deliberate and is safe *here*: these are
128-bit random machine-generated tokens, not user-chosen passwords. There is no
dictionary to attack, so the slow-hash property buys nothing, and every request
would pay for it.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid

import psycopg2
import psycopg2.extras
from fastapi import HTTPException

DB_URL = os.environ.get("DB_URL")

# The pre-013 shared token. Still honoured so something2 does not break during
# the migration, but it no longer disables auth when absent.
LEGACY_TOKEN = os.environ.get("SPRITE_API_TOKEN", "").strip()

TOKEN_PREFIX = "sk_"
PREFIX_LEN = 11  # "sk_" + 8 characters

ALL_SCOPES = ("read", "generate", "admin")


def _db():
    return psycopg2.connect(DB_URL)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_token() -> str:
    """A 256-bit URL-safe token. `secrets`, never `random`."""
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

def active_key_count() -> int:
    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM api_keys WHERE revoked_at IS NULL")
            return int(cur.fetchone()[0])
    except Exception:
        # A database that cannot be reached must not silently become open mode.
        # Reporting "enforced" makes requests fail closed.
        return 1


def is_enforced() -> bool:
    return bool(LEGACY_TOKEN) or active_key_count() > 0


def describe_mode() -> dict:
    """For the settings UI, so 'is my API open?' has a visible answer."""
    keys = active_key_count()
    enforced = bool(LEGACY_TOKEN) or keys > 0
    return {
        "enforced": enforced,
        "active_keys": keys,
        "legacy_token": bool(LEGACY_TOKEN),
        "message": (
            f"Authentication enforced ({keys} active key"
            f"{'s' if keys != 1 else ''}"
            f"{', plus legacy token' if LEGACY_TOKEN else ''})."
            if enforced else
            "API is OPEN - anyone who can reach this port can queue GPU work. "
            "Create a key to enforce authentication."
        ),
    }


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def create_key(name: str, scopes: list[str] | None = None) -> dict:
    """Mint a key. The returned `token` is the ONLY time it is ever available.

    THE BOOTSTRAP KEY ALWAYS GETS `admin`, even if not asked for.

    Found by walking into it: minting the first key with only read+generate
    flipped the API to enforced mode, and then nothing could revoke or replace
    that key, because key management needs `admin` and no admin key existed.
    The API was locked with its own credential.

    A first key that cannot manage keys is never what anyone wants, so it is
    not offered. `bootstrap: true` comes back in the response so the caller can
    say why the scopes differ from what it asked for.
    """
    scopes = list(scopes or ["read", "generate"])
    bad = [s for s in scopes if s not in ALL_SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s): {', '.join(bad)}; "
                         f"known: {', '.join(ALL_SCOPES)}")

    bootstrap = active_key_count() == 0 and not LEGACY_TOKEN
    if bootstrap and "admin" not in scopes:
        scopes.append("admin")

    token = _new_token()
    key_id = uuid.uuid4()
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (id, name, key_hash, key_prefix, scopes) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(key_id), name, hash_token(token), token[:PREFIX_LEN], scopes))
    return {"id": str(key_id), "name": name, "scopes": scopes,
            "key_prefix": token[:PREFIX_LEN], "token": token,
            "bootstrap": bootstrap}


def list_keys() -> list[dict]:
    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, key_prefix, scopes, created_at, last_used_at, "
            "       revoked_at "
            "FROM api_keys ORDER BY created_at DESC")
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["id"] = str(d["id"])
            for f in ("created_at", "last_used_at", "revoked_at"):
                d[f] = d[f].isoformat() if d[f] else None
            d["revoked"] = d["revoked_at"] is not None
            out.append(d)
        return out


def revoke_key(key_id: str) -> bool:
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = now() "
            "WHERE id = %s::uuid AND revoked_at IS NULL", (key_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _principal_from_token(token: str) -> dict | None:
    if LEGACY_TOKEN and secrets.compare_digest(token, LEGACY_TOKEN):
        return {"id": None, "name": "legacy SPRITE_API_TOKEN",
                "scopes": list(ALL_SCOPES)}

    with _db() as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, scopes FROM api_keys "
            "WHERE key_hash = %s AND revoked_at IS NULL", (hash_token(token),))
        row = cur.fetchone()
        if not row:
            return None
        # Best-effort; a failed timestamp update must not fail the request.
        try:
            cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s",
                        (row["id"],))
        except Exception:
            pass
        return {"id": str(row["id"]), "name": row["name"],
                "scopes": list(row["scopes"])}


def require(authorization: str | None, scope: str = "read") -> dict:
    """Authorise a request, or raise 401/403. Returns the calling principal.

    In open mode returns a synthetic principal marked `open`, so a caller that
    wants to warn about it can.
    """
    # A scope this server does not define is a programming error, not a
    # permission problem. Without this it degrades silently into
    # "admin only" - the `scope not in principal["scopes"]` test can never
    # pass for a name no key can hold - and reports 403 "lacks the '<typo>'
    # scope", sending the operator off to create a key with a scope
    # add_key() will reject. Raised before the open-mode return so a
    # keyless machine finds it too; that is where the one real instance of
    # this hid (maintenance.py asked for "write").
    if scope not in ALL_SCOPES:
        raise RuntimeError(
            f"unknown scope {scope!r} requested by this endpoint; "
            f"known: {', '.join(ALL_SCOPES)}")

    if not is_enforced():
        return {"id": None, "name": "anonymous (open mode)",
                "scopes": list(ALL_SCOPES), "open": True}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token. Send 'Authorization: Bearer <token>'.")

    principal = _principal_from_token(authorization[7:].strip())
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked token")

    if scope not in principal["scopes"] and "admin" not in principal["scopes"]:
        raise HTTPException(
            status_code=403,
            detail=f"Key '{principal['name']}' lacks the '{scope}' scope")

    principal["open"] = False
    return principal


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
#
# Kept in this module rather than a separate one: four endpoints that are
# entirely about API keys, sharing every helper above. Splitting them would
# mean a file that imports everything from here and adds nothing.

from fastapi import APIRouter, Header  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

router = APIRouter()


class NewKey(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    scopes: list[str] = Field(default_factory=lambda: ["read", "generate"])


@router.get("/api/auth/mode")
def auth_mode():
    """Whether the API is open. Unauthenticated on purpose: a client must be
    able to discover that it needs a token without already having one."""
    return describe_mode()


@router.get("/api/auth/keys")
def get_keys(authorization: str | None = Header(None)):
    require(authorization, "admin")
    return {"keys": list_keys(), "mode": describe_mode()}


@router.post("/api/auth/keys", status_code=201)
def post_key(body: NewKey, authorization: str | None = Header(None)):
    """Mint a key. The token is in the response and nowhere else, ever.

    Note the bootstrap: while in open mode `require` permits this, which is how
    the first key gets created. Creating it flips the API to enforced, so the
    window is exactly one call wide and closes by being used.
    """
    require(authorization, "admin")
    try:
        return create_key(body.name, body.scopes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/auth/keys/{key_id}")
def delete_key(key_id: str, authorization: str | None = Header(None)):
    require(authorization, "admin")
    if not revoke_key(key_id):
        raise HTTPException(status_code=404, detail="No such active key")
    return {"revoked": key_id, "mode": describe_mode()}
