#!/usr/bin/env python3
"""Mint or list API keys from the command line, bypassing HTTP auth.

WHY THIS EXISTS

Key management needs the `admin` scope, which is a credential problem: revoke
your only admin key and the HTTP API can no longer mint a replacement. That is
the correct behaviour for the API - and the reason every system with API keys
also has an out-of-band path (`createsuperuser`, `htpasswd`, a root shell).

The trust boundary here is database access. Anyone who can run this can already
read `api_keys` and write rows to it; this just makes the supported way easier
than the unsupported one.

Run it inside the container, which is where DB_URL is set:

    docker exec sprite_generator python /app/../scripts/mint-key.py --list

or, since scripts/ is not mounted into the container, more usefully:

    docker exec -e KEY_NAME=recovery sprite_generator python -c \
        "import auth; print(auth.create_key('recovery', ['read','generate','admin'])['token'])"

Usage:
    python mint-key.py --list
    python mint-key.py --name recovery
    python mint-key.py --name something2 --scopes read,generate
    python mint-key.py --revoke <key-id>
    python mint-key.py --rotate <key-id>          # same name+scopes, new secret
    ROTATE_TOKEN=sk_... python mint-key.py --rotate-token --quiet
    make api-key                                  # create or rotate .env's key
"""

import argparse
import os
import sys

# The module lives beside the app, not beside this script.
for candidate in ("/app",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "src", "sprite_generator")):
    if os.path.isfile(os.path.join(candidate, "auth.py")):
        sys.path.insert(0, candidate)
        break
else:
    sys.exit("auth.py not found - run this inside the sprite_generator container")

import auth  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", help="name for the new key")
    p.add_argument("--scopes", default="read,generate,admin",
                   help="comma-separated; default gives full control because "
                        "this path exists for recovery")
    p.add_argument("--list", action="store_true")
    p.add_argument("--revoke", metavar="KEY_ID")
    p.add_argument("--rotate", metavar="KEY_ID",
                   help="new secret for this key: same name and scopes, old revoked")
    p.add_argument("--rotate-token", action="store_true",
                   help="rotate the key whose token is in $ROTATE_TOKEN")
    p.add_argument("--quiet", action="store_true",
                   help="print only the new token (for scripts)")
    a = p.parse_args()

    if not os.environ.get("DB_URL"):
        sys.exit("DB_URL is not set - run this inside the container")

    if a.list:
        keys = auth.list_keys()
        if not keys:
            print("no keys; API is in OPEN mode")
            return 0
        print(f"{'id':38} {'prefix':12} {'scopes':26} {'state':8} name")
        for k in keys:
            print(f"{k['id']:38} {k['key_prefix']:12} "
                  f"{','.join(k['scopes']):26} "
                  f"{'revoked' if k['revoked'] else 'active':8} {k['name']}")
        print(f"\n{auth.describe_mode()['message']}")
        return 0

    if a.revoke:
        print("revoked" if auth.revoke_key(a.revoke) else "no such active key")
        return 0

    if a.rotate or a.rotate_token:
        key_id = a.rotate
        if a.rotate_token:
            # The token comes from the environment, never argv: argv is
            # visible in `ps` and shell history.
            # $ROTATE_TOKEN, or stdin (make api-key pipes it in, so the token
            # is not in a `docker exec -e` argument either).
            token = os.environ.get("ROTATE_TOKEN", "") or (
                "" if sys.stdin.isatty() else sys.stdin.read().strip())
            key_id = auth.key_id_for_token(token)
            if not key_id:
                sys.exit("ROTATE_TOKEN is not an active key")
        key = auth.rotate_key(key_id)
        if key is None:
            sys.exit("no such active key")
    else:
        if not a.name:
            p.error("give --name, --list, --revoke, --rotate or --rotate-token")
        scopes = [s.strip() for s in a.scopes.split(",") if s.strip()]
        try:
            key = auth.create_key(a.name, scopes)
        except ValueError as e:
            sys.exit(str(e))

    if a.quiet:
        # For scripts (make api-key): the token alone on stdout.
        print(key["token"])
        return 0
    if key.get("rotated_from"):
        print(f"rotated: {key['rotated_from']} (now revoked)")

    print(f"name:   {key['name']}")
    print(f"id:     {key['id']}")
    print(f"scopes: {', '.join(key['scopes'])}")
    if key.get("bootstrap"):
        print("        (admin added: this is the first key, and a first key "
              "that cannot manage keys locks you out)")
    print(f"\ntoken:  {key['token']}")
    print("\nThis token is not stored and cannot be shown again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
