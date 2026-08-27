#!/usr/bin/env python3
"""Does every endpoint ask for a scope this server can actually grant?

WHY THIS EXISTS

`auth.require(authorization, scope)` compares the requested scope against the
scopes on the caller's key, and passes an admin key unconditionally:

    if scope not in principal["scopes"] and "admin" not in principal["scopes"]:
        raise HTTPException(403, ...)

A scope name that is not in `ALL_SCOPES` therefore does not fail loudly. It
degrades into "admin only" - no key can hold a name `add_key` refuses to
store - and every non-admin caller is told it "lacks the 'write' scope", a
scope that does not exist and that they cannot be given. The operator's next
move is to create a key with it, which is rejected. The endpoint looks broken
and the message points away from the cause.

That is not hypothetical. `POST /api/commands/{name}` asked for "write" from
the day it was written. It was exercised only in open mode, where `require`
returns a synthetic principal BEFORE the comparison, so the mistake was
invisible to every test that ran it.

TWO CHECKS, AND WHY BOTH

`require` now raises RuntimeError on an unknown scope. That is the runtime
guard, and it fires when an endpoint is called. It says nothing about an
endpoint nobody called - which is exactly the situation that hid this one.

So the first check here is static: read every `auth.require(...)` and
`_require_auth(...)` literal out of the source and compare it against
`ALL_SCOPES`. It needs no server, no database and no key, and it covers
endpoints that have never been hit.

The second check is that the runtime guard actually raises, including in open
mode. A guard placed after the `is_enforced()` return would pass the static
check and still let a keyless machine sail past, so its position is asserted
rather than assumed.
"""

import ast
import os
import re
import sys
import textwrap

_here = os.path.dirname(os.path.abspath(__file__))

# Host layout is <repo>/src/sprite_generator; inside the container the package
# IS /app. Try both rather than guess, the same way test-train-prep.py does.
CANDIDATES = [os.path.join(_here, "..", "src", "sprite_generator"), "/app"]
PKG = next((p for p in CANDIDATES
            if os.path.isfile(os.path.join(p, "auth.py"))), None)

# `auth.require(authorization, "read")` and the two thin wrappers that forward
# a scope, `_require_auth(authorization, "read")`. Both spellings, one pattern.
CALL = re.compile(r'(?:auth\.require|_require_auth)\(\s*[^,()]+,\s*"([^"]+)"')

# Matches ALL_SCOPES = ("read", "generate", "admin") without importing auth,
# which would need psycopg2 and a reachable database to import at all.
DECL = re.compile(r'^ALL_SCOPES\s*=\s*\(([^)]*)\)', re.M)


def declared_scopes(src):
    m = DECL.search(src)
    if not m:
        return None
    return tuple(s.strip().strip('"\'') for s in m.group(1).split(",")
                 if s.strip())


def static_check(fails):
    """Every scope literal in the package names a scope that exists."""
    auth_src = open(os.path.join(PKG, "auth.py"), encoding="utf-8").read()
    scopes = declared_scopes(auth_src)
    if not scopes:
        fails.append("could not find ALL_SCOPES in auth.py - this test is "
                     "reading the wrong file, or the declaration moved")
        return
    print("  ALL_SCOPES =", ", ".join(scopes))

    sites = 0
    for name in sorted(os.listdir(PKG)):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(PKG, name), encoding="utf-8").read()
        for m in CALL.finditer(src):
            sites += 1
            asked = m.group(1)
            line = src[:m.start()].count("\n") + 1
            if asked not in scopes:
                print("  FAIL %s:%d asks for %r" % (name, line, asked))
                fails.append("%s:%d requests unknown scope %r" %
                             (name, line, asked))

    # A pattern that matches nothing would report a clean run over an empty
    # set. The repo has dozens of these; anything near zero means the regex
    # stopped matching, not that the endpoints stopped checking.
    print("  scope literals found: %d" % sites)
    if sites < 20:
        fails.append("only %d call sites matched - the regex has drifted away "
                     "from how these calls are written, so a clean result here "
                     "means nothing" % sites)


def runtime_check(fails):
    """The guard raises, and raises BEFORE the open-mode shortcut."""
    src = open(os.path.join(PKG, "auth.py"), encoding="utf-8").read()

    # Position, read from the source: the unknown-scope check has to come
    # before `if not is_enforced()`, or a keyless machine never reaches it.
    guard = src.find("if scope not in ALL_SCOPES")
    body = src.find("def require(")
    openm = src.find("if not is_enforced():", body)
    if guard < 0:
        fails.append("no unknown-scope guard in require()")
    elif not (body < guard < openm):
        fails.append("the unknown-scope guard is not inside require() before "
                     "the open-mode return, so open mode skips it")
    else:
        print("  guard sits inside require(), ahead of the open-mode return")

    # And that it actually raises. auth.py imports psycopg2 and fastapi at
    # module scope, so neither importing it nor slicing the text by hand works
    # here - `require` is followed by a comment block, a `from fastapi import`
    # and a decorated route, and every text rule I tried swept one of them in.
    # `ast` gives the function's exact line span from a parse that runs no
    # imports at all.
    lines = src.splitlines(True)
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "require"), None)
    if fn is None:
        fails.append("auth.py has no require() to execute")
        return
    body = textwrap.dedent("".join(lines[fn.lineno - 1:fn.end_lineno]))

    ns = {"ALL_SCOPES": ("read", "generate", "admin"),
          "is_enforced": lambda: False,
          "HTTPException": RuntimeError,
          "_principal_from_token": lambda t: None}
    exec(compile(body, "auth.require", "exec"), ns)
    require = ns["require"]

    try:
        require(None, "write")
    except RuntimeError as e:
        if "write" in str(e):
            print("  open mode, unknown scope -> RuntimeError(%s)"
                  % str(e)[:60])
        else:
            fails.append("raised, but the message does not name the scope")
    except Exception as e:
        fails.append("unknown scope raised %s, not RuntimeError" % type(e))
    else:
        fails.append("unknown scope was ACCEPTED in open mode")

    # The guard must not break the scopes that do exist.
    for good in ("read", "generate", "admin"):
        try:
            p = require(None, good)
        except Exception as e:
            fails.append("open mode rejected the real scope %r: %s" % (good, e))
        else:
            if not p.get("open"):
                fails.append("open mode returned a principal not marked open")


def main():
    if PKG is None:
        print("could not find sprite_generator/auth.py; looked in:")
        for c in CANDIDATES:
            print("   ", os.path.abspath(c))
        return 1
    print("package:", os.path.abspath(PKG))

    fails = []
    print("\n== every scope literal names a real scope ==")
    static_check(fails)
    print("\n== the guard fires, and fires early enough ==")
    runtime_check(fails)

    print("\nFAILURES:", "none" if not fails else "")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
