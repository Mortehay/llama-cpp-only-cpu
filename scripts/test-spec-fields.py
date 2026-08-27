#!/usr/bin/env python3
"""Can a caller's flag reach the worker, or does the model drop it in silence?

WHY THIS EXISTS

A request model and the task that reads its output are written apart. Pydantic
ignores undeclared fields, `model_dump()` emits only declared ones, and the
task reads what it wants with `spec.get(name, default)`. Put those three
together and a field the caller sent vanishes, a plausible default takes its
place, and NOTHING between them can tell the difference - not the endpoint,
which validated happily, and not the task, which got a value.

That is not hypothetical here. `JobSpec` never declared `concept_check`, while
`build_sheet_job` read `spec.get("concept_check", True)` and its refusal
message said, in as many words, "Pass concept_check=false to build anyway".
A caller who hit the concept check, followed the instruction and re-sent the
job got the identical refusal. The documented escape hatch for a wide creature
or a vehicle had never existed. Measured, before the fix:

    caller sent           {"prompt": "...", "concept_check": false}
    model_dump keys       [actions, cell, colors, concept_image, directions,
                           frames, prompt, seed, style_profile]
    what the task read    True

A peer found the same pairing in the map path, where `Terrain` never declared
`walkable`: every terrain in every tilemap the service ever produced came back
walkable, water included, and three guards that depended on it returned
answers indistinguishable from a legitimate "nothing here".

THREE CHECKS

The behavioural one sends a flag through the real model and asserts it
survives to the dump. It fails the moment the field stops being declared.

The static one is the general form, and it is the one that would have caught
this without anyone suspecting the field: for each (model, reader function)
pair below, every key the reader pulls out of the spec with a default must be
declared on the model. It needs no server and no database.

`tiles.py` is the counter-example the static check must NOT flag: it augments
the dump explicitly - `{**spec.model_dump(), "tile_h": h}` - for values the
SERVER computes. A key added that way is not a dropped one, so the pairs below
name the reader function rather than sweeping every `.get` in the file.

The third asks the RUNNING service. Both of the others read source, and both
pass in a fresh interpreter against a fixed file while the live process still
runs the old one - which is what happened on the map side: the walkable fix
was committed and its suite 24/24 green while the API, up for four hours,
still dropped the field. `/openapi.json` is the process describing its own
models and needs no key. It reports a SKIP, loudly, when the API is not
reachable; a skip that reads like a pass is the failure this file is about.
"""

import ast
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(_here, "..", "src", "sprite_generator"), "/app"]
PKG = next((p for p in CANDIDATES
            if os.path.isfile(os.path.join(p, "jobs.py"))), None)

# (module holding the model, model, module holding the reader, reader, var)
PAIRS = [
    ("jobs.py", "JobSpec", "tasks.py", "build_sheet_job", "spec"),
]


def model_fields(path, model):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == model:
            return {st.target.id for st in node.body
                    if isinstance(st, ast.AnnAssign)
                    and isinstance(st.target, ast.Name)}
    return None


def keys_read(path, func, var):
    """`var.get("k", <default>)` inside `func`.

    Two-argument gets only. A one-argument get returns None and the caller has
    to handle it, which is visible in the code; it is the silent default that
    hides the drop.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == func), None)
    if fn is None:
        return None
    out = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out[node.args[0].value] = ast.unparse(node.args[1])[:30]
    return out


def static_check(fails):
    for mod, model, rmod, func, var in PAIRS:
        fields = model_fields(os.path.join(PKG, mod), model)
        if not fields:
            fails.append("%s:%s not found or declares nothing - this check is "
                         "reading the wrong file" % (mod, model))
            continue
        read = keys_read(os.path.join(PKG, rmod), func, var)
        if read is None:
            fails.append("%s:%s() not found" % (rmod, func))
            continue
        print("  %s declares %d; %s() reads %d with a default"
              % (model, len(fields), func, len(read)))

        # A pattern that matches nothing reports a clean run over an empty set.
        # This pairing is the whole subject of the file.
        if not read:
            fails.append("%s() reads no `%s.get(k, default)` at all - the "
                         "pairing this check exists for has moved, so a clean "
                         "result here means nothing" % (func, var))
            continue

        for key, dflt in sorted(read.items()):
            if key not in fields:
                print("  FAIL %r read with default %s but %s does not declare "
                      "it" % (key, dflt, model))
                fails.append("%s() reads %r (default %s); %s drops it, so the "
                             "caller's value never arrives and the default "
                             "answers instead" % (func, key, dflt, model))


def behaviour_check(fails):
    """Send the flag through the real model and see whether it comes out."""
    sys.path.insert(0, PKG)
    try:
        from jobs import JobSpec
    except Exception as e:
        fails.append("could not import JobSpec (%s); run this in the "
                     "container, where its dependencies exist" % e)
        return

    dumped = JobSpec(prompt="a wide siege wagon",
                     concept_check=False).model_dump()
    if "concept_check" not in dumped:
        fails.append("concept_check=False did not survive model_dump(); the "
                     "task reads its default, and the refusal message tells "
                     "the caller to send exactly this")
        return
    if dumped["concept_check"] is not False:
        fails.append("concept_check survived as %r, not False"
                     % dumped["concept_check"])
        return
    print("  concept_check=False survives model_dump()")

    # And the default still holds when nobody asks, or the check stops running
    # for everyone - the opposite failure, which a fixture that only ever sent
    # False could not see.
    if JobSpec(prompt="x").model_dump().get("concept_check") is not True:
        fails.append("concept_check defaults to something other than True; "
                     "omitting it must still run the check")
    else:
        print("  omitted, it still defaults to True")


def served_check(fails):
    """What the RUNNING process serves, which is not what is on disk.

    Both checks above read source. Both pass in a fresh interpreter against a
    fixed file while the live service still runs the old one - which is exactly
    what happened: the walkable fix was committed and 24/24 green while the API
    had been up for four hours and was still dropping the field. A peer caught
    it by asking the process instead of reasoning from timestamps.

    /openapi.json is the running process describing its own models, and it
    needs no key. It answers "what is served", which no source-reading check
    can.

    WHAT IT DOES NOT ANSWER is why served and disk differ, and that limit has
    already caught someone out: a peer read one such reading as "the process is
    stale", restarted, saw it agree, and wrote up a diagnosis. Neither of us
    could then reproduce any staleness - this API reloads on a poll, hundreds of
    times a day - so the reading was right and the cause was invented. Report
    the difference; do not name a mechanism for it.

    The retry below is not politeness. A reload takes about 7 seconds
    detect-to-serving and the API is genuinely unreachable for part of it, so
    running this straight after an edit - which is exactly when someone runs it
    - would skip spuriously. Measured during a reload: +3s unreachable, +6s
    serving the new schema.
    """
    url = os.environ.get("SPRITE_API", "http://sprite-generator:8001")

    def fetch():
        import urllib.request
        with urllib.request.urlopen(url + "/openapi.json", timeout=10) as r:
            return json.load(r)["components"]["schemas"]

    try:
        schemas = fetch()
    except Exception as first:
        print("  unreachable at %s (%s); a reload takes ~7s, so waiting once"
              % (url, type(first).__name__))
        time.sleep(9)
        try:
            schemas = fetch()
            print("  reachable on retry - that was a reload, not a dead API")
        except Exception as e:
            # NOT a pass. A skip that reads like a pass is the failure this
            # whole file is about, so it says what it could not do and why.
            print("  SKIPPED: still no API after 9s (%s)" % type(e).__name__)
            print("  This says NOTHING about what the running service serves.")
            print("  Set SPRITE_API, or run this where the API is reachable.")
            return

    for mod, model, _rmod, _func, _var in PAIRS:
        declared = model_fields(os.path.join(PKG, mod), model) or set()
        served = set(schemas.get(model, {}).get("properties", {}))
        if not served:
            fails.append("%s is not in the served schema at all" % model)
            continue
        missing = sorted(declared - served)
        print("  %s serves %d properties" % (model, len(served)))
        if missing:
            fails.append("%s declares %s in source but the RUNNING service "
                         "does not serve %s - the process is older than the "
                         "code; restart it"
                         % (model, ", ".join(missing), ", ".join(missing)))
        else:
            print("  everything the source declares is served")


def main():
    if PKG is None:
        print("could not find sprite_generator/jobs.py; looked in:")
        for c in CANDIDATES:
            print("   ", os.path.abspath(c))
        return 1
    print("package:", os.path.abspath(PKG))

    fails = []
    print("\n== every key a task reads is declared on the model ==")
    static_check(fails)
    print("\n== a caller's flag survives validation ==")
    behaviour_check(fails)
    print("\n== the RUNNING service serves what the source declares ==")
    served_check(fails)

    print("\nFAILURES:", "none" if not fails else "")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
