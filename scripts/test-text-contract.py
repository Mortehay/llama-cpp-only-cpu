#!/usr/bin/env python3
"""/api/text's request rules, table-tested. No Redis, no GPU, no brain.

What is checked here is the part of the contract (.ai/specs/something2-text/)
that decides 422 vs everything else - the client treats 422 as terminal, so
calling a satisfiable schema impossible would silently stop it from ever
asking again. The live path (gate, worker, brain) is exercised with curl.

Run: make test-text-contract
"""

import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src",
                                "sprite_generator"))

import brain_engine  # noqa: E402
import text  # noqa: E402

failures = []

MUSIC = {"type": "object", "additionalProperties": False,
         "required": ["style", "prompt"],
         "properties": {"style": {"type": "string", "enum": [
             "medieval_fantasy", "tavern", "dungeon", "battle", "village"]},
             "prompt": {"type": "string", "maxLength": 600}}}


def expect(name, schema, want_ok):
    try:
        text.validate_schema(schema)
        got_ok, why = True, ""
    except text.TextRefused as e:
        got_ok, why = False, f"{e.status} {e.reason}: {e.detail}"
        if e.status != 422:
            failures.append(f"{name}: refused with {e.status}, not 422")
    ok = got_ok == want_ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" -> {why}" if why else ""))
    if not ok:
        failures.append(name)


expect("music schema is accepted", MUSIC, True)
expect("empty object schema (any JSON) is accepted", {}, True)
expect("anyOf with a satisfiable branch is accepted",
       {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 1}]}, True)
expect("boolean enum value is not counted as an integer",
       {"type": "integer", "enum": [True]}, False)
expect("empty enum is impossible", {"type": "string", "enum": []}, False)
expect("nested empty enum is impossible",
       {"type": "object", "properties": {"style": {"type": "string", "enum": []}}},
       False)
expect("enum of the wrong type is impossible",
       {"type": "string", "enum": [1, 2]}, False)
expect("minItems > maxItems is impossible",
       {"type": "array", "minItems": 3, "maxItems": 1}, False)
expect("required but forbidden property is impossible",
       {"type": "object", "additionalProperties": False, "properties": {},
        "required": ["entity"]}, False)
expect("not a schema at all", {"type": "no-such-type"}, False)
expect("a list is not a schema", [1, 2], False)

# The roster: static, default first, every entry labelled for the gateway.
r = brain_engine.roster()
if not r or not r[0]["default"] or sum(e["default"] for e in r) != 1:
    failures.append("roster: exactly one default, listed first")
    print("FAIL roster default")
else:
    print(f"ok   roster: {[e['id'] for e in r]}, default first")
if brain_engine.label(None) != "brain:" + brain_engine.default_brain():
    failures.append("label(None) is not the default brain's label")
    print("FAIL label(None)")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall text-contract checks passed")
