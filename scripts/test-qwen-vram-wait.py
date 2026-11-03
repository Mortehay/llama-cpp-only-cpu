#!/usr/bin/env python3
"""The Qwen VRAM gate waits out a park instead of refusing. No GPU, no Redis.

WHY THIS EXISTS

`_qwen_image` refuses a Qwen-Image request when the card has less than
`QWEN_MIN_FREE_MB` free. The number was right and the MOMENT was wrong: the
persistent child (`qwen_server.py`) writes `RESULT` and only then moves the
transformer and VAE back to host RAM, so a caller that sends its next image
straight away is measured in the middle of that transfer.

THE MEASUREMENT THIS LOCKS IN (sprite_worker's log, 2026-11-02)

Eight refusals that day, and every single one landed 1.2-3.4 s AFTER a
successful image, reading 9.0-9.1 GiB free against the 9.8 GiB wanted:

    12:43:00,556  qwen server: {... 'place_s': 2.8, 'total_s': 54.49}
    12:43:01,584  generate_raw_task succeeded
    12:43:01,772  refused: 9.0 GiB free of 12.0, needs 9.8 GiB

The same card measured idle reads 10.98 GiB free, and spaced requests ran at
~56 s for half an hour with no refusal at all. So nothing else held the card -
the 2-4 minutes something2 lost per art batch bought nothing.

A second process loading a second copy of the model was the first hypothesis
and it was wrong: `generate_raw_task` is the only txt2img route, the worker is
`--pool=solo`, and there is exactly one persistent child. Hence a WAIT, not
routing work.

MUTATION RESULT, so nobody has to trust that this test can fail

Replacing `_wait_free_vram`'s loop condition with `if True:` - that is,
deleting the wait and restoring the behaviour something2 reported:

    replays the measured park and still generates   FAIL (9.00 GiB, 0.00s)
    the park costs under 5s of waiting              FAIL (waited 0.00s)
    and refuses within limit_s                      FAIL (gave up at 0.00s)

Three of eleven, and the three that name the bug. `replays the measured park`
is the anti-goal case: it feeds the exact readings from that log and fails if
the gate gives up on them.

The function's source is lifted out of `tasks.py` with `ast` rather than
imported: importing `tasks` drags in celery, torch and the database for two
functions whose only real dependency is a clock.

Run: make test-qwen-vram-wait
"""

import ast
import os
import re
import sys
import types

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src",
                   "sprite_generator")
SRC = SRC if os.path.isdir(SRC) else "/app"

GIB = 2 ** 30
MIB = 2 ** 20
TOTAL = int(11.99951171875 * GIB)   # what this 12 GB card actually reports
NEED_MB = 10000                     # QWEN_MIN_FREE_MB
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


# --- lift _card_holder and _wait_free_vram out of tasks.py ------------------

class Clock:
    """A fake clock, so a 30 s wait costs the test nothing."""

    def __init__(self):
        self.t = 0.0
        self.slept = 0

    def time(self):
        return self.t

    def sleep(self, s):
        self.slept += 1
        self.t += s


def load(clock):
    """Exec the two functions in a namespace of stubs. Returns the namespace."""
    with open(os.path.join(SRC, "tasks.py"), encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    wanted = {"_card_holder", "_wait_free_vram"}
    picked = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in picked} != wanted:
        raise SystemExit(f"tasks.py no longer defines {wanted} at module "
                         f"level; found {[n.name for n in picked]}")

    # `_card_holder` does `import brain_engine` for one `alive()` call.
    sys.modules.setdefault("brain_engine",
                           types.SimpleNamespace(alive=lambda b=None: False))
    notes = []
    ns = {
        "time": clock,
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
            mem_get_info=lambda: (_ for _ in ()).throw(
                AssertionError("the default probe must not be reached")))),
        "logger": types.SimpleNamespace(info=lambda *a: None,
                                        debug=lambda *a: None),
        "model_gateway": types.SimpleNamespace(
            note_card=lambda *a: notes.append(a)),
        "pipes": {},
        "_qwen_server_alive": lambda model=None: True,
        "_qwen_srv": {"proc": object(), "model": "gguf:qwen-image-2512-Q2_K"},
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<tasks>", "exec"),
         ns)
    ns["_notes"] = notes
    return ns


def run(readings, limit_s=30.0, need_mb=NEED_MB):
    """Drive the gate over a list of free-byte readings; the last one repeats."""
    clock = Clock()
    ns = load(clock)
    seen = []

    def probe():
        free = readings[min(len(seen), len(readings) - 1)]
        seen.append(free)
        return free, TOTAL

    free, total, waited = ns["_wait_free_vram"](need_mb, limit_s, "test",
                                                probe=probe)
    return {"free": free, "total": total, "waited": waited,
            "polls": len(seen), "slept": clock.slept, "notes": ns["_notes"],
            "holder": ns["_card_holder"]()}


# --- the cases -------------------------------------------------------------

# 1. A free card is not slowed down by the wait existing.
r = run([int(10.98 * GIB)])
check("an idle card returns at once", r["waited"] == 0 and r["slept"] == 0,
      f"waited {r['waited']}s over {r['polls']} poll(s), {r['slept']} sleep(s)")
check("an idle card passes the gate", r["free"] >= NEED_MB * MIB,
      f"{r['free'] / GIB:.2f} GiB free")

# 2. THE ANTI-GOAL. The exact readings from 2026-11-02: a park in flight, then
#    the card back to its idle figure. The old gate refused on the first one.
PARK = [int(9.0 * GIB), int(9.1 * GIB), int(9.1 * GIB), int(10.98 * GIB)]
r = run(PARK)
check("replays the measured park and still generates",
      r["free"] >= NEED_MB * MIB,
      f"{r['free'] / GIB:.2f} GiB free after {r['waited']:.2f}s")
check("the park costs under 5s of waiting", 0 < r["waited"] < 5,
      f"waited {r['waited']:.2f}s")

# 3. A card that is genuinely held still refuses - bounded, not hanging.
r = run([int(3.0 * GIB)], limit_s=30.0)
check("a held card still refuses", r["free"] < NEED_MB * MIB,
      f"{r['free'] / GIB:.2f} GiB free")
check("and refuses within limit_s", 30.0 <= r["waited"] < 31.0,
      f"gave up after {r['waited']:.2f}s, {r['polls']} polls")

# 4. limit_s=0 is the mutant: it reproduces the bug this test is about.
r = run(PARK, limit_s=0.0)
check("limit_s=0 reproduces the old refusal (so the case can fail)",
      r["free"] < NEED_MB * MIB and r["waited"] == 0,
      f"{r['free'] / GIB:.2f} GiB free, waited {r['waited']}s")

# 5. Both outcomes publish a reading for GET /api/model-gateway, and the
#    holder is named rather than guessed.
r = run(PARK)
check("publishes one card reading per call", len(r["notes"]) == 1,
      f"{len(r['notes'])} note(s): {r['notes']}")
check("names the persistent child as the holder",
      "persistent Qwen child" in r["holder"] and "llm-server" not in r["holder"],
      r["holder"])

# --- static: the two ends of the 503 must not drift apart ------------------

with open(os.path.join(SRC, "tasks.py"), encoding="utf-8") as fh:
    tasks_src = fh.read()
with open(os.path.join(SRC, "a1111.py"), encoding="utf-8") as fh:
    a1111_src = fh.read()

check("tasks.py tags the refusal vram_busy on both paths",
      tasks_src.count('"error_kind": "vram_busy"') == 2,
      f"{tasks_src.count(chr(34) + 'error_kind' + chr(34) + ': ' + chr(34) + 'vram_busy' + chr(34))} site(s)")
check("the refusal carries retry_after_s",
      tasks_src.count('"retry_after_s": QWEN_VRAM_RETRY_S') == 2)
busy = re.search(r'if kind == "vram_busy":(.*?)\n        if kind ==',
                 a1111_src, re.S)
check("a1111 maps vram_busy to 503 + Retry-After",
      bool(busy) and "status = 503" in busy.group(1)
      and "Retry-After" in busy.group(1),
      "the facade must not answer 500 for an occupied card")
check("nothing still blames llm-server for a busy card",
      "usually llm-server's chat model" not in tasks_src,
      "the retired llm_engine lane is not the holder any more")
check("a1111 sends the kind as a header too",
      'headers = {**(headers or {}), "X-Error-Kind": kind}' in a1111_src,
      "so a caller need not regex the prose")

# --- the two phrases something2's drains match on --------------------------
#
# Their art and audio drains classify "card busy" by matching BOTH
# /Not enough free VRAM/ and /holds the card/ over `detail`, and three answers
# they cannot classify stop a batch as "provider looks down". So these are
# contract text, not prose. An earlier draft of the gate said "Holding the
# card:" and would have broken them silently (caught 2026-11-02).
#
# Checked against the SOURCE of both refusals - the gate in tasks.py and the
# child's own in qwen_server.py. The child's reaches them verbatim through
# `_qwen_server_request`, and its lower-case "not enough free VRAM" never
# matched the first pattern at all.
with open(os.path.join(SRC, "qwen_server.py"), encoding="utf-8") as fh:
    child_src = fh.read()

for where, src in (("the gate", tasks_src), ("the Qwen child", child_src)):
    check(f"{where} says 'Not enough free VRAM'",
          '"Not enough free VRAM' in src or 'f"Not enough free VRAM' in src,
          "something2 matches this phrase; rewording it is a contract change")
    check(f"{where} says 'holds the card'", "holds the card" in src,
          "something2 matches this phrase too, and both must be present")

# The gate recognises the child's refusal case-insensitively, or capitalising
# the child's message would silently stop it being classed as vram_busy.
check("the gate detects the child's refusal case-insensitively",
      '"not enough free vram" in err.lower()' in tasks_src,
      "a capitalised child message must still be seen")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall qwen-vram-wait checks passed")
