#!/usr/bin/env python3
"""The model gateway's rules, table-tested. No Redis, no GPU, no database.

`model_gateway.decide` is pure on purpose so that the part that is easy to get
subtly wrong - who may take the card, and when - is checked without a worker.
The Redis shell around it is thin and exercised live, not here.

Also a static check: every gated task name must exist as a real task, or a
typo in GATED would silently leave that path ungated.

Run: make test-model-gateway
"""

import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src",
                                "sprite_generator"))

import model_gateway as g  # noqa: E402

SDXL, QWEN, AUDIO = "sdxl+nerijs", "gguf:qwen", g.AUDIO_STABLE
NOW = 10_000.0
IDLE = 300
failures = []


def check(name, got, want_action):
    action = got[0]
    ok = action == want_action
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {action}"
          + (f" ({got[1]})" if got[1] else ""))
    if not ok:
        failures.append(f"{name}: wanted {want_action}, got {got}")


def active(model, pinned=False, idle_for=0):
    return {"model": model, "pinned": pinned, "since": NOW - 1000,
            "last_activity": NOW - idle_for}


def job(model, age=0, running=False):
    return {"model": model, "queued_at": NOW - age, "running": running}


d = g.decide

check("nothing active yet -> switch",
      d(SDXL, "t1", None, {"t1": job(SDXL)}, NOW, IDLE), "switch")
check("job for the active model -> run",
      d(SDXL, "t1", active(SDXL), {"t1": job(SDXL), "t2": job(SDXL)}, NOW, IDLE),
      "run")
check("other model, active has queued work -> defer",
      d(QWEN, "t1", active(SDXL), {"t1": job(QWEN, 10), "t2": job(SDXL, 5)},
        NOW, IDLE), "defer")
check("other model, active idle and unpinned -> switch",
      d(QWEN, "t1", active(SDXL), {"t1": job(QWEN)}, NOW, IDLE), "switch")
check("pinned, queue empty but idle < window -> defer",
      d(QWEN, "t1", active(SDXL, pinned=True, idle_for=60), {"t1": job(QWEN)},
        NOW, IDLE), "defer")
check("pinned, idle past the window -> switch",
      d(QWEN, "t1", active(SDXL, pinned=True, idle_for=301), {"t1": job(QWEN)},
        NOW, IDLE), "switch")
check("pinned, idle past window but its own job queued -> defer",
      d(QWEN, "t1", active(SDXL, pinned=True, idle_for=400),
        {"t1": job(QWEN), "t2": job(SDXL)}, NOW, IDLE), "defer")
check("older waiting job for a third model goes first -> defer",
      d(AUDIO, "t1", active(SDXL), {"t1": job(AUDIO, 5), "t2": job(QWEN, 50)},
        NOW, IDLE), "defer")
check("the oldest waiting job itself -> switch",
      d(QWEN, "t2", active(SDXL), {"t1": job(AUDIO, 5), "t2": job(QWEN, 50)},
        NOW, IDLE), "switch")
check("same-model waiters do not block each other -> switch",
      d(QWEN, "t1", active(SDXL), {"t1": job(QWEN, 5), "t2": job(QWEN, 50)},
        NOW, IDLE), "switch")
check("starvation cap: unpinned stream yields after IDLE -> switch",
      d(QWEN, "t1", active(SDXL), {"t1": job(QWEN, 301), "t2": job(SDXL, 1)},
        NOW, IDLE), "switch")
check("starvation cap does not apply while pinned -> defer",
      d(QWEN, "t1", active(SDXL, pinned=True),
        {"t1": job(QWEN, 301), "t2": job(SDXL, 1)}, NOW, IDLE), "defer")
check("facade (no task id), active busy -> defer",
      d(QWEN, None, active(SDXL), {"t2": job(SDXL, running=True)}, NOW, IDLE),
      "defer")
check("facade, same model while busy -> run",
      d(SDXL, None, active(SDXL), {"t2": job(SDXL, running=True)}, NOW, IDLE),
      "run")
check("facade, active idle -> switch",
      d(QWEN, None, active(SDXL), {}, NOW, IDLE), "switch")

# retry_after_s: a pinned model reports what is left of its window.
ra = g.retry_after_s(active(SDXL, pinned=True, idle_for=100), NOW, IDLE)
print(f"{'ok  ' if ra == 201 else 'FAIL'} retry_after for pinned: {ra}")
if ra != 201:
    failures.append(f"retry_after_s pinned: {ra} != 201")

# Model labels for the fixed-model tasks, and the kwargs path send_task uses.
for name, args, kwargs, want in [
    ("tasks.generate_raw_task", ("p", "n", QWEN, 1, 1, 1, 1, 1), {}, QWEN),
    ("tasks.generate_core_task", ("p",), {"llm_name": SDXL}, SDXL),
    ("tasks.generate_audio_task", (), {"kind": "music", "name": "x"},
     g.AUDIO_MUSIC),
    ("tasks.generate_audio_task", (), {"kind": "ambience", "name": "x"},
     g.AUDIO_STABLE),
    ("tasks.build_sheet_job", ("j",), {}, g.QWEN_EDIT),
    ("tasks.run_command_job", ("x",), {}, None),
]:
    got = g.model_of(name, args, kwargs)
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} model_of {name}: {got}")
    if not ok:
        failures.append(f"model_of {name}: {got} != {want}")

# preloadable() vs warmable(): only diffusers checkpoints are preloaded through
# get_sd_pipeline (which refuses gguf), but a switch to the persistent fast
# Qwen does real loading too - it starts qwen_server.py - so its switch is
# timed and shown. The 20-step Q3_K_M and the fixed labels only free the card.
FAST = "gguf:qwen-image-2512-Q2_K+lightning8"
SLOW = "gguf:Qwen-Image-2512-Q3_K_M"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0+nerijs/pixel-art-xl"
os.environ.pop("QWEN_PERSISTENT", None)
for name, got, want in [
    ("preloadable(SDXL)", g.preloadable(SDXL), True),
    ("preloadable(fast gguf)", g.preloadable(FAST), False),
    ("warmable(SDXL)", g.warmable(SDXL), True),
    ("warmable(fast gguf)", g.warmable(FAST), True),
    ("warmable(slow gguf)", g.warmable(SLOW), False),
    ("warmable(audio label)", g.warmable(g.AUDIO_STABLE), False),
]:
    ok = got == want
    print(("ok  " if ok else "FAIL"), f"{name} -> {got}")
    if not ok:
        failures.append(f"{name}: {got} != {want}")
os.environ["QWEN_PERSISTENT"] = "0"
if g.warmable(FAST):
    failures.append("QWEN_PERSISTENT=0 should make the fast gguf not warmable")
    print("FAIL QWEN_PERSISTENT=0 still warmable")
else:
    print("ok   QWEN_PERSISTENT=0 turns the persistent path off")
os.environ.pop("QWEN_PERSISTENT", None)

# The brain (decisions/0013): its own label per brain, gated, sync, and a
# switch to it is a real load. A brain asking while a UI-pinned image model
# is warm must defer - the brain gets no special priority.
import brain_engine as be  # noqa: E402
B35, B8 = be.label("qwen3.6-35b-a3b"), be.label("qwen3-vl-8b")
for want, args, kwargs in ((B8, (), {"brain": "qwen3-vl-8b"}),
                           (B35, ("qwen3.6-35b-a3b",), {}),
                           (be.label(be.default_brain()), (), {})):
    got = g.model_of("tasks.generate_text_task", args, kwargs)
    if got != want:
        failures.append(f"model_of text task {args}{kwargs} -> {got}, want {want}")
        print(f"FAIL model_of text task -> {got}, want {want}")
ok = (all(lbl in g.FIXED_LABELS for lbl in (B35, B8))
      and "tasks.generate_text_task" in g.SYNC_TASKS
      and not g.preloadable(B8) and g.warmable(B8))
if not ok:
    failures.append("brain labels: FIXED_LABELS / SYNC_TASKS / preloadable / warmable")
    print("FAIL brain labels not wired into FIXED_LABELS / SYNC_TASKS / warmable")
else:
    print("ok   brain labels fixed, sync, warmable, not preloadable")
check("brain defers to a pinned, warm image model",
      g.decide(B8, None, active(SDXL, pinned=True, idle_for=10), {}, NOW, IDLE),
      "defer")
check("brain takes an idle, unpinned card",
      g.decide(B8, None, active(SDXL, idle_for=10), {}, NOW, IDLE), "switch")
check("second brain request runs on the active brain",
      g.decide(B8, None, active(B8), {}, NOW, IDLE), "run")
check("image job defers while a brain job is running",
      g.decide(SDXL, "t2", active(B8), {"t1": job(B8, running=True)}, NOW, IDLE),
      "defer")

# Static: every GATED name is a real task name in the source.
src = os.path.dirname(g.__file__)
declared = set()
for fn in ("tasks.py", "map_tasks.py"):
    with open(os.path.join(src, fn), encoding="utf-8") as fh:
        declared |= set(re.findall(r'name="([\w.]+)"', fh.read()))
for name in g.GATED:
    if name not in declared:
        failures.append(f"GATED names {name}, which no task declares")
        print(f"FAIL GATED {name} is not a declared task")
print(f"ok   {len(g.GATED)} gated task names all declared"
      if all(n in declared for n in g.GATED) else "")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall model-gateway checks passed")
