#!/usr/bin/env python3
"""Can a caller ask for the HQ lane, and can they tell whether they got it?

WHY THIS EXISTS

Two failure modes, both silent, both already on record in this repo for other
fields:

  1. THE FIELD IS DROPPED. Pydantic discards undeclared fields and
     `model_dump()` is what reaches the worker, so a caller asking for
     `comfyui-hq` would be served the fast lane with nothing in the response
     saying so. `concept_check` did exactly this - a caller followed the
     failure message's own advice, re-sent the flag, and got the identical
     refusal because the model had never declared it.
  2. THE LANE FALLS BACK. An HQ job quietly rendered with diffusers produces a
     sheet indistinguishable from the one the caller already had, after hours
     of waiting. There must be no fallback path, and the atlas must say what
     ran.

These are structural checks: no GPU, no ComfyUI, no network.

EIGHT CHECKS

  1. a bare spec is "diffusers" and the field survives model_dump
  2. an unknown engine is rejected at validation, naming the valid set
  3. unavailable_reason is None for diffusers and specific for a missing HQ
  4. provenance echoes the fast lane's real quant and its two LoRAs
  5. provenance takes the values ACTUALLY used, not the configured defaults
  6. loras are {name, scale} objects, never bare strings
  7. the gateway gives the two lanes DIFFERENT labels
  8. engines.py contains no fallback-to-diffusers path

MUTATION RESULT (proved able to fail, 2026-11-03): removing `engine` from
JobSpec reddens checks 1 and 2; returning `{"name": "diffusers"}` from
provenance for comfyui-hq reddens 5 and 6; pointing both gateway labels at
QWEN_EDIT reddens 7.

    docker compose run --rm --entrypoint python sprite-worker \
        /app/scripts/test-engine-contract.py
"""

import re
import sys

sys.path.insert(0, "/app")

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILS.append(label)


def main() -> int:
    import engines
    import jobs
    import model_gateway

    print("\n== 1. the default lane, and the field survives model_dump ==")
    s = jobs.JobSpec()
    check(s.engine == "diffusers", "a bare spec is 'diffusers'",
          "every existing something2 integration omits the field")
    check("engine" in s.model_dump(),
          "engine survives model_dump", "this is what reaches the worker")
    check(jobs.DEFAULT_ENGINE == "diffusers",
          "DEFAULT_ENGINE is the fast lane")

    print("\n== 2. an unknown engine is rejected, with the valid set named ==")
    try:
        jobs.JobSpec(engine="sdxl-turbo")
        check(False, "unknown engine rejected")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        check(True, "unknown engine rejected")
        check("comfyui-hq" in msg and "diffusers" in msg,
              "the error names the valid engines", msg.splitlines()[1][:90]
              if len(msg.splitlines()) > 1 else msg[:90])
    check(jobs.JobSpec(engine="comfyui-hq").engine == "comfyui-hq",
          "a known engine is accepted")

    print("\n== 3. availability is answered separately from validity ==")
    check(engines.unavailable_reason("diffusers") is None,
          "diffusers is always available")
    why = engines.unavailable_reason("comfyui-hq")
    if why is None:
        check(True, "comfyui-hq reports available",
              "weights present and server answering")
    else:
        check(bool(why) and len(why) > 20,
              "comfyui-hq gives a SPECIFIC reason, not a boolean",
              why[:100])
        check("not on disk" in why or "not answering" in why,
              "the reason names weights or the server")

    print("\n== 4. the fast lane's provenance describes the fast lane ==")
    p = engines.provenance("diffusers")
    check(p["name"] == "diffusers", "name")
    check(p["model_file"].endswith(".gguf"), "model_file is the real quant",
          p["model_file"])
    check(p["precision"] == "gguf", "precision")
    check(p["cfg"] == 1.0,
          "cfg is 1.0 - the Lightning LoRA forces it, which is why that lane "
          "cannot have a negative prompt")
    names = [l["name"] for l in p["loras"]]
    check(any("angles" in n for n in names) and
          any("Lightning" in n for n in names),
          "both LoRAs are echoed", str(names))

    print("\n== 5. provenance echoes what was USED, not the defaults ==")
    p2 = engines.provenance("comfyui-hq", steps=37, cfg=6.5,
                            model_file="other.safetensors", precision="q8_0")
    check(p2["steps"] == 37 and p2["cfg"] == 6.5,
          "passed steps/cfg win over the configured defaults",
          f"steps={p2['steps']} cfg={p2['cfg']}")
    check(p2["model_file"] == "other.safetensors" and p2["precision"] == "q8_0",
          "passed model_file/precision win too")
    check(engines.provenance("comfyui-hq")["name"] == "comfyui-hq",
          "the HQ lane does not report itself as diffusers")

    print("\n== 6. loras are objects with a scale ==")
    for lane in ("diffusers", "comfyui-hq"):
        ls = engines.provenance(lane)["loras"]
        check(all(isinstance(x, dict) and "name" in x and "scale" in x
                  for x in ls),
              f"{lane}: every lora is {{name, scale}}",
              "the scale is load-bearing - angles_scale pins pose as well as "
              "camera")

    print("\n== 7. the two lanes are DIFFERENT gateway labels ==")
    check(model_gateway.QWEN_EDIT != model_gateway.QWEN_EDIT_HQ,
          "distinct labels",
          f"{model_gateway.QWEN_EDIT} vs {model_gateway.QWEN_EDIT_HQ}")
    check("tasks.build_sheet_job" in model_gateway.GATED
          and "tasks.redraw_frames_job" in model_gateway.GATED,
          "both sheet tasks are gated",
          "a GPU task missing from GATED bypasses the gateway silently")

    print("\n== 8. there is no fallback to diffusers ==")
    src = open(engines.__file__, encoding="utf-8").read()
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    # A fallback would look like assigning the default engine inside a failure
    # branch. The docstring says there is none; this checks the code agrees.
    suspicious = re.findall(r"except[^\n]*\n(?:[^\n]*\n){0,3}?[^\n]*"
                            r"return\s+[\"']diffusers[\"']", body)
    check(not suspicious,
          "no 'return \"diffusers\"' inside a failure branch",
          "a lane that silently degrades is indistinguishable from a lane "
          "that is broken")

    print("\nFAILURES:", "none" if not FAILS else "")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
