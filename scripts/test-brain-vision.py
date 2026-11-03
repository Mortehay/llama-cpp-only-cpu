#!/usr/bin/env python3
"""Can the brain actually see, and does it REFUSE when it cannot?

WHY THIS EXISTS

The Qwen3VL language GGUF has no vision tensors - 399 tensors, prefixes
blk/output/output_norm/token_embd, no `v.` tower, no `mm.` projector. Only
that half was fetched in 2026-09, so `brain_engine` could load a model named
Qwen3VL, answer about an image it had never received, and sound exactly as
confident as if it had looked. Nothing anywhere said otherwise.

That is the failure this guards: not "vision is broken" but "vision is absent
and the answers keep coming". A missing projector must be a refusal, never a
plausible verdict.

SIX CHECKS, all CPU-only - no brain is started, no card is touched, so this is
safe to run while a build holds the GPU.

  1. has_vision() is True only when the projector file is really on disk
  2. the 35B brain declares no projector and reports no vision
  3. --mmproj reaches the llama-server command line, and only with the file
  4. image_message() builds OpenAI content parts with a base64 data URI
  5. image_message() raises on a missing file instead of sending text alone
  6. reference_qc refuses with VisionUnavailable when vision is absent

MUTATION RESULT (proved able to fail, 2026-11-03). The guard was stripped in a
copy of brain_engine and `BRAIN_ENGINE_SRC` pointed at it; the real file was
never touched:

    `if has_vision(brain):` removed from around the --mmproj append
        -> both assertions in check 3 go red ("NO SUCH GUARD")
        -> exit 1 mutated, 0 real

The first version of check 3 would NOT have caught that. It rebuilt `ensure`'s
argv here, the same way the source does, and asserted against its own
reconstruction - so it tested the test. It now parses the source. Note also
that shadowing the module via PYTHONPATH cannot mutate this suite, because the
script puts /app at the front of sys.path itself and the real module wins
silently; that false negative was observed before `BRAIN_ENGINE_SRC` existed.

WHAT THIS SUITE DOES NOT COVER: whether the verdict is CORRECT. Changing
judge_one's `ok` to read `whole_figure` instead of the two edge-derived fields
leaves every check here green. That logic is measured against hand-labelled
images by scripts/judge-references.py, which needs the card and the weights.

    docker compose run --rm --entrypoint python sprite-worker \
        /app/scripts/test-brain-vision.py
"""

import os
import sys

sys.path.insert(0, "/app")

import brain_engine  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        FAILS.append(label)


def main() -> int:
    vl, big = "qwen3-vl-8b", "qwen3.6-35b-a3b"
    mm = brain_engine.BRAINS[vl].get("mmproj")

    print("\n== 1. has_vision() tracks the file on disk ==")
    on_disk = bool(mm) and os.path.exists(mm)
    check(brain_engine.has_vision(vl) == on_disk,
          "has_vision agrees with the filesystem",
          f"mmproj={mm} exists={on_disk}")
    if not on_disk:
        print("        NOTE: projector absent, so this run exercises the "
              "REFUSAL path. Run `make provision` to cover the other half.")

    print("\n== 2. the 35B brain declares no projector ==")
    check(not brain_engine.BRAINS[big].get("mmproj"),
          "35B has no mmproj key")
    check(brain_engine.has_vision(big) is False,
          "has_vision(35B) is False")

    print("\n== 3. --mmproj reaches the command line, and only with the file ==")
    # READ THE SOURCE, do not rebuild the command here. An earlier version of
    # this check reconstructed `ensure`'s argv the way the source does and
    # then asserted against its own reconstruction, so deleting the guard in
    # brain_engine would have left it green - it tested the test. `ensure`
    # cannot simply be called, because it STARTS llama-server and this suite
    # must not touch the card.
    import ast
    # Parse the whole module and pick the function out of it: dedenting a
    # single function's source to make it parse standalone is the fiddly way
    # and gets the line numbers wrong.
    # Overridable so a candidate - or a deliberately mutated copy, to prove
    # this check can fail - can be inspected without touching the real file.
    # Note PYTHONPATH cannot do that job here: this script puts /app at the
    # front of sys.path itself, so an import-shadowing trick silently loads
    # the real module and the check passes while appearing to be mutated.
    src_path = os.environ.get("BRAIN_ENGINE_SRC") or brain_engine.__file__
    print(f"        (static check reads {src_path})")
    mod = ast.parse(open(src_path, encoding="utf-8").read())
    tree = next(n for n in mod.body
                if isinstance(n, ast.FunctionDef) and n.name == "ensure")
    guarded = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "has_vision" in ast.unparse(n.test)
        and "--mmproj" in ast.unparse(n)
    ]
    check(bool(guarded),
          "`--mmproj` is appended only inside an `if has_vision(...)` block",
          "found it under: "
          f"{ast.unparse(guarded[0].test) if guarded else 'NO SUCH GUARD'}")
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == "--mmproj"]
    check(len(bare) == len([n for n in ast.walk(guarded[0])
                            if isinstance(n, ast.Constant)
                            and n.value == "--mmproj"]) if guarded else False,
          "no unguarded `--mmproj` anywhere else in ensure()",
          f"{len(bare)} occurrence(s) total")
    check(brain_engine.has_vision(big) is False,
          "and the 35B brain reports no vision, so the guard excludes it")

    print("\n== 4. image_message builds OpenAI content parts ==")
    png = "/tmp/_vision_probe.png"
    with open(png, "wb") as f:
        # Smallest valid PNG; the encoder must not care what is inside.
        f.write(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000"
            "001f15c4890000000a49444154789c6300010000050001"
            "0d0a2db40000000049454e44ae426082"))
    msg = brain_engine.image_message("describe this", [png])
    parts = msg.get("content")
    check(msg.get("role") == "user" and isinstance(parts, list) and len(parts) == 2,
          "one text part and one image part",
          f"parts={[p.get('type') for p in parts]}")
    check(parts[0] == {"type": "text", "text": "describe this"},
          "the text part is first and verbatim")
    url = parts[1].get("image_url", {}).get("url", "")
    check(url.startswith("data:image/png;base64,") and len(url) > 40,
          "the image is a base64 data URI, not a path",
          f"url[:38]={url[:38]!r}")
    check("/tmp" not in url,
          "no filesystem path leaks into the request",
          "a path would be resolved in the SERVER's filesystem")

    print("\n== 5. a missing image is a refusal, not a text-only message ==")
    try:
        brain_engine.image_message("x", ["/tmp/does-not-exist-9e3f.png"])
        check(False, "raises on a missing image")
    except FileNotFoundError:
        check(True, "raises FileNotFoundError on a missing image")
    except Exception as e:  # noqa: BLE001
        check(False, "raises FileNotFoundError on a missing image", f"got {e!r}")

    print("\n== 6. reference_qc refuses when the brain cannot see ==")
    import reference_qc
    saved = brain_engine.BRAINS[vl].get("mmproj")
    try:
        brain_engine.BRAINS[vl]["mmproj"] = "/models/definitely-absent.gguf"
        try:
            reference_qc.judge_one(png)
            check(False, "VisionUnavailable raised with no projector")
        except reference_qc.VisionUnavailable:
            check(True, "VisionUnavailable raised with no projector")
        except Exception as e:  # noqa: BLE001
            check(False, "VisionUnavailable raised with no projector",
                  f"got {type(e).__name__}: {e}")
    finally:
        brain_engine.BRAINS[vl]["mmproj"] = saved

    print("\n== 7. the ledger flattens image content instead of storing a PNG ==")
    flat = brain_engine._prompt_text([msg])
    check("describe this" in flat and "base64" not in flat,
          "content parts flatten to text plus an image count",
          f"-> {flat!r}")
    check(brain_engine._prompt_text([{"role": "user", "content": "plain"}]) == "plain",
          "a plain string message is unchanged")

    os.path.exists(png) and os.remove(png)
    print("\nFAILURES:", "none" if not FAILS else "")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
