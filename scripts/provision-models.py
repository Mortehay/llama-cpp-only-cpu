#!/usr/bin/env python3
"""Fetch everything `models.toml` declares into MODELS_DIR.

    python provision-models.py [--tier required|optional|all] [--dry-run]
                               [--only NAME[,NAME...]] [--manifest PATH]
                               [--models-dir DIR] [--yes]

WHERE THIS RUNS
Inside the `downloader` container, which has huggingface_hub, /models mounted
and no torch. `make provision` does that for you. It deliberately does NOT run
as part of `make up`: two containers downloading into the same /models cache
deadlocked on HuggingFace's file locks ("Still waiting to acquire lock on
/models/.cache/huggingface/download/..."), which is why the downloader service
idles on `tail -f /dev/null`.

WHY NOT `hf download`
Because three of the four kinds in the manifest cannot be expressed as one.
An hf-cache entry must land in HF cache layout (models--org--name/) so
diffusers can resolve it by repo id; `--local-dir` produces a flat tree that
nothing finds. An hf-file entry needs an explicit destination two levels down,
sometimes under a DIFFERENT filename than the Hub's. See `.ai/domain.md` and
`.ai/decisions/0014`.

SAFE TO RE-RUN. Anything already present is skipped, so an interrupted run
resumes. Nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import (  # noqa: E402
    CacheBusy, cache_root, have, load, provision_lock, select)
from provision_fetch import fetch_all, hub_size, resolve_files  # noqa: E402

DEFAULT_MANIFEST = "/usr/local/bin/models.toml"
TIERS = ("required", "optional", "all")


# --- reporting ----------------------------------------------------------
# Plain ASCII and flush-as-you-go: this streams through `docker compose run`
# into a terminal that may not be a tty, beside huggingface_hub's own bars.

def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def gb(n: float | int | None) -> str:
    return "     ?" if not n else f"{n / 1024 ** 3:7.2f}G"


def _report(msg: str) -> None:
    """Reporter handed to provision_fetch.fetch_all.

    Its per-entry header ("[2/5] name - repo") gets the banner treatment that
    separated entries in the output before the fetch loop moved out of this
    file; everything else is a plain line.
    """
    if msg.startswith("[") and "] " in msg and "/" in msg.split("]")[0]:
        rule(msg)
    else:
        say(msg)


# --- main ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS, default="required")
    ap.add_argument("--only", default="", help="comma-separated manifest names")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only; queries the Hub for real sizes")
    ap.add_argument("--audit", action="store_true",
                    help="resolve EVERY selected entry against the Hub, "
                         "including ones already on disk: real sizes, and "
                         "proof that each allow/ignore selects what it should")
    ap.add_argument("--files", action="store_true",
                    help="with --dry-run/--audit, list the files each glob selects")
    ap.add_argument("--manifest", default=os.environ.get("MODELS_MANIFEST", DEFAULT_MANIFEST))
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    doc = load(args.manifest)
    root = cache_root(doc, args.models_dir)

    token = os.environ.get("HF_TOKEN") or None
    # Never echo the token: this runs under `make` and lands in docker logs.
    say(f"manifest   : {args.manifest}")
    say(f"models dir : {root}")
    say(f"HF_TOKEN   : {'set (' + str(len(token)) + ' chars)' if token else 'NOT SET'}")
    say(f"tier       : {args.only or args.tier}")

    if not os.path.isdir(root):
        sys.exit(f"\nmodels dir does not exist: {root}\n"
                 f"Inside the container this is the /models bind mount; check "
                 f"MODELS_DIR in compose/develop/.env.")

    models = doc.get("model", [])
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    chosen = select(models, args.tier, only)
    if not chosen:
        say("\nnothing selected")
        return 0

    # --- plan -----------------------------------------------------------
    rule("PLAN")
    todo, skipped, manual = [], [], []
    for m in chosen:
        if m.get("kind") == "local":
            manual.append(m)
        elif have(root, m):
            skipped.append(m)
        elif m.get("fetch") == "self":
            manual.append(m)
        else:
            todo.append(m)

    for m in skipped:
        say(f"  present   {m['name']:<26} {m.get('repo', m.get('dest', ''))}")
    for m in manual:
        why = "trained here, cannot be downloaded" if m.get("kind") == "local" \
              else "fetched by the owning project on first run"
        say(f"  MANUAL    {m['name']:<26} {why}")
    # --audit re-resolves everything, present or not: the point is to check
    # the globs and sizes of entries that are ALREADY on disk, which the plan
    # would otherwise skip without ever looking at them.
    audited = chosen if args.audit else todo
    total = 0
    for m in (audited if args.audit else todo):
        size = m.get("bytes")
        if (args.dry_run or args.audit) and m.get("repo"):
            if m.get("kind") == "hf-file":
                size = hub_size(m, token) or size
            else:
                files, err = resolve_files(m, token)
                if err:
                    say(f"  !! {m['name']:<26} cannot resolve: {err}")
                    if m.get("gated"):
                        say("      gated - have its terms been accepted for this token?")
                else:
                    size = sum(s for _, s in files)
                    if args.files:
                        say(f"  -- {m['name']}: {len(files)} file(s) selected by "
                            f"allow={m.get('allow')} ignore={m.get('ignore')}")
                        for fn, sz in sorted(files, key=lambda x: -x[1]):
                            say(f"       {gb(sz)}  {fn}")
        if args.audit:
            mark = "present" if have(root, m) else "MISSING"
            say(f"  {mark:<8} {gb(size)}  {m['name']:<26} {m.get('repo', '')}")
            total += size or 0
            continue
        if not size and m.get("approx_gb"):
            size = float(m["approx_gb"]) * 1024 ** 3
        total += size or 0
        flag = " GATED" if m.get("gated") else ""
        say(f"  FETCH    {gb(size)}  {m['name']:<26} {m.get('repo', '')}{flag}")

    say()
    if args.audit:
        say(f"  {len(audited)} entries resolved, {total / 1024 ** 3:.1f} GB total "
            f"as declared (allow/ignore applied)")
        say("  audit only - nothing downloaded")
        return 0
    say(f"  {len(skipped)} present, {len(manual)} manual, {len(todo)} to fetch, "
        f"~{total / 1024 ** 3:.1f} GB")

    gated = [m for m in todo if m.get("gated")]
    if gated:
        say()
        say("  GATED repos below need the model terms accepted on huggingface.co")
        say("  with the SAME account as HF_TOKEN, or the fetch returns 401:")
        for m in gated:
            say(f"    {m['repo']}")

    if manual:
        say()
        say("  MANUAL entries are not fetched by this script. See their `notes`")
        say("  in the manifest for what to do instead.")

    if args.dry_run:
        say("\ndry run - nothing downloaded")
        return 0
    if not todo:
        say("\nnothing to do")
        return 0
    if not token and gated:
        sys.exit("\nHF_TOKEN is not set and the plan includes gated repos. "
                 "Set it in compose/develop/.env and retry.")

    if not args.yes and sys.stdin.isatty():
        try:
            if input(f"\nfetch {len(todo)} item(s), ~{total / 1024 ** 3:.1f} GB? [y/N] "
                     ).strip().lower() not in ("y", "yes"):
                say("aborted")
                return 1
        except EOFError:
            pass

    # --- fetch ----------------------------------------------------------
    # Under the shared cache lock for the WHOLE run, not per entry: the
    # background daemon (scripts/provision-daemon.py) writes the same cache,
    # and two writers deadlock on huggingface_hub's per-file locks. Taking it
    # once also means an interleaved daemon download cannot land between two
    # of these entries and make "11 present, 2 to fetch" a lie mid-run.
    #
    # Waits rather than failing: a CLI run that loses a race to a UI-queued
    # download should queue behind it, not abort. The daemon is the one that
    # gives up quickly, because it can simply try the row again.
    try:
        with provision_lock(root, holder="provision-models (cli)",
                            timeout_s=float(os.environ.get(
                                "PROVISION_LOCK_WAIT_S", "900"))):
            failed = fetch_all(todo, root, token, say=_report)
    except CacheBusy as e:
        say(f"\n{e}")
        say("Another provisioning run or a UI-queued download holds the cache.")
        return 1

    rule("RESULT")
    say(f"  fetched {len(todo) - len(failed)} of {len(todo)}")
    for name, e in failed:
        say(f"  FAILED  {name}: {e}")
    if failed:
        say("\n  Re-run to retry; anything already on disk is skipped.")
        return 1
    say("\n  Verify with: make models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
