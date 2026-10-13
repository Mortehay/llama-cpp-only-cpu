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
    cache_root, have, load, select, target_path)

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


# --- fetching -----------------------------------------------------------

def fetch_cache(entry: dict, root: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=entry["repo"],
        cache_dir=root,
        allow_patterns=entry.get("allow") or None,
        ignore_patterns=entry.get("ignore") or None,
        token=token,
    )


def fetch_file(entry: dict, root: str, token: str | None) -> None:
    """A single file placed at `dest/<as or file>`, as a REAL file.

    `local_dir=`, not `cache_dir=`. With cache_dir the file lands in the HF
    cache and has to be copied out, which costs the size TWICE on the way -
    34 GB of peak disk for the 17 GB brain, on a host with ~60 GB free. With
    local_dir there is one copy and no blob store.

    These are also not HF-cache models: nothing resolves them by repo id, only
    by the exact path in brain_engine.BRAINS or core_models.GGUF_DIR.
    """
    from huggingface_hub import hf_hub_download
    dest = os.path.join(root, entry.get("dest", ""))
    os.makedirs(dest, exist_ok=True)
    src = hf_hub_download(repo_id=entry["repo"], filename=entry["file"],
                          local_dir=dest, token=token)
    final = target_path(root, entry)
    if os.path.abspath(src) != os.path.abspath(final):
        # `as`: the Hub's filename is not the one the code looks for. Renamed
        # rather than copied - a second 17 GB would defeat the point.
        os.replace(src, final)


def fetch_tree(entry: dict, root: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    dest = os.path.join(root, entry["dest"])
    os.makedirs(dest, exist_ok=True)
    snapshot_download(
        repo_id=entry["repo"],
        local_dir=dest,
        allow_patterns=entry.get("allow") or None,
        ignore_patterns=entry.get("ignore") or None,
        token=token,
    )


def hub_size(entry: dict, token: str | None) -> int | None:
    """Authoritative size for an hf-file, straight from the Hub."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        url = hf_hub_url(repo_id=entry["repo"], filename=entry["file"])
        return get_hf_file_metadata(url, token=token).size
    except Exception:
        return None


def resolve_files(entry: dict, token: str | None) -> tuple[list[tuple[str, int]], str | None]:
    """What a repo entry would ACTUALLY fetch, and how big, from the Hub.

    Applies this entry's allow/ignore through `filter_repo_objects` - the same
    function snapshot_download uses - against the repo's real file list. So
    this answers "did my globs do what I meant" without downloading anything,
    which is the only cheap way to check a filter whose failure mode is a
    clean download that dies at load time.

    Also the only honest source of size for an hf-cache entry: `du` on a
    models--* directory under-reports badly, because snapshots are symlinks
    into a shared blob store.

    -> ([(filename, bytes)], error message or None)
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import filter_repo_objects
        info = HfApi().model_info(entry["repo"], files_metadata=True, token=token)
        sibs = list(info.siblings or [])
        kept = list(filter_repo_objects(
            sibs,
            allow_patterns=entry.get("allow") or None,
            ignore_patterns=entry.get("ignore") or None,
            key=lambda s: s.rfilename,
        ))
        return [(s.rfilename, s.size or 0) for s in kept], None
    except Exception as e:  # gated-without-terms, network, renamed repo
        return [], f"{type(e).__name__}: {str(e)[:120]}"


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
    # One failure must not strand the rest: a 401 on the one gated repo should
    # still leave a machine with every other weight on disk.
    failed = []
    for i, m in enumerate(todo, 1):
        rule(f"[{i}/{len(todo)}] {m['name']} - {m.get('repo', '')}")
        if m.get("provenance") == "substitute":
            say("NOTE: not the exact file this project was benched against; "
                "see `notes` in the manifest.")
        try:
            kind = m["kind"]
            if kind == "hf-cache":
                fetch_cache(m, root, token)
            elif kind == "hf-file":
                fetch_file(m, root, token)
            elif kind == "checkpoint-tree":
                fetch_tree(m, root, token)
            say(f"[OK] {m['name']}")
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((m["name"], e))
            say(f"[FAILED] {m['name']}: {type(e).__name__}: {e}")
            if m.get("gated"):
                say("         gated repo - have you accepted its terms on the Hub?")

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
