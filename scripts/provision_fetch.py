"""The actual fetching, shared by the CLI provisioner and the background daemon.

Imported by `provision-models.py` (interactive, whole tiers) and by
`provision-daemon.py` (one entry at a time, queued from the UI). It lives in
its own underscored module for the same reason `models_manifest.py` does: both
callers need it, their filenames are hyphenated and therefore not importable,
and a second copy of "how do you fetch each kind" is exactly the drift this
whole lane exists to prevent.

Reporting is injected (`say`) rather than printed, because the two callers want
different destinations - a terminal, and a log line prefixed per daemon job.

Nothing here takes the cache lock. Locking is the CALLER's job: the CLI holds
it across a whole tier so its plan cannot be invalidated mid-run, while the
daemon holds it per entry so a long queue does not starve a CLI run. Putting it
in here would force one of those to be wrong.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import target_path  # noqa: E402


def fetch_cache(entry: dict, root: str, token: str | None) -> None:
    """A repo in HF cache layout (models--org--name/), resolved by repo id."""
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

    `local_dir=`, not `cache_dir=`. Via the cache the file lands in the blob
    store and must be copied out, costing its size TWICE - 34 GB of peak disk
    for the 17 GB brain, on a host with ~60 GB free.

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
    """A component layout another project owns, e.g. acestep/<component>/."""
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


FETCHERS = {"hf-cache": fetch_cache,
            "hf-file": fetch_file,
            "checkpoint-tree": fetch_tree}


def fetch_all(entries: list[dict], root: str, token: str | None,
              say=print) -> list[tuple[str, Exception]]:
    """Fetch each entry in turn. -> [(name, exception)] for the failures.

    One failure must not strand the rest: a 401 on the single gated repo
    should still leave a machine with every other weight on disk.
    """
    failed: list[tuple[str, Exception]] = []
    for i, m in enumerate(entries, 1):
        say(f"[{i}/{len(entries)}] {m['name']} - {m.get('repo', '')}")
        if m.get("provenance") == "substitute":
            say("NOTE: not the exact file this project was benched against; "
                "see `notes` in the manifest.")
        fetcher = FETCHERS.get(m.get("kind"))
        if fetcher is None:
            # `local` and `fetch = "self"` are filtered out by the callers;
            # reaching here means the manifest grew a kind nobody handles.
            failed.append((m["name"],
                           ValueError(f"no fetcher for kind {m.get('kind')!r}")))
            continue
        try:
            fetcher(m, root, token)
            say(f"[OK] {m['name']}")
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((m["name"], e))
            say(f"[FAILED] {m['name']}: {type(e).__name__}: {e}")
            if m.get("gated"):
                say("         gated repo - have you accepted its terms on the Hub?")
    return failed


def hub_size(entry: dict, token: str | None) -> int | None:
    """Authoritative size for an hf-file, straight from the Hub."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        url = hf_hub_url(repo_id=entry["repo"], filename=entry["file"])
        return get_hf_file_metadata(url, token=token).size
    except Exception:
        return None


def resolve_files(entry: dict,
                  token: str | None) -> tuple[list[tuple[str, int]], str | None]:
    """What a repo entry would ACTUALLY fetch, and how big, from the Hub.

    Applies this entry's allow/ignore through `filter_repo_objects` - the same
    function snapshot_download uses - against the repo's real file list. So
    this answers "did my globs do what I meant" without downloading anything,
    which is the only cheap way to check a filter whose failure mode is a
    clean download that dies at LOAD time.

    Also the only honest source of size for an hf-cache entry: `du` on a
    models--* directory under-reports badly, because snapshots are symlinks
    into a shared, xet-backed blob store.

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
