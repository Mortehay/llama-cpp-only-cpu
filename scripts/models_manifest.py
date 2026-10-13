"""Shared reader for `models.toml`: parsing and the "is it there?" question.

Imported by BOTH `provision-models.py` and `check-models-manifest.py`.

It exists as its own module for one reason: "is this model present" is subtle
enough to get wrong (a bare cache directory is what an interrupted download
leaves behind, and hf-file entries are not cache directories at all), and two
copies of it would drift. Drift between two copies of one fact is the bug this
whole manifest exists to prevent - see `.ai/decisions/0014`. Underscored, not
hyphenated, because the other two import it.
"""

from __future__ import annotations

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        sys.exit("need Python 3.11+ for tomllib, or `pip install tomli`")

KINDS = ("hf-cache", "hf-file", "checkpoint-tree", "local")
TIERS = ("required", "optional")


def load(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        sys.exit(f"manifest not found: {path}")
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"manifest is not valid TOML: {e}")


def cache_root(doc: dict, override: str | None = None) -> str:
    meta = doc.get("meta", {})
    return (override
            or os.environ.get(meta.get("cache_root_env", "HF_HUB_CACHE") or "")
            or meta.get("cache_root_default", "/models"))


def cache_dir_for(root: str, repo: str) -> str:
    return os.path.join(root, "models--" + repo.replace("/", "--"))


def have_cache(root: str, repo: str) -> bool:
    """True when the repo has a non-empty snapshot AND no partial blobs.

    Checking snapshots/ rather than the repo directory is what distinguishes a
    real download from the directory an interrupted one leaves behind - the
    same test core_models.is_cached() makes.

    The `.incomplete` check is STRICTER than core_models, and deliberately.
    huggingface_hub populates snapshots/ with symlinks early, so a repo that is
    still downloading answers "present" to the snapshot test alone - observed
    2026-10-13, when a dry run called `openpose-sdxl` present while two of its
    blobs were still in flight. For the runtime that is harmless (nothing tries
    to load it yet); for a provisioner it is the worst possible answer, because
    a resumed run would skip the repo and leave a model that exists and cannot
    load. Treat in-flight as absent and let snapshot_download resume it.
    """
    repo_dir = cache_dir_for(root, repo)
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    try:
        if not any(r.is_dir() and any(os.scandir(r.path))
                   for r in os.scandir(snapshots)):
            return False
    except OSError:
        return False
    blobs = os.path.join(repo_dir, "blobs")
    try:
        return not any(e.name.endswith(".incomplete") for e in os.scandir(blobs))
    except OSError:
        return True  # no blobs/ dir at all: xet-backed, nothing in flight here


def saved_name(entry: dict) -> str:
    """The filename an hf-file entry is stored under.

    `as` overrides the Hub's own name, and two entries need it: the Hub
    publishes `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` where BRAIN_8B_PATH expects
    `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, and CORE_MODELS capitalises the
    Qwen-Image-2512 Q3_K_M stem. Downloading under the Hub's name leaves a file
    that exists and is never found.
    """
    return entry.get("as") or entry["file"]


def target_path(root: str, entry: dict) -> str:
    return os.path.join(root, entry.get("dest", ""), saved_name(entry))


def have(root: str, entry: dict) -> bool:
    kind = entry.get("kind")
    if kind == "hf-cache":
        return have_cache(root, entry["repo"])
    if kind == "hf-file":
        p = target_path(root, entry)
        return os.path.isfile(p) and os.path.getsize(p) > 0
    if kind in ("checkpoint-tree", "local"):
        d = os.path.join(root, entry.get("dest", ""))
        if not os.path.isdir(d):
            return False
        if kind == "local":
            return all(os.path.isfile(os.path.join(d, f))
                       for f in entry.get("files", []))
        try:
            return any(os.scandir(d))
        except OSError:
            return False
    return False


class CacheBusy(RuntimeError):
    """Another process is already writing this /models cache."""


def provision_lock(root: str, holder: str, timeout_s: float = 0.0):
    """Exclusive lock over one /models cache, shared by every writer.

    WHY: huggingface_hub takes per-file locks, and two processes downloading
    into the same cache deadlocked on them - "Still waiting to acquire lock on
    /models/.cache/huggingface/download/..." - which is why the downloader
    service idles instead of fetching on startup. The background daemon is a
    single consumer and so serialises itself, but `make provision` runs as its
    own container and knows nothing about it. The lock is what makes those two
    paths safe together, so BOTH take it.

    A file in the cache directory rather than a Redis key, because the cache
    is the shared thing: any container that can write the models dir can see
    it, including a one-off `docker compose run`, and it cannot drift out of
    sync with the resource it protects.

    Uses flock, so the lock dies with the process. A container killed mid
    download therefore does not strand it - which matters, because the
    alternative (a stale lock file holding a timestamp) would need an
    expiry guess, and guessing wrong either blocks forever or lets two
    writers in.

    Returns a context manager. Raises CacheBusy rather than waiting, unless
    `timeout_s` is given.
    """
    import contextlib
    import fcntl
    import time as _time

    @contextlib.contextmanager
    def _cm():
        path = os.path.join(root, ".provision.lock")
        os.makedirs(root, exist_ok=True)
        fh = open(path, "a+")
        try:
            deadline = _time.monotonic() + max(0.0, timeout_s)
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if _time.monotonic() >= deadline:
                        fh.seek(0)
                        who = (fh.read() or "").strip() or "another process"
                        raise CacheBusy(
                            f"{root} is being written by {who}; "
                            f"wait for it rather than running two writers")
                    _time.sleep(0.5)
            fh.seek(0)
            fh.truncate()
            fh.write(f"{holder} pid={os.getpid()}\n")
            fh.flush()
            try:
                yield
            finally:
                fh.seek(0)
                fh.truncate()
                fh.flush()
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    return _cm()


def select(models: list[dict], tier: str, only: set[str]) -> list[dict]:
    if only:
        picked = [m for m in models if m.get("name") in only]
        missing = only - {m.get("name") for m in picked}
        if missing:
            sys.exit("--only named entries that are not in the manifest: "
                     + ", ".join(sorted(missing)))
        return picked
    if tier == "all":
        return list(models)
    return [m for m in models if m.get("tier") == tier]
