#!/usr/bin/env python3
"""Background model downloads: the one consumer of `model_downloads`.

Runs as the `downloader` service's command, replacing the `tail -f /dev/null`
it used to idle on. The UI enqueues a row; this picks it up, fetches the
manifest entry, and reports progress back to the row.

WHY NOT CELERY
`sprite-worker` runs `--concurrency=1 --pool=solo` on the default queue, so it
runs exactly one task at a time. A 16 GB download submitted there would hold
that slot and block EVERY generation until it finished. A dedicated Celery
worker was the alternative, but `tasks.py` imports torch at module scope, so it
would need a separate torch-free app and a second service for the same
guarantee this loop already gives. See `.ai/decisions/0014`.

ONE WRITER
Two processes downloading into one /models cache deadlock on huggingface_hub's
per-file locks; that is why this service used to refuse to fetch at all. This
loop is a single consumer, so it serialises itself - but `make provision` is a
separate container that knows nothing about it, so every fetch here is taken
under the shared advisory lock in `models_manifest.provision_lock`. If the CLI
holds it, this backs off and leaves the row queued; the next tick retries.

POLLING, DELIBERATELY
A download takes minutes. A 2-second poll against a partial index is not worth
a Redis dependency, and the DB is already required for the row itself.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import (  # noqa: E402
    CacheBusy, cache_dir_for, cache_root, have, load, provision_lock,
    target_path)
from provision_fetch import fetch_all  # noqa: E402

MANIFEST = os.environ.get("MODELS_MANIFEST", "/usr/local/bin/models.toml")
POLL_S = float(os.environ.get("PROVISION_POLL_S", "3"))
# How long a fetch may hold the cache lock before giving up and leaving the row
# queued. Generous: the CLI waits 900s by default, and losing a race should
# cost a retry, not a failed row.
LOCK_WAIT_S = float(os.environ.get("PROVISION_DAEMON_LOCK_WAIT_S", "30"))
DB_URL = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL") or ""


def log(msg: str) -> None:
    print(f"[provision-daemon] {msg}", flush=True)


def _connect():
    import psycopg2
    return psycopg2.connect(DB_URL)


def _exec(sql: str, args: tuple = ()) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, args)


def claim_next() -> dict | None:
    """Take the oldest queued row, atomically.

    SKIP LOCKED and a single-row update, so a second daemon - someone running
    two stacks against one database - cannot claim the same row. That is
    cheap insurance; the file lock is what actually protects the cache.
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE model_downloads SET status = 'running', started_at = now() "
            "WHERE id = ("
            "  SELECT id FROM model_downloads WHERE status = 'queued' "
            "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
            ") RETURNING id::text, entry, bytes_total")
        row = cur.fetchone()
        conn.commit()
    if not row:
        return None
    return {"id": row[0], "entry": row[1], "bytes_total": row[2]}


def measure(root: str, entry: dict) -> int:
    """Bytes on disk for this entry.

    FOLLOWS SYMLINKS, and that is the whole subtlety. An HF snapshot is a tree
    of symlinks into a shared, xet-backed blob store, so walking it with
    `follow_symlinks=False` measures the links and not the weights: the first
    version of this reported **17 KB for a 42 MB repo**, which would have
    rendered as a progress bar stuck at 0%.

    Dedupes by (device, inode) because the blob store is shared - the same blob
    can appear under two snapshot revisions of one repo, and counting it twice
    would push the bar past 100%.

    Still only an estimate for a repo whose blobs were already present from
    another entry (deduped away by the Hub, so never "downloaded" here).
    Exact for hf-file, which is a real single file.
    """
    try:
        if entry.get("kind") == "hf-file":
            p = target_path(root, entry)
            # The in-flight file lives beside the final one while downloading.
            total = 0
            for cand in (p, p + ".incomplete"):
                if os.path.isfile(cand):
                    total += os.path.getsize(cand)
            d = os.path.join(os.path.dirname(p), ".cache", "huggingface",
                             "download")
            if os.path.isdir(d):
                for r, _, fs in os.walk(d):
                    for f in fs:
                        if f.endswith(".incomplete"):
                            total += os.path.getsize(os.path.join(r, f))
            return total
        base = (cache_dir_for(root, entry["repo"]) if entry.get("kind") == "hf-cache"
                else os.path.join(root, entry.get("dest", "")))
        total = 0
        seen: set[tuple[int, int]] = set()
        for r, _, fs in os.walk(base, followlinks=False):
            for f in fs:
                fp = os.path.join(r, f)
                try:
                    st = os.stat(fp)          # follows: we want the BLOB size
                except OSError:
                    continue                  # dangling link mid-download
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue                  # same blob under two revisions
                seen.add(key)
                total += st.st_size
        return total
    except Exception:
        return 0


def run_one(row: dict, doc: dict, root: str) -> None:
    entries = {m["name"]: m for m in doc.get("model", [])}
    entry = entries.get(row["entry"])
    if entry is None:
        _exec("UPDATE model_downloads SET status='failed', error=%s, "
              "finished_at=now() WHERE id=%s",
              (f"'{row['entry']}' is not in the manifest any more", row["id"]))
        return
    if entry.get("kind") == "local":
        _exec("UPDATE model_downloads SET status='failed', error=%s, "
              "finished_at=now() WHERE id=%s",
              ("this model was trained on a machine, not downloaded; "
               "copy MODELS_DIR/loras by hand", row["id"]))
        return
    if entry.get("fetch") == "self":
        _exec("UPDATE model_downloads SET status='failed', error=%s, "
              "finished_at=now() WHERE id=%s",
              ("the owning project fetches this on first use; "
               "nothing to do here", row["id"]))
        return
    if have(root, entry):
        _exec("UPDATE model_downloads SET status='done', bytes_done=%s, "
              "finished_at=now() WHERE id=%s",
              (measure(root, entry), row["id"]))
        log(f"{row['entry']}: already present")
        return

    # Progress reporting from a thread: huggingface_hub exposes no callback
    # that survives its own retries, so the size on disk is the only honest
    # signal. A daemon thread dies with the process.
    import threading
    stop = threading.Event()

    def watch():
        while not stop.wait(2.0):
            try:
                _exec("UPDATE model_downloads SET bytes_done=%s WHERE id=%s",
                      (measure(root, entry), row["id"]))
            except Exception:
                pass  # a progress update is never worth failing a download

    token = os.environ.get("HF_TOKEN") or None
    log(f"{row['entry']}: fetching {entry.get('repo', '')}")
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        # Per entry, not per queue: a long queue must not starve a `make
        # provision` run, which holds the lock across a whole tier.
        with provision_lock(root, holder=f"provision-daemon {row['entry']}",
                            timeout_s=LOCK_WAIT_S):
            failed = fetch_all([entry], root, token,
                               say=lambda m: log(f"{row['entry']}: {m}"))
    except CacheBusy as e:
        stop.set()
        log(f"{row['entry']}: cache busy, re-queued ({e})")
        _exec("UPDATE model_downloads SET status='queued', started_at=NULL "
              "WHERE id=%s", (row["id"],))
        return
    except Exception as e:  # noqa: BLE001
        stop.set()
        traceback.print_exc()
        _exec("UPDATE model_downloads SET status='failed', error=%s, "
              "finished_at=now() WHERE id=%s",
              (f"{type(e).__name__}: {e}"[:2000], row["id"]))
        return
    finally:
        stop.set()

    done = measure(root, entry)
    if failed:
        msg = f"{type(failed[0][1]).__name__}: {failed[0][1]}"[:2000]
        if entry.get("gated"):
            msg += " (gated: accept the model terms on huggingface.co with " \
                   "the same account as HF_TOKEN)"
        _exec("UPDATE model_downloads SET status='failed', error=%s, "
              "bytes_done=%s, finished_at=now() WHERE id=%s",
              (msg, done, row["id"]))
        log(f"{row['entry']}: FAILED {msg}")
        return
    _exec("UPDATE model_downloads SET status='done', bytes_done=%s, "
          "finished_at=now() WHERE id=%s", (done, row["id"]))
    log(f"{row['entry']}: done ({done / 1024 ** 3:.2f} GiB on disk)")


def main() -> int:
    if not DB_URL:
        log("DB_URL is not set; nothing to consume. Idling.")
        while True:
            time.sleep(3600)

    log(f"manifest {MANIFEST}, poll {POLL_S}s")
    # Recover rows left 'running' by a container that died mid-download. Their
    # partial blobs are .incomplete, which the presence check reads as absent,
    # so re-queueing is correct and resumes rather than restarts.
    for attempt in range(60):
        try:
            _exec("UPDATE model_downloads SET status='queued', started_at=NULL "
                  "WHERE status='running'")
            break
        except Exception as e:
            if attempt == 0:
                log(f"waiting for the database ({e})")
            time.sleep(2)

    while True:
        try:
            doc = load(MANIFEST)
            root = cache_root(doc)
            row = claim_next()
            if row is None:
                time.sleep(POLL_S)
                continue
            log(f"claimed {row['id'][:8]} {row['entry']}")
            run_one(row, doc, root)
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # noqa: BLE001 - never let the loop die
            log(f"loop error: {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(min(30.0, POLL_S * 5))


if __name__ == "__main__":
    sys.exit(main())
