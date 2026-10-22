#!/usr/bin/env python3
"""Reclaim the build scratch from FINISHED sheet jobs. Dry run by default.

    prune-job-cache.py                 what it would delete, deletes nothing
    prune-job-cache.py --apply         delete it
    prune-job-cache.py --older-than 7  only jobs finished over 7 days ago

WHAT IT DELETES, AND WHY ONLY THESE TWO THINGS

A sheet's work directory is 490 MB for a 7-action build, and **71% of that is
`embeds_*.pt`** - the text-encoder output for every cell. Nothing reads those
once the build has finished:

  * `embeds_turn.pt` is read by `turnaround-denoise`, during the build.
  * `embeds_act.pt` is read by `actions-denoise` and by the automatic redraw
    rounds, all of which run inside the build.
  * an explicit redraw (`POST /api/jobs/{id}/redraw`) encodes its OWN
    `embeds_redraw.pt`, because its conditioning differs - it does not reuse
    the build's embeddings.

`att*_*.png` are the per-attempt raw renders, kept during a build so a redraw
that came back worse cannot overwrite a better cell. Once the ledger has
picked a winner they are history.

WHAT IT MUST NEVER TOUCH, because a redraw re-reads all of it:

    config.json   dir_<d>.png   cut_*.png   scores.json   frames/

`cell_*.png` (the raw 512px renders) are also kept: `score-sheet.py`
re-scores from them, and compose falls back to keying them when a cut-out is
missing. They are 59 MB of the 490 MB, and losing the ability to re-score a
finished sheet is not worth that.

WHY IT IS OPT-IN AND NEVER AUTOMATIC

A redraw is only possible while the parent's work directory survives, so
anything that deletes from it is deciding how long a sheet stays repairable.
This command deliberately has no scheduler, no hook and no default run: the
`make up` lane must not quietly acquire a retention policy. See
`.ai/specs/sheet-quality/contract.md` 5d.

SAFETY. A job is only a candidate when its row says `done`, `failed` or
`cancelled`. A queued or running job is skipped whatever its files look
like, because its embeddings are exactly what the next stage is about to
read. Directories with no job row at all are reported but skipped unless
`--orphans` is given - nothing can redraw them through the API, but an
unexplained directory is a reason to look rather than to delete.
"""

import argparse
import glob
import json
import os
import sys

IMAGES = os.environ.get("IMAGES_DIR", "/app/images")
TERMINAL = ("done", "failed", "cancelled")

# Only these. Globs are anchored to the work directory, never recursive.
PRUNABLE = ("embeds_*.pt", "att*_*.png")

# Named so a reader can see at a glance what is deliberately spared.
KEEP = ("config.json", "dir_*.png", "cut_*.png", "cell_*.png", "scores.json",
        "frames/")


def job_statuses() -> dict:
    """{job_id: (status, finished_at)} for every job row."""
    import psycopg2
    import psycopg2.extras

    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set; run this inside a container")
    with psycopg2.connect(url) as conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, status, finished_at FROM jobs")
        return {str(r["id"]): (r["status"], r["finished_at"])
                for r in cur.fetchall()}


def candidates(status_by_id, older_than_days, orphans):
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    out, skipped = [], []
    for work in sorted(glob.glob(os.path.join(IMAGES, "_job_*"))):
        if not os.path.isdir(work):
            continue
        job_id = os.path.basename(work)[len("_job_"):]
        status, finished = status_by_id.get(job_id, (None, None))

        if status is None:
            (out if orphans else skipped).append(
                (work, job_id, "orphan (no job row)", reclaimable(work)))
            continue
        if status not in TERMINAL:
            skipped.append((work, job_id, f"status {status}",
                            reclaimable(work)))
            continue
        if older_than_days and finished:
            age = (now - finished).days
            if age < older_than_days:
                skipped.append((work, job_id, f"finished {age}d ago",
                                reclaimable(work)))
                continue
        out.append((work, job_id, status, reclaimable(work)))
    return out, skipped


def reclaimable(work):
    """(files, bytes) matching PRUNABLE in this directory."""
    files = []
    for pat in PRUNABLE:
        files.extend(glob.glob(os.path.join(work, pat)))
    return files, sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="actually delete; without it nothing is removed")
    p.add_argument("--older-than", type=int, default=0, metavar="DAYS",
                   help="only jobs that finished more than DAYS ago")
    p.add_argument("--orphans", action="store_true",
                   help="also prune directories with no job row")
    p.add_argument("--only", metavar="JOB_ID",
                   help="restrict to one job; an id prefix is enough")
    p.add_argument("--json", help="write the per-directory plan here")
    a = p.parse_args()

    status_by_id = job_statuses()
    todo, skipped = candidates(status_by_id, a.older_than, a.orphans)
    if a.only:
        todo = [t for t in todo if t[1].startswith(a.only)]
        skipped = [t for t in skipped if t[1].startswith(a.only)]
        if not todo:
            print(f"no prunable job matches {a.only!r}"
                  + (f"; it is {skipped[0][2]}" if skipped else ""))
            return 1

    total = sum(b for _, _, _, (_, b) in todo)
    print(f"keeping, always: {', '.join(KEEP)}")
    print(f"pruning:         {', '.join(PRUNABLE)}\n")

    for work, job_id, status, (files, b) in todo:
        if not files:
            continue
        print(f"  {job_id[:8]}  {status:22s} {human(b):>8s}  "
              f"{len(files)} file(s)")
    if skipped:
        print(f"\n  skipped {len(skipped)}:")
        for work, job_id, why, (_, b) in skipped[:8]:
            print(f"    {job_id[:8]}  {why:26s} holds {human(b)}")
        if len(skipped) > 8:
            print(f"    ... and {len(skipped) - 8} more")

    print(f"\nreclaimable: {human(total)} across "
          f"{sum(1 for _, _, _, (f, _) in todo if f)} job(s)")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump([{"job": j, "status": s, "bytes": b,
                        "files": [os.path.basename(x) for x in f]}
                       for _, j, s, (f, b) in todo], fh, indent=2)

    if not a.apply:
        print("\nDRY RUN - nothing deleted. Re-run with --apply "
              "(make prune-job-cache apply=1).")
        return 0

    freed = 0
    for work, job_id, _, (files, _) in todo:
        for f in files:
            try:
                freed += os.path.getsize(f)
                os.remove(f)
            except OSError as e:
                print(f"  could not remove {f}: {e}")
    print(f"\nfreed {human(freed)}. Redraw still works on every job above - "
          f"nothing it reads was touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
