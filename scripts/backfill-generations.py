#!/usr/bin/env python3
"""Give the orphaned `raw_*.png` files a ledger row, so the gallery can see them.

WHAT THIS RECOVERS

Until migration 017, `generate_raw_task` wrote `raw_<uuid>.png` to disk and
returned the path, and nothing recorded it anywhere. Measured 2026-09-10:

    images/raw_*.png on disk            731
    sprite_images rows matching 'raw_'    0

Every image something2 ever pulled through the A1111 facade is in that first
number. They are real generated art - the largest single population in
`images/` - and they were invisible in the UI because no table knew they
existed. This walks the directory and inserts one `generations` row per file so
`assets_v` surfaces them.

WHAT IT CANNOT RECOVER, AND WILL NOT INVENT

The prompt, the model, the seed, the caller and the parameters are gone. They
were never written down; they are not recoverable from `raw_a1b2c3d4e5f6.png`.
So a backfilled row carries an empty prompt, a NULL model and
`principal_name = 'unknown (backfilled)'`, and `params.backfill` is true. The
only real fact available is the file's mtime, which becomes `created_at`.

That is deliberate. A plausible-looking reconstruction here would be worse than
a blank: these rows sit in the same list as rows that DO know their prompt, and
nothing downstream would be able to tell the guessed ones apart.

Idempotent - a file that already has a row is skipped, so it is safe to re-run
after more generations have accumulated.

Run it inside the container, which is where DB_URL and /app/images are:

    docker exec sprite_generator python /app/scripts/backfill-generations.py --dry-run
    docker exec sprite_generator python /app/scripts/backfill-generations.py --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import psycopg2

IMAGES_DIR = os.environ.get("IMAGES_DIR", "/app/images")

# Only the untagged facade output. `core_`, `sheet_`, `tile_` and `ref_` files
# already have rows in sprite_images, jobs or reference_assets, and inserting a
# second row for one of them would put it in the gallery twice.
PATTERN = "raw_*.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the rows; without it, only report")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op, the default")
    ap.add_argument("--images-dir", default=IMAGES_DIR)
    args = ap.parse_args()

    db_url = os.environ.get("DB_URL")
    if not db_url:
        print("DB_URL is not set - run this inside the container", file=sys.stderr)
        return 2

    files = sorted(glob.glob(os.path.join(args.images_dir, PATTERN)))
    if not files:
        print(f"no {PATTERN} under {args.images_dir}")
        return 0

    conn = psycopg2.connect(db_url)
    with conn, conn.cursor() as cur:
        cur.execute("SELECT file_path FROM generations "
                    "WHERE file_path IS NOT NULL")
        known = {r[0] for r in cur.fetchall()}

    missing = [f for f in files if f not in known]
    print(f"{len(files)} {PATTERN} on disk, {len(known)} already in the ledger, "
          f"{len(missing)} to backfill")

    if not args.apply:
        for f in missing[:10]:
            print("  would add", os.path.basename(f))
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print("\nre-run with --apply to write them")
        return 0

    added = 0
    with conn, conn.cursor() as cur:
        for path in missing:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path),
                                               tz=timezone.utc)
            except OSError as e:
                print(f"  skipped {path}: {e}", file=sys.stderr)
                continue
            cur.execute(
                "INSERT INTO generations "
                "(id, kind, status, served_from, route, prompt, file_path, "
                " params, principal_name, created_at, finished_at) "
                "VALUES (%s, 'raw', 'done', 'generated', '/sdapi/v1/txt2img', "
                "        '', %s, %s, 'unknown (backfilled)', %s, %s)",
                (str(uuid.uuid4()), path,
                 json.dumps({"backfill": True,
                             "note": "prompt, model and seed were never "
                                     "recorded for this file"}),
                 mtime, mtime))
            added += 1

    conn.close()
    print(f"backfilled {added} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
