"""One-off: move audio written before AUDIO_DIR existed out of IMAGES_DIR.

Until 2026-09-28 every artefact landed flat in IMAGES_DIR as
`audio_<kind>_<name>_<uid>.{ogg,wav}`, because `generations._url` served
files by basename from /images. Audio now has its own tree,
`<AUDIO_DIR>/<kind>/<name>_<uid>.{ogg,wav}`. This moves the files and
rewrites the ledger's `file_path` and `params.master_path` so cached rows keep
serving. Also moves `audio-takes/` -> `_takes/` and `audio-spike/` -> `spike/`.

Idempotent. Dry run unless --apply.

    docker exec sprite_generator python /app/scripts/move-audio-out-of-images.py
    docker exec sprite_generator python /app/scripts/move-audio-out-of-images.py --apply
"""

import argparse
import glob
import json
import os
import shutil

import psycopg2

IMAGES_DIR = os.environ.get("IMAGES_DIR", "/app/images")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/app/audio")
KINDS = ("music", "ambience", "sfx")
DIRS = {"audio-takes": "_takes", "audio-spike": "spike"}


def new_path(old: str) -> str | None:
    base = os.path.basename(old)
    for kind in KINDS:
        prefix = f"audio_{kind}_"
        if base.startswith(prefix):
            return os.path.join(AUDIO_DIR, kind, base[len(prefix):])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    verb = "move" if args.apply else "would move"

    moves = {}
    for old in sorted(glob.glob(os.path.join(IMAGES_DIR, "audio_*"))):
        dst = new_path(old)
        if dst:
            moves[old] = dst
    for src_name, dst_name in DIRS.items():
        src = os.path.join(IMAGES_DIR, src_name)
        if os.path.isdir(src):
            moves[src] = os.path.join(AUDIO_DIR, dst_name)

    for old, dst in moves.items():
        print(f"{verb}: {old} -> {dst}")
        if not args.apply:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(old) and os.path.isdir(dst):
            for f in os.listdir(old):
                shutil.move(os.path.join(old, f), os.path.join(dst, f))
            os.rmdir(old)
        else:
            shutil.move(old, dst)

    # Rows are rewritten from the path mapping, not from what was moved this
    # run, so a second run fixes rows a first run's crash left behind.
    conn = psycopg2.connect(os.environ["DB_URL"])
    updated = 0
    with conn, conn.cursor() as cur:
        cur.execute("SELECT id, file_path, params FROM generations "
                    "WHERE kind = ANY(%s) AND file_path LIKE %s",
                    (list(KINDS), IMAGES_DIR.rstrip("/") + "/audio_%"))
        for gen_id, fp, params in cur.fetchall():
            dst = new_path(fp)
            if not dst:
                continue
            params = params or {}
            if params.get("master_path"):
                params["master_path"] = new_path(params["master_path"]) \
                    or params["master_path"]
            print(f"{'update' if args.apply else 'would update'} row "
                  f"{gen_id}: {fp} -> {dst}")
            if args.apply:
                cur.execute("UPDATE generations SET file_path = %s, "
                            "params = %s::jsonb WHERE id = %s",
                            (dst, json.dumps(params), gen_id))
            updated += 1
    conn.close()
    print(f"{len(moves)} paths, {updated} ledger rows"
          + ("" if args.apply else " (dry run; pass --apply)"))


if __name__ == "__main__":
    main()
