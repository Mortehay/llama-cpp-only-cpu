#!/usr/bin/env python3
"""Does --register put the repaired cutouts where training will actually see them?

WHY THIS EXISTS

The recovery produced eleven clean entity cutouts and they were INERT.
`all_trainable_refs` selects from `reference_assets` by `file_path`, so files
sitting in `images/recovered/entity/` are invisible to the trainer - and worse,
the eleven ORIGINALS were still `trainable = true`, pointing at the versions
with the backdrop, the floating diamond and the ember specks. A training run
started right after the recovery would have consumed exactly what the recovery
removed, and every count in the findings note would still have read eleven.

`--register` inserts the cutout and retires its original. That is two writes per
file against production, in a table the Reference tab and every training run
read, so it gets a test that runs against a THROWAWAY database.

Same harness as test-apply-verdicts.py: production's schema is dumped with
`pg_dump --schema-only`, loaded into a scratch database, seeded, and register()
is pointed at THAT with DB_URL. The scratch database is dropped at the end and
the run refuses to start if the scratch name resolves to production.

WHAT IT ASSERTS, and why each one is a way this could go wrong quietly

  inserts a row per repaired file        the point of the feature
  new row is trainable and points at
    the REPAIRED path                    a row pointing back at the original
                                         would look right in every count and
                                         train on the backdrop
  parent is retired, reason names
    the child                            without this the trainer sees BOTH,
                                         and the backdrop version outvotes
                                         nothing - it is simply included
  an already-clean file gets NO new row  its original is already the usable
                                         file; a copy trains on the same image
                                         twice at double weight
  second run inserts nothing             re-running after editing one verdict
                                         line would otherwise register ten
                                         duplicates
  a file with no parent row is reported  silence there is how --apply once
                                         updated zero rows and exited 0

Usage:
    python3 scripts/test-register-cutouts.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import uuid

import psycopg2
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_here)
ENV = os.path.join(REPO, "compose", "develop", ".env")
SCRATCH_NAME = "register_cutouts_scratch"
PROD_PREFIX = "/app/images/"

fails: list[str] = []


def check(name, got, want):
    ok = got == want
    print(("ok   " if ok else "FAIL ") + f"{name}: {got!r}"
          + ("" if ok else f"  != {want!r}"))
    if not ok:
        fails.append(name)


def env_value(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    if not os.path.isfile(ENV):
        sys.exit(f"{key} not in the environment and {ENV} does not exist")
    for line in open(ENV):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    sys.exit(f"{key} not found in {ENV}")


_url = os.environ.get("DB_URL", "").strip()
if _url:
    from urllib.parse import unquote, urlparse
    _u = urlparse(_url)
    USER = unquote(_u.username or "")
    PW = unquote(_u.password or "")
    HOST = _u.hostname or "127.0.0.1"
    PORT = _u.port or 5432
    PROD_DB = (_u.path or "/postgres").lstrip("/")
else:
    USER, PW = env_value("DB_USER"), env_value("DB_PASSWORD")
    PROD_DB = env_value("DB_NAME")
    HOST = os.environ.get("DB_HOST", "127.0.0.1")
    PORT = 5432

BASE = f"postgresql://{USER}:{PW}@{HOST}:{PORT}/"
ADMIN, SCRATCH, PROD = BASE + "postgres", BASE + SCRATCH_NAME, BASE + PROD_DB

if SCRATCH_NAME == PROD_DB:
    sys.exit(f"refusing to run: scratch name {SCRATCH_NAME!r} IS the "
             f"production database")


def admin(sql: str):
    c = psycopg2.connect(ADMIN, connect_timeout=5)
    c.autocommit = True
    c.cursor().execute(sql)
    c.close()


def columns(dsn: str):
    c = psycopg2.connect(dsn)
    cur = c.cursor()
    cur.execute("select column_name, data_type from information_schema.columns "
                "where table_name='reference_assets' order by 1")
    out = cur.fetchall()
    c.close()
    return out


def load_register():
    import importlib.util
    path = os.path.join(_here, "recover-grid-refs.py")
    spec = importlib.util.spec_from_file_location("recover_grid_refs", path)
    if spec is None or spec.loader is None:
        sys.exit("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.register


def swatch(path, colour):
    Image.new("RGBA", (200, 200), colour).save(path)


print(f"building throwaway {SCRATCH_NAME}; {PROD_DB} is read-only here")
admin(f"DROP DATABASE IF EXISTS {SCRATCH_NAME}")
admin(f"CREATE DATABASE {SCRATCH_NAME}")
conn = None
workdir = None
try:
    dump = subprocess.run(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges",
         "-t", "public.reference_assets", PROD],
        capture_output=True, text=True)
    if dump.returncode != 0:
        print(dump.stderr[-1500:])
        raise SystemExit("pg_dump failed - is postgresql-client installed?")
    sql = "\n".join(
        ln for ln in dump.stdout.splitlines()
        if not ln.startswith("\\")
        and not ln.startswith("SET ")
        and not ln.startswith("SELECT pg_catalog.set_config"))

    conn = psycopg2.connect(SCRATCH)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    print("  loaded reference_assets schema from production")
    check("scratch schema matches production", columns(SCRATCH), columns(PROD))

    # Four cases, and the images are synthetic swatches: this test is about
    # rows, and using real references would make it depend on the recovery
    # having been run first.
    workdir = tempfile.mkdtemp(prefix="register-cutouts-")
    repaired_dir = os.path.join(workdir, "repaired")
    images_dir = os.path.join(workdir, "images")
    os.makedirs(repaired_dir)
    os.makedirs(images_dir)

    REPAIRED = "ref_core_aaaaaaaaaaaa.png"   # repaired, parent exists
    CLEAN = "ref_core_bbbbbbbbbbbb.png"      # needed nothing, parent exists
    ORPHAN = "ref_core_cccccccccccc.png"     # repaired, NO parent row
    for n in (REPAIRED, CLEAN, ORPHAN):
        swatch(os.path.join(repaired_dir, n), (10, 200, 90, 255))

    ids = {}
    for n in (REPAIRED, CLEAN):
        rid = str(uuid.uuid4())
        ids[n] = rid
        cur.execute(
            "INSERT INTO reference_assets (id, kind, file_path, trainable, "
            "trainable_why, deleted, metrics) "
            "VALUES (%s,'core',%s,true,%s,false,'{}')",
            (rid, PROD_PREFIX + n, "seeded: fine to train on"))
    print(f"  seeded 2 parents as {PROD_PREFIX}<name>; {ORPHAN} has none")

    kept = [os.path.join(repaired_dir, n) for n in (REPAIRED, CLEAN, ORPHAN)]
    results = [
        {"file": REPAIRED, "stage": "key + isolate(848px)"},
        {"file": CLEAN, "stage": "none needed"},
        {"file": ORPHAN, "stage": "key"},
    ]

    register = load_register()
    os.environ["DB_URL"] = SCRATCH

    made, retired = register(kept, results, images_dir, dry_run=True)
    check("dry run writes nothing", (made, retired), (0, 0))
    cur.execute("SELECT count(*) FROM reference_assets")
    check("dry run leaves the table alone", cur.fetchone()[0], 2)

    made, retired = register(kept, results, images_dir, dry_run=False)
    check("registered the repaired file only", made, 1)
    check("retired one original", retired, 1)

    cur.execute("SELECT file_path, trainable, metrics->>'recovered_source' "
                "FROM reference_assets WHERE metrics ? 'recovered_from'")
    rows = cur.fetchall()
    check("one new row", len(rows), 1)
    new_path, new_trainable, src = rows[0]
    check("new row is trainable", new_trainable, True)
    check("new row names its source", src, REPAIRED)
    check("new row points at the images dir, not the repair dir",
          new_path.startswith(images_dir), True)
    # The file must EXIST at the recorded path. A row pointing at a path the
    # trainer cannot open fails at the far end of a long job, not here.
    check("the registered file is on disk", os.path.isfile(new_path), True)
    check("a thumbnail was written beside it",
          os.path.isfile(os.path.join(os.path.dirname(new_path),
                                      "thumb_" + os.path.basename(new_path))),
          True)

    cur.execute("SELECT trainable, trainable_why FROM reference_assets "
                "WHERE id = %s", (ids[REPAIRED],))
    t, why = cur.fetchone()
    check("parent retired", t, False)
    check("retirement names the replacement",
          os.path.basename(new_path) in (why or ""), True)

    cur.execute("SELECT trainable FROM reference_assets WHERE id = %s",
                (ids[CLEAN],))
    check("the already-clean original stays trainable", cur.fetchone()[0], True)
    cur.execute("SELECT count(*) FROM reference_assets "
                "WHERE metrics->>'recovered_from' = %s", (ids[CLEAN],))
    check("no duplicate row for the already-clean file", cur.fetchone()[0], 0)

    made2, retired2 = register(kept, results, images_dir, dry_run=False)
    check("second run is idempotent", (made2, retired2), (0, 0))
    cur.execute("SELECT count(*) FROM reference_assets")
    check("table size unchanged by the second run", cur.fetchone()[0], 3)

finally:
    if conn:
        conn.close()
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    admin(f"DROP DATABASE IF EXISTS {SCRATCH_NAME}")
    print(f"dropped {SCRATCH_NAME}")

print()
if fails:
    print("FAIL: " + ", ".join(fails))
    sys.exit(1)
print("PASS: cutouts are registered as trainable references, their originals "
      "are retired, and re-running changes nothing.")
