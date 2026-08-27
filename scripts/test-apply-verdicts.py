#!/usr/bin/env python3
"""Exercise `audit-character-refs.py --apply` against a THROWAWAY database.

    python scripts/test-apply-verdicts.py

WHY THIS EXISTS

`--apply` is the only part of the audit that WRITES, and it had never executed
once. It was written, documented, put in the Makefile, and reported in an ADR
as "not done because Docker is down" - which quietly turned an untested branch
into an environmental excuse. The database was in fact reachable the whole
time; nobody had looked.

An untested write path is the worst thing to leave in a curation tool. Its
failure mode is not a crash: it is `trainable=false` on rows that should have
kept it, discovered weeks later as a smaller dataset than anyone expected.

WHAT IT DOES, AND WHAT IT REFUSES TO DO

Production is READ-ONLY here. Its `reference_assets` schema is copied out with
`pg_dump --schema-only`, loaded into a scratch database, seeded with rows
pointing at real image files, and `--apply` is run against THAT. The scratch
database is dropped afterwards whether the test passes or fails, and the script
refuses to run at all if the scratch name resolves to the production database.

The schema is dumped rather than rebuilt from migrations 013/014 on purpose:
replaying 013 needs `jobs`, `sprite_images` and 012's trigger function before
it will even parse, and stubbing those means testing against a hand-written
approximation of the table rather than the one `--apply` will actually meet.

THE ASSUMPTION THIS TEST IS BUILT TO FALSIFY

Seeded paths are PRODUCTION-format (`/app/images/<name>`) while `--data` points
at a local directory. They are made to disagree deliberately. The first version
seeded whatever path its own invocation produced, so seed and query agreed and
every assertion passed - a test that shares the assumption it is checking
cannot check it. A sibling session shipped exactly that and its `--apply`
updated zero rows while exiting 0 for weeks.

CHECKED BY MUTATION, not asserted: replacing the exact-suffix match with a
whole-path comparison reddens "blocking row flipped", "reason recorded" and
"exactly one row written", and the run prints `applied: 0` - the sibling's
symptom, reproduced on demand.

CHECKED BY DECOY, which is the stronger of the two: one seeded row is shaped so
that only a wildcard could match it (`refXspriteY<hex>.png` against a filename
whose underscores LIKE treats as any-single-character). Restoring the LIKE
predicate makes the run print `applied: 2` and exit 0, and leaves the decoy row
holding the reject reason belonging to a DIFFERENT file - a wrong-row write no
rowcount would have exposed. Mutation proves the test can fail; the decoy
proves the FIXTURE is not written from the same belief that produced the bug.
Both this session and a sibling shipped a path fix verified against a fixture
that shared its assumption, so the fix inherited the flaw one level up and
every assertion stayed green.
"""

import os
import subprocess
import sys
import uuid

import psycopg2

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_here)
ENV = os.path.join(REPO, "compose", "develop", ".env")
SCRATCH_NAME = "audit_smoke_scratch"

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


USER, PW, PROD_DB = env_value("DB_USER"), env_value("DB_PASSWORD"), env_value("DB_NAME")
HOST = os.environ.get("DB_HOST", "127.0.0.1")
BASE = f"postgresql://{USER}:{PW}@{HOST}:5432/"
ADMIN, SCRATCH, PROD = BASE + "postgres", BASE + SCRATCH_NAME, BASE + PROD_DB

# The one guard that matters. Everything below drops a database by name.
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


print(f"building throwaway {SCRATCH_NAME}; {PROD_DB} is read-only here")
admin(f"DROP DATABASE IF EXISTS {SCRATCH_NAME}")
admin(f"CREATE DATABASE {SCRATCH_NAME}")
conn = None
try:
    dump = subprocess.run(
        ["pg_dump", "--schema-only", "--no-owner", "--no-privileges",
         "-t", "public.reference_assets", PROD],
        capture_output=True, text=True)
    if dump.returncode != 0:
        print(dump.stderr[-1500:])
        raise SystemExit("pg_dump failed - is postgresql-client installed?")

    # Keep only the DDL. A newer pg_dump against an older server emits things
    # the server cannot take:
    #   \restrict / \unrestrict  - psql meta-commands; psycopg2 sees a stray "\"
    #   SET transaction_timeout  - a parameter that does not exist before 17
    # Both are session settings for psql's benefit and say nothing about the
    # table, so dropping them changes nothing this test is about.
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

    # If this fails the rest proves nothing: it would be testing a table that
    # is not the one --apply will meet.
    check("scratch schema matches production", columns(SCRATCH), columns(PROD))

    IMG = os.path.join(REPO, "images")
    reject = "ref_sprite_004228080a22.png"   # 135 subjects, 5.6:1 strip
    keep = "ref_core_01ac2de061db.png"       # single subject, flat backdrop
    gone = "ref_sprite_5e0e1537bee8.png"     # checkerboard, but soft-deleted
    for f in (reject, keep, gone):
        if not os.path.isfile(os.path.join(IMG, f)):
            sys.exit(f"fixture missing: {f} - this test needs the real images")

    # Seed PRODUCTION-format paths, not the ones this invocation happens to use.
    #
    # This is the whole point, and the first version of this test got it wrong.
    # It seeded `os.path.join(IMG, fname)` - the same local path it then passed
    # as `--data`. Seed and query agreed with each other, every assertion went
    # green, and the test could not have detected a path-matching bug because it
    # shared the assumption it was supposed to be checking.
    #
    # Production stores `/app/images/<name>` (the container's mount); the audit
    # runs against a local `--data ./images`. Making them DISAGREE here is what
    # forces the matching logic to be a real suffix match rather than a
    # coincidence. A sibling session's equivalent test passed for months over a
    # `--apply` that updated zero rows and exited 0.
    PROD_PREFIX = "/app/images/"
    ids = {}
    for fname, deleted in ((reject, False), (keep, False), (gone, True)):
        rid = str(uuid.uuid4())
        ids[fname] = rid
        cur.execute(
            "INSERT INTO reference_assets (id, kind, file_path, trainable, "
            "trainable_why, deleted, metrics) VALUES (%s,%s,%s,true,%s,%s,%s)",
            (rid, "sprite" if "sprite" in fname else "core",
             PROD_PREFIX + fname, "seeded: fine to train on", deleted,
             '{"coverage": 0.42}'))
    # A DECOY: a row the previous, believed-correct predicate would have hit.
    #
    # `file_path LIKE '%/' || name` reads as a suffix match and is not one.
    # Every filename here is `ref_sprite_<hex>.png` - three underscores, and in
    # LIKE each is a single-character wildcard - so the pattern also matches
    # `refXspriteY<hex>.png`. A wrong-row update would be silent, because
    # `--apply` reports the same rowcount either way.
    #
    # This row exists to be reachable by the WRONG predicate and unreachable by
    # the right one. It is a different kind of evidence from the mutation check
    # below: mutation proves the test CAN fail, but says nothing about whether
    # the fixture encodes the same misunderstanding as the code. That is the
    # failure mode that caught both this session and a sibling one - a fix
    # verified against a fixture written from the belief that produced the bug.
    #
    # Deliberately no such file on disk: the audit never names it, so nothing
    # but a wildcard can reach it.
    decoy = "refXspriteY004228080a22.png"
    decoy_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO reference_assets (id, kind, file_path, trainable, "
        "trainable_why, deleted, metrics) VALUES (%s,%s,%s,true,%s,false,%s)",
        (decoy_id, "sprite", PROD_PREFIX + decoy, "seeded: fine to train on",
         '{"coverage": 0.42}'))
    print(f"  seeded 3 rows as {PROD_PREFIX}<name>, while --data is {IMG}")
    print(f"  plus a wildcard decoy {decoy} that only LIKE could match")

    run = subprocess.run(
        [sys.executable, os.path.join(_here, "audit-character-refs.py"),
         "--data", IMG, "--apply"],
        capture_output=True, text=True, env=dict(os.environ, DB_URL=SCRATCH))
    applied_lines = [l for l in run.stdout.splitlines()
                     if l.startswith(("applied:", "NOT applied:"))]
    print("  --apply exit", run.returncode, "|", applied_lines)
    if run.returncode != 0:
        print(run.stdout[-1500:], run.stderr[-1500:])
        fails.append("--apply crashed")

    # A silent no-op reported as success is the failure being guarded against,
    # so assert the run SAYS how many findings matched nothing. Most will:
    # the audit rejects hundreds of files and only three have rows here.
    check("run reports what did not land",
          any(l.startswith("NOT applied:") for l in applied_lines), True)

    def row(f):
        cur.execute("select trainable, trainable_why, metrics->>'coverage' "
                    "from reference_assets where id=%s", (ids[f],))
        return cur.fetchone()

    t, why, cov = row(reject)
    check("blocking row flipped to trainable=false", t, False)
    check("reason recorded on it", bool(why and "contact sheet" in why), True)
    # Currently trivial - `--apply` writes only trainable/trainable_why and
    # never touches metrics. Kept as a REGRESSION guard: the day someone adds
    # a metrics write, `metrics = %s` instead of `metrics || %s` would silently
    # destroy everything measure.py had recorded, and the row would still look
    # perfectly well-formed afterwards.
    check("pre-existing metrics survived the write", cov, "0.42")

    t, why, _ = row(keep)
    check("clean row NOT flipped", t, True)
    check("clean row keeps its prior reason", why, "seeded: fine to train on")

    t, why, _ = row(gone)
    check("soft-deleted row untouched",
          (t, why), (True, "seeded: fine to train on"))

    # The decoy. Under the old LIKE predicate this row matched and flipped;
    # under `right(file_path, n)` it cannot be reached at all.
    cur.execute("select trainable, trainable_why from reference_assets "
                "where id=%s", (decoy_id,))
    check("wildcard decoy untouched", cur.fetchone(),
          (True, "seeded: fine to train on"))

    # The match is a byte-exact suffix comparison over every row in the table.
    # Prove it does not splash onto rows it was not aimed at - with the decoy
    # seeded, this count is 2 under the wildcard predicate and 1 under the
    # right one, so it is now a real discriminator rather than a tautology.
    cur.execute("select count(*) from reference_assets where trainable = false")
    check("exactly one row written", cur.fetchone()[0], 1)
finally:
    if conn is not None:
        conn.close()
    admin(f"DROP DATABASE IF EXISTS {SCRATCH_NAME}")
    print(f"dropped {SCRATCH_NAME}")

print("\nFAILURES:", ", ".join(fails) if fails else "none")
sys.exit(1 if fails else 0)
