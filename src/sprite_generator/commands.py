"""The operator commands the UI is allowed to run, and nothing else.

WHY A TABLE RATHER THAN A COMMAND STRING

Every one of these already exists as a Makefile target, so the obvious design
is an endpoint that takes a target name and shells out. That is a remote shell
in a browser tab, and this API has an OPEN MODE with no authentication at all
(`auth.is_enforced()` is false until a token is set). A single missing
`shlex.quote` or a `../` would be the whole machine.

So the browser sends a KEY. The argv lives here and is never assembled from
anything the client sent. An unknown key is a 404, not a shell error.

WHY IT IS STDLIB-ONLY

Both sides import it: the API process to list and validate, the Celery worker
to execute. `jobs.py` keeps `actions.py` importable for the same reason - the
API must not pull in the Celery/torch chain just to read a name. Do not import
psycopg2, fastapi or celery into this module.

WHAT `writes` MEANS, AND WHY IT IS NOT A BOOLEAN

    "nothing"   reads images, writes a report. Safe to run at any time.
    "files"     writes new images under /app/images/recovered. Never
                overwrites an original - the scripts refuse without --out.
    "database"  writes `trainable` and `trainable_why` on live reference rows.

The third needs an explicit confirmation from the caller and a rowcount shown
first. It is the only one here that changes what the app does next, and its
effect is invisible in the UI until somebody notices the dataset shrank.

A NOTE ON THE QUEUE, WHICH IS NOT OBVIOUS FROM THE UI

The worker runs `--concurrency=1` because the box has one GPU and a training
run and a sheet build cannot overlap. These commands are CPU-only, but they sit
in the SAME queue - so a 5-minute audit delays the next generation by 5
minutes. That is a real cost and the UI says so rather than letting someone
discover it. Splitting a CPU queue out is the right fix and is deliberately not
done here: it needs a second worker container, which is a deployment change.
"""

# Where the scripts live inside the worker. Read-only mount, deliberately
# separate from /app - see the note on the volume in docker-compose.yml.
SCRIPTS = "/app/scripts"
IMAGES = "/app/images"

COMMANDS: dict[str, dict] = {
    "audit-refs": {
        "label": "Audit references",
        "group": "Curation",
        "order": 1,
        "writes": "nothing",
        "summary": "Give every sprite and core reference a verdict and a "
                   "reason, and write the report.",
        "detail": "Reads the images, decides keep / review / reject for each, "
                  "and records WHY. Writes nothing to the database - run "
                  "'Apply verdicts' for that. Takes about five minutes.",
        "argv": ["python", f"{SCRIPTS}/audit-character-refs.py",
                 "--out", f"{IMAGES}/reference-audit.md",
                 "--json", f"{IMAGES}/reference-audit.json"],
        "minutes": 5,
    },
    "audit-cells": {
        "label": "Audit recovered cells",
        "group": "Curation",
        "order": 2,
        "writes": "nothing",
        "summary": "Put the recovered cells through the same gate as the "
                   "references.",
        "detail": "The cells in images/recovered/cells are not registered "
                  "references, so the normal audit never sees them. Skipping "
                  "this is how they were described as training-grade for half "
                  "a day on the strength of thumbnails.",
        "argv": ["python", f"{SCRIPTS}/audit-character-refs.py",
                 "--kind", "sprite",
                 "--data", f"{IMAGES}/recovered/cells",
                 "--pattern", "cell_sprite_*.png",
                 "--out", f"{IMAGES}/recovered/cells-audit.md"],
        "minutes": 2,
    },
    "audit-refs-apply": {
        "label": "Apply verdicts to the database",
        "group": "Curation",
        "order": 3,
        "writes": "database",
        "summary": "Set trainable = false and record the reason on every "
                   "rejected reference.",
        "detail": "Changes what the Reference tabs show and what the next "
                  "training run consumes. Only rejections are written - a "
                  "'review' verdict is a human's call and is never applied. "
                  "Soft-deleted rows are untouched.",
        "argv": ["python", f"{SCRIPTS}/audit-character-refs.py", "--apply"],
        "minutes": 5,
    },
    "key-checkerboard": {
        "label": "Key out painted checkerboards",
        "group": "Recovery",
        "order": 1,
        "writes": "files",
        "summary": "Turn a painted-on transparency checker back into real "
                   "alpha.",
        "detail": "Twelve sprite sheets have the checker pattern drawn into "
                  "their pixels, which trains the model to draw the checker. "
                  "Originals are never modified; output goes to "
                  "images/recovered/keyed.",
        "argv": ["python", f"{SCRIPTS}/key-checkerboard.py",
                 "--out", f"{IMAGES}/recovered/keyed", "--write"],
        "minutes": 2,
    },
    "recover-cells": {
        "label": "Split keyed sheets into cells",
        "group": "Recovery",
        "order": 2,
        "writes": "files",
        "summary": "Cut the keyed sheets into single-subject cells.",
        "detail": "Run 'Key out painted checkerboards' first - this reads its "
                  "output. Drops cells under 160px and anything longer than "
                  "2:1, which is what keeps the 63px mush out.",
        "argv": ["python", f"{SCRIPTS}/split-sheets.py",
                 "--images", f"{IMAGES}/recovered/keyed",
                 "--kind", "sprite",
                 "--out", f"{IMAGES}/recovered/cells", "--write",
                 "--min-side", "160", "--max-aspect", "2.0",
                 "--drop-edge-slivers"],
        "minutes": 3,
    },
    "test-train-prep": {
        "label": "Trainer preparation",
        "group": "Tests",
        "order": 1,
        "writes": "nothing",
        "summary": "The trainer's GPU-free half: captions, fit, resampling, "
                   "grid detection.",
        "detail": "No CUDA and no weights. Runs in seconds.",
        "argv": ["python", f"{SCRIPTS}/test-train-prep.py"],
        "minutes": 1,
    },
    "test-split-sheets": {
        "label": "Sheet splitting",
        "group": "Tests",
        "order": 2,
        "writes": "nothing",
        "summary": "Cell cleanup, including the sliver removal that edits "
                   "training pixels.",
        "detail": "Checked by mutation - see the module note.",
        "argv": ["python", f"{SCRIPTS}/test-split-sheets.py"],
        "minutes": 1,
    },
    "test-apply-verdicts": {
        "label": "Verdict writing",
        "group": "Tests",
        "order": 3,
        "writes": "nothing",
        "summary": "Exercises the database write against a THROWAWAY copy.",
        "detail": "Builds a scratch database from production's schema, runs "
                  "the write there, and drops it. Production is read-only "
                  "throughout, and the script refuses to run if the scratch "
                  "name resolves to the live database.",
        "argv": ["python", f"{SCRIPTS}/test-apply-verdicts.py"],
        # postgresql-client is now installed in the image, so this passes. The
        # declaration stays: it is what turns "the image is missing a package"
        # into a disabled button with a reason, instead of a queued job that
        # dies on `FileNotFoundError: pg_dump` and reads as a broken test. If
        # the package is ever dropped from the Dockerfile, this reappears as a
        # clear message rather than as a mystery.
        "requires": ["pg_dump"],
        "minutes": 1,
    },
    "test-auth-scopes": {
        "label": "Endpoint scopes",
        "group": "Tests",
        "order": 4,
        "writes": "nothing",
        "summary": "Every endpoint asks for a scope this server can actually "
                   "grant.",
        "detail": "auth.require passes an admin key unconditionally, so a "
                  "scope outside ALL_SCOPES becomes 'admin only' silently and "
                  "tells everyone else they lack a scope that cannot be "
                  "granted. Reads the source, so it covers endpoints nobody "
                  "has called - including this one, which had exactly that "
                  "bug.",
        "argv": ["python", f"{SCRIPTS}/test-auth-scopes.py"],
        "minutes": 1,
    },
    "test-spec-fields": {
        "label": "Request fields",
        "group": "Tests",
        "order": 5,
        "writes": "nothing",
        "summary": "A caller's flag reaches the worker instead of being "
                   "dropped in silence.",
        "detail": "Pydantic ignores undeclared fields and model_dump() emits "
                  "only declared ones, so a field a task reads with a default "
                  "but the model never declares is replaced by that default "
                  "with nothing able to tell. JobSpec did not declare "
                  "concept_check while the refusal message told callers to "
                  "send it.",
        "argv": ["python", f"{SCRIPTS}/test-spec-fields.py"],
        "minutes": 1,
    },
}

# Ordered for display. Curation first because it is the reason this exists;
# Tests last because they are for when something looks wrong.
GROUPS = ["Curation", "Recovery", "Tests"]


def missing_requirements(name: str) -> list[str]:
    """Executables a command needs that are not on PATH.

    The API and the worker are built from the SAME Dockerfile, so asking here
    answers for there. If that ever stops being true this becomes a guess, and
    the honest replacement is a probe task on the worker itself.
    """
    import shutil
    return [b for b in COMMANDS[name].get("requires", ()) if not shutil.which(b)]


def public(name: str) -> dict:
    """One command as the browser should see it - everything except `argv`.

    The argv is withheld rather than merely unused. A UI that renders it
    invites someone to make it editable, and this table's whole purpose is
    that the client never supplies a command line.
    """
    c = COMMANDS[name]
    missing = missing_requirements(name)
    return ({k: v for k, v in c.items() if k != "argv"}
            | {"name": name,
               "available": not missing,
               "unavailable_why": (
                   f"needs {', '.join(missing)}, which is not installed in this "
                   f"image" if missing else None)})


def listing() -> list[dict]:
    """Grouped, then in the order somebody would actually run them.

    Sorting by LABEL inside a group put "Apply verdicts to the database" first
    in Curation - the one destructive command at the top of the page, above the
    audit you have to run before it means anything. Alphabetical order is not a
    neutral default when one entry writes to live rows.
    """
    group_rank = {g: i for i, g in enumerate(GROUPS)}
    return sorted((public(n) for n in COMMANDS),
                  key=lambda c: (group_rank.get(c["group"], 99),
                                 c.get("order", 99), c["label"]))
