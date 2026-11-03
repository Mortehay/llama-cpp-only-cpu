#!/usr/bin/env python3
"""Cancel stops a build stage within one cell. No GPU, no database, ~5 s.

WHY THIS EXISTS (SOMET-694)

The conveyor looked for a cancel only BETWEEN stages, and the stages are
enormous - `actions-denoise` measured 1 h 43 m on a 7-action sheet. So a
cancel landed, the row went terminal, and the stage drew every remaining cell
anyway. Measured on job 3061288d: cancelled 18:12 UTC, still reporting
`cell 176/205` at 19:34, with the next job queued behind it for ~75 minutes.
A second DELETE answered "already finished; nothing to cancel", because the
first one had written `cancelled` and `finished_at` up front.

`run_stage` already reads each stage's stdout per cell to move the progress
bar, so it is the one place awake often enough to act - in every stage, with
no database access inside build-sheet.py. This test drives that real loop
against a fake stage that prints the real `denoised N/M` protocol, so the
subprocess handling, the terminate and the sentinel are all exercised; only
the pixels are missing.

MUTATION RESULT, so nobody has to trust that this test can fail

Disabling the cancel branch in `run_stage` - the behaviour this replaces:

    a cancelled stage returns CANCELLED_RC         FAIL (rc 0)
    a cancelled stage stops early                  FAIL (saw 6 of 6 cells)
    it stops at the FIRST cell it checks           FAIL (saw 6)
    and it stops quickly                           FAIL (1.82s of 1.80s)
    the caller is told where it stopped            FAIL (['cell 5/6','cell 6/6'])
    a cancelled stage leaves no process running    FAIL (1 still alive)

Six of twenty-one, and the last one is the one that was costing GPU hours.

Note that the process-table check only runs where `ps` exists; it is skipped
in the worker image. Do not rely on it alone - the cell count is the signal
that works everywhere.

Run: make test-job-cancel
"""

import os
import subprocess
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src",
                   "sprite_generator")
sys.path.insert(0, SRC if os.path.isdir(SRC) else "/app")

import job_runner  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


# `run_stage` writes progress to Postgres. Capture it instead - this test is
# about the control flow, and a database would make it a stack test.
updates = []
job_runner.job_update = lambda job_id, **kw: updates.append(kw)

CELLS = 6
# A stage that speaks the real protocol: one `denoised N/M` line per "cell".
FAKE_STAGE = [
    sys.executable, "-c",
    "import sys, time\n"
    "for i in range(1, %d + 1):\n"
    "    print('denoised %%d/%%d' %% (i, %d), flush=True)\n"
    "    time.sleep(0.3)\n"
    "print('wrote %d cell(s)', flush=True)\n" % (CELLS, CELLS, CELLS),
]


def cells_seen():
    """How many cells the runner reported, from the captured progress writes."""
    msgs = [u.get("progress_msg", "") for u in updates]
    return [m for m in msgs if m.startswith("cell ")]


# --- 1. no cancel: the stage runs to completion, unchanged ----------------
updates.clear()
t0 = time.time()
rc, tail = job_runner.run_stage(FAKE_STAGE, "job-1", 0, 100)
ran_s = time.time() - t0
check("an uncancelled stage returns its real exit code", rc == 0, f"rc {rc}")
check("an uncancelled stage runs every cell", len(cells_seen()) == CELLS,
      f"saw {len(cells_seen())} of {CELLS}")
check("cancelled=None keeps the old behaviour (no cancel machinery)",
      rc == 0 and ran_s >= CELLS * 0.3 * 0.8, f"{ran_s:.2f}s")

# --- 2. THE POINT. A cancel stops it within a cell, not a stage -----------
updates.clear()
t0 = time.time()
rc, tail = job_runner.run_stage(FAKE_STAGE, "job-2", 0, 100,
                                cancelled=lambda: True, cancel_poll_s=0)
stopped_s = time.time() - t0
seen = len(cells_seen())
check("a cancelled stage returns CANCELLED_RC",
      rc == job_runner.CANCELLED_RC, f"rc {rc}")
check("a cancelled stage stops early", seen < CELLS,
      f"saw {seen} of {CELLS} cells")
check("it stops at the FIRST cell it checks", seen <= 2, f"saw {seen}")
check("and it stops quickly", stopped_s < CELLS * 0.3,
      f"{stopped_s:.2f}s against {CELLS * 0.3:.2f}s for the whole stage")
check("the caller is told where it stopped",
      any("cancelling at cell" in u.get("progress_msg", "") for u in updates),
      str([u.get("progress_msg") for u in updates][-2:]))

# --- 3. the process must actually be gone, not just abandoned -------------
# A stage holds the card. Returning while it runs on would defeat the point.
leaked = subprocess.run(
    [sys.executable, "-c",
     "import subprocess, sys;"
     "out = subprocess.run(['ps','-eo','args'], capture_output=True, text=True);"
     "print(sum('denoised' in l for l in out.stdout.splitlines()))"],
    capture_output=True, text=True)
if leaked.returncode == 0 and leaked.stdout.strip().isdigit():
    check("a cancelled stage leaves no process running",
          int(leaked.stdout.strip()) == 0,
          f"{leaked.stdout.strip()} fake stage process(es) still alive")
else:
    print("skip no process-table check available in this image")

# --- 4. a cancel that never fires must not disturb a healthy stage --------
updates.clear()
rc, tail = job_runner.run_stage(FAKE_STAGE, "job-4", 0, 100,
                                cancelled=lambda: False, cancel_poll_s=0)
check("a probe that says 'not cancelled' changes nothing",
      rc == 0 and len(cells_seen()) == CELLS,
      f"rc {rc}, saw {len(cells_seen())} of {CELLS}")

# --- 5. the status machine (SOMET-694 item 4) ------------------------------
import jobs  # noqa: E402

check("`cancelling` is not terminal", jobs.CANCELLING not in jobs.TERMINAL,
      f"TERMINAL={sorted(jobs.TERMINAL)} - a job still drawing must not read "
      f"as finished")
check("`cancelled` stays terminal", "cancelled" in jobs.TERMINAL)

with open(os.path.join(SRC if os.path.isdir(SRC) else "/app", "tasks.py"),
          encoding="utf-8") as fh:
    tasks_src = fh.read()
check("the worker treats `cancelling` as cancelled",
      'row[0] in ("cancelled", "cancelling")' in tasks_src,
      "otherwise DELETE writes a status the conveyor ignores")
check("both build tasks pass a cancel probe to the stage runner",
      tasks_src.count("cancelled=_cancel_probe(self.request.id, job_id)") == 2,
      f"{tasks_src.count('cancelled=_cancel_probe(self.request.id, job_id)')} "
      f"of 2 (sheet build and redraw)")
check("a cancelled stage is not reported as a stage failure",
      tasks_src.count("if rc == _CANCELLED_RC:") == 2,
      "a cancel must not set status=failed")

with open(os.path.join(SRC if os.path.isdir(SRC) else "/app", "jobs.py"),
          encoding="utf-8") as fh:
    jobs_src = fh.read()
check("DELETE re-issues on a terminal row whose task is still live",
      'if row["status"] in TERMINAL and not live:' in jobs_src,
      "a second DELETE must not answer 'already finished' while it draws")
check("DELETE clears finished_at when it asks a live task to stop",
      "finished_at = NULL" in jobs_src)
# Scoped to the revoke CALL, not the file: the comment above it says the
# words "terminate=True" to explain why they are not used, and a check that
# cannot tell code from the comment explaining it is a check that nags.
revokes = [l for l in jobs_src.splitlines() if "control.revoke(" in l]
check("DELETE revokes the task", len(revokes) == 1, str(revokes))
check("DELETE revokes without terminate (solo pool)",
      bool(revokes) and "terminate" not in revokes[0],
      "terminate would signal the only worker process: " + str(revokes))

check("revoked tasks leave the gateway's pending list",
      "@task_revoked.connect" in tasks_src
      and "_reap_pending_ghosts" in tasks_src,
      "the signal for what this worker sees, the boot sweep for the rest")

if failures:
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1)
print("\nall job-cancel checks passed")
