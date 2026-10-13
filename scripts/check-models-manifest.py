#!/usr/bin/env python3
"""Fail when the code references a model the manifest does not account for.

    python scripts/check-models-manifest.py [--manifest PATH] [--src DIR]
                                            [--models-dir DIR]

WHY THIS EXISTS
`models.txt` did not rot because it was written badly. It rotted because
nothing ever compared it to the code, so the brain lane, the GGUF lane and the
Edit lane all grew up beside it unnoticed. This repo has the same bug class in
at least three other places - trigger_for/CORE_TRIGGERS, CORE_MODELS/
KNOWN_MODELS, model_gateway.GATED - and a manifest is a fourth unless something
checks it.

WHAT IT CHECKS
  1. Every Hub repo id in the source is DECLARED in the manifest or listed
     under [[dead]].
  2. Every `gguf:<stem>` in CORE_MODELS resolves to a declared hf-file entry
     (matching on the name the file is SAVED as, which is not always the name
     the Hub publishes).
  3. Every brain path in brain_engine.BRAINS is declared.
  4. Nothing is declared twice, and no [[dead]] entry is also declared live.
  5. Advisory: on-disk files nobody declares, and declared files not on disk.
     Reported, never fatal - a thin deploy is a choice.

Run on the host, no containers, no GPU: `make test-models-manifest`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import (  # noqa: E402
    KINDS, TIERS, cache_root, have, load, saved_name)

# A Hub repo id as it appears in Python source: "<org>/<name>" in quotes.
# Deliberately narrow. Broadening it to bare slashes drowns the result in
# MIME types, URL paths and docstring prose.
REPO_RE = re.compile(r'"([A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*)"')

# Strings that look like repo ids and are not. Each one is a false positive
# this check produced; keep the list tight and explain every addition.
NOT_REPOS = {
    "a/b",                     # docstring example
    "dungeon/level1",          # a map key
    "../..", "../escape",      # paths
}
NOT_REPO_PREFIXES = ("application/", "image/", "text/", "audio/", "multipart/",
                     "v1/", "api/", "./", "/")
NOT_REPO_SUFFIXES = (".py", ".json", ".png", ".sh", ".md", ".gguf", ".safetensors",
                     ".jpg", ".webp", ".wav", ".ogg")

GGUF_RE = re.compile(r'"gguf:([A-Za-z0-9][A-Za-z0-9_.\-]*)')
BRAIN_PATH_RE = re.compile(r'\{MODELS\}/([A-Za-z0-9_.\-/]+\.gguf)')


def looks_like_repo(s: str) -> bool:
    if s in NOT_REPOS or s.startswith(NOT_REPO_PREFIXES) or s.endswith(NOT_REPO_SUFFIXES):
        return False
    org, _, name = s.partition("/")
    # Hub ids do not contain spaces and both halves are non-trivial.
    return bool(org) and bool(name) and " " not in s and len(name) > 1


def scan(src: str, collect_stems: bool = True) -> tuple[dict[str, set[str]], set[str], set[str]]:
    """-> ({repo: {where}}, {gguf stems}, {brain relative paths})

    `.sh` as well as `.py`: on 2026-10-13 `scripts/two-step-test.sh` was found
    pinning BOTH dead models in shell variables, so `make test-flow` could
    only fail. A Python-only scan had missed it.
    """
    repos: dict[str, set[str]] = {}
    ggufs: set[str] = set()
    brains: set[str] = set()
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "node_modules", ".git")]
        for fn in files:
            if not fn.endswith((".py", ".sh")):
                continue
            # This file holds the patterns and the default paths; scanning it
            # only finds its own source strings.
            if fn == os.path.basename(__file__):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            rel = os.path.relpath(path, src)
            for m in REPO_RE.finditer(text):
                s = m.group(1)
                if looks_like_repo(s):
                    repos.setdefault(s, set()).add(rel)
            # ".." filters docstring placeholders like `a repo called "gguf:..."`.
            if collect_stems:
                ggufs.update(s for s in (m.group(1).split("+")[0]
                                         for m in GGUF_RE.finditer(text))
                             if ".." not in s)
            if collect_stems:
                brains.update(BRAIN_PATH_RE.findall(text))
    return repos, ggufs, brains


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=os.path.join(here, "compose/develop/downloader/models.toml"))
    # Two roots: the app, and the operator scripts. scripts/ is included
    # because a hardcoded dead model in a test harness fails exactly as badly
    # as one in the app, and is less likely to be noticed.
    ap.add_argument("--src", action="append", default=None,
                    help="directory to scan; repeatable "
                         "(default: src/sprite_generator and scripts)")
    ap.add_argument("--models-dir", default=None,
                    help="optional: also report on-disk drift")
    args = ap.parse_args()

    doc = load(args.manifest)
    models = doc.get("model", [])
    dead = {d["ref"] for d in doc.get("dead", [])}

    # Case-insensitive throughout. tasks.CORE_TRIGGERS and the step-2 tables are
    # keyed by LOWERCASED repo ids, so the same model legitimately appears in
    # two casings; the Hub treats ids case-insensitively too.
    declared_repos = {m["repo"] for m in models if m.get("repo")}
    declared_lc = {r.lower() for r in declared_repos}
    dead_lc = {r.lower() for r in dead}
    declared_files = {}   # saved filename -> entry name
    for m in models:
        if m.get("kind") == "hf-file":
            declared_files[saved_name(m)] = m["name"]

    errors: list[str] = []
    warnings: list[str] = []

    # --- 4. manifest self-consistency, first: a broken manifest makes every
    # other result untrustworthy.
    seen: set[str] = set()
    for m in models:
        n = m.get("name")
        if not n:
            errors.append("an entry has no `name`")
        elif n in seen:
            errors.append(f"duplicate entry name: {n}")
        seen.add(n)
        if not m.get("used_by"):
            errors.append(f"{n}: no `used_by` - every entry must name its caller")
        if m.get("tier") not in TIERS:
            errors.append(f"{n}: tier must be one of {'|'.join(TIERS)}, "
                          f"got {m.get('tier')!r}")
        kind = m.get("kind")
        if kind not in KINDS:
            errors.append(f"{n}: unknown kind {kind!r}")
        # Per-kind required fields. Each of these, left out, produces a
        # provisioner that fetches to the wrong place rather than an error.
        if kind == "hf-file":
            for f in ("repo", "file", "dest"):
                if not m.get(f):
                    errors.append(f"{n}: hf-file needs `{f}`")
            if m.get("dest", "").count("/") < 1:
                errors.append(f"{n}: dest {m.get('dest')!r} is one level deep; "
                              f"GGUFs must sit TWO levels below /models or the "
                              f"llama.cpp router advertises them as chat models")
        elif kind == "hf-cache":
            if not m.get("repo"):
                errors.append(f"{n}: hf-cache needs `repo`")
            if m.get("dest"):
                errors.append(f"{n}: hf-cache must NOT set `dest` - it lands in "
                              f"HF cache layout, resolved by repo id")
        elif kind == "checkpoint-tree":
            for f in ("repo", "dest"):
                if not m.get(f):
                    errors.append(f"{n}: checkpoint-tree needs `{f}`")
        elif kind == "local":
            if not m.get("files"):
                errors.append(f"{n}: local needs `files` - otherwise its "
                              f"absence cannot be detected")
            if m.get("repo"):
                errors.append(f"{n}: local must NOT set `repo` - it cannot be "
                              f"downloaded")
        if m.get("tier") == "optional" and not m.get("skipping"):
            errors.append(f"{n}: optional entries need `skipping` - what breaks "
                          f"without it is the whole basis for skipping it")
    for r in sorted(declared_lc & dead_lc):
        errors.append(f"{r} is both declared and listed dead")

    # --- 1. repo ids in the source
    srcs = args.src or [os.path.join(here, "src/sprite_generator"),
                        os.path.join(here, "scripts")]
    repos, ggufs, brains = {}, set(), set()
    for s in srcs:
        # `gguf:<stem>` and BRAINS paths are roster concepts and live only in
        # the app. Collecting them from scripts/ picks up test fixtures - the
        # gateway test's short label "gguf:qwen" is not a model - so repo ids
        # are checked everywhere and stems only in the app.
        r, g, b = scan(s, collect_stems=s.replace("\\", "/").endswith("sprite_generator"))
        for repo, where in r.items():
            repos.setdefault(repo, set()).update(
                os.path.join(os.path.basename(s), w) for w in where)
        ggufs |= g
        brains |= b
    for repo, where in sorted(repos.items()):
        if repo.lower() in declared_lc or repo.lower() in dead_lc:
            continue
        errors.append(
            f"UNDECLARED repo {repo}\n"
            f"      referenced in: {', '.join(sorted(where))}\n"
            f"      Add it to models.toml, or to [[dead]] with a reason.")

    # --- 2. gguf: stems from CORE_MODELS
    for stem in sorted(ggufs):
        if f"{stem}.gguf" not in declared_files:
            errors.append(
                f"UNDECLARED gguf stem '{stem}'\n"
                f"      CORE_MODELS offers gguf:{stem}, so "
                f"{stem}.gguf must be a declared hf-file (check `as`).")

    # --- 3. brain paths
    for rel in sorted(brains):
        fn = os.path.basename(rel)
        if fn not in declared_files:
            errors.append(
                f"UNDECLARED brain file '{rel}'\n"
                f"      brain_engine.BRAINS points at it; declare it as hf-file.")

    # --- 5. advisory, on-disk drift
    # Per-KIND, via the shared have(). An hf-file is a file at `dest`, not a
    # cache directory, so asking "is there a models--* for it" reports every
    # GGUF as missing - which is how the first version of this check managed to
    # call all three brains absent while they sat on disk.
    root = cache_root(doc, args.models_dir) if (args.models_dir or
                                                os.environ.get("HF_HUB_CACHE")) else None
    if root and os.path.isdir(root):
        for m in models:
            if m.get("kind") == "local":
                continue  # absence is a documented, normal state
            if not have(root, m):
                extra = " (also absent where the manifest was written)" \
                        if m.get("missing_on_source_host") else ""
                warnings.append(f"declared but not present: {m['name']}"
                                f" [{m['tier']}]{extra}")
        on_disk = {d[len("models--"):].replace("--", "/", 1)
                   for d in os.listdir(root) if d.startswith("models--")}
        # Splitting on the FIRST -- is wrong for an org containing one; none
        # here do, and this branch is advisory anyway.
        for repo in sorted(r for r in on_disk
                           if r.lower() not in declared_lc | dead_lc):
            warnings.append(f"on disk but undeclared: {repo}")

    # --- report
    print(f"manifest : {args.manifest}")
    print(f"source   : {', '.join(srcs)}")
    print(f"declared : {len(models)} models ({len(declared_repos)} repos), "
          f"{len(dead)} dead")
    print(f"scanned  : {len(repos)} repo refs, {len(ggufs)} gguf stems, "
          f"{len(brains)} brain paths")

    if warnings:
        print(f"\n{len(warnings)} advisory (not failures):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\nFAIL - {len(errors)} problem(s):\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK - every model the code references is declared or explicitly dead")
    return 0


if __name__ == "__main__":
    sys.exit(main())
