"""Serve something2 map specs: generate one, look at it, take it or leave it.

SYNCHRONOUS ON PURPOSE

Every other surface here is a 202 plus polling, because a sheet is an hour of
GPU and a tile is a minute. A map spec is neither - it is arithmetic over a
biome table - so this is the one thing this service can hand something2 inside
the single blocking POST their provider system supports (`docs/ai-providers.md`:
"Sync services only"). No job row, no queue, no cache-reader facade.

STORAGE IS THE FILESYSTEM

A spec is a file that `make seed-map SPEC=<name>` reads. Putting it in Postgres
would mean exporting it again to be useful, so the generated `.map.json` IS the
artefact and the listing is a directory. No migration, and the files are
visible in the repo's images mount like everything else.

THE LLM'S JOB, AND WHAT IS NOT ITS JOB

It picks BIOMES per world - the semantic half, tens of decisions with meaning.
It never picks densities or counts: `world_gen.choose_density` solves those
against something2's own tier table so a target creatures-per-screen is
actually hit. A model cannot be trusted to do arithmetic it has no feedback on,
and does not need to be. If no text model is loaded the plan falls back to the
deterministic author and the response says so rather than pretending.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import auth
import world_gen
import world_preview

logger = logging.getLogger(__name__)
router = APIRouter()

WORLDS_DIR = os.environ.get("WORLDS_DIR", "/app/images/worlds")
# Short. The brain is an improvement to a path that already works without it,
# so a slow or absent brain must cost a few seconds, never the request. Shorter
# than a cold 35B load on purpose: a world spec that wins a cold card falls
# back to rules, and the load carries on for the next caller (decisions/0013).
LLM_TIMEOUT = float(os.environ.get("WORLD_LLM_TIMEOUT", "45"))

SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,58}[a-z0-9]$")


class WorldSpec(BaseModel):
    name: str = Field(..., min_length=2, max_length=60,
                      description="region name; also the SPEC= filename")
    worlds: int = Field(6, ge=1, le=36)
    # The knob that fixes empty space. Their own table: sparse is 2/screen,
    # normal 4, dense 8. Anything under 3 reads as emptiness.
    target_per_screen: float = Field(
        6.0, ge=0.5, le=25.0,
        description="creatures visible on one screen, after biome multipliers")
    size: int = Field(128, ge=32, le=224)
    chunk_size: int = Field(32, ge=8, le=64)
    biome_cell: int = Field(32, ge=8, le=64)
    theme: str | None = Field(None, description="what this region is, for the LLM")
    # The look of this region's sound effects (0010 D8, level 2 of the engine
    # precedence). Stored in the `.gen.json` sidecar only - never in the spec
    # something2 seeds - and read by POST /api/audio/sfx when a request names
    # this world. Not used by map generation at all.
    sfx_engine: str | None = Field(None, pattern="^(realistic|retro)$")
    author: str = Field("rules", pattern="^(rules|llm)$")
    overwrite: bool = False
    # Overwriting a region that was hand-edited in the game's World Map Editor
    # has to be asked for by name, exactly as on PATCH - an overwrite IS a
    # regeneration. See `_refuse_if_hand_edited`.
    discard_hand_edits: bool = False


def _slug(s: str) -> str:
    return world_gen._slug(s).replace("_", "-")


def _paths(name: str):
    # Refused rather than sanitised. `_slug` would happily turn "../escape"
    # into "escape" - which cannot leave WORLDS_DIR, so it is not a traversal,
    # but it IS a lookup silently resolving to a DIFFERENT region than the one
    # asked for. A caller that types a path deserves an error, not somebody
    # else's map.
    if any(c in name for c in "/\\") or ".." in name:
        raise HTTPException(status_code=400,
                            detail=f"region name {name!r} looks like a path")

    s = _slug(name)
    if not SAFE_NAME.match(s):
        raise HTTPException(status_code=400, detail=f"unusable region name {name!r}")
    return (os.path.join(WORLDS_DIR, f"{s}.map.json"),
            os.path.join(WORLDS_DIR, f"{s}.preview.png"),
            # The generation parameters, beside the spec but NOT inside it.
            # `<name>.map.json` is something2's format and must stay exactly
            # their shape; storing "how this was generated" in it would ship
            # fields their seeder never asked for. This sidecar is what makes
            # editing possible - without it a PATCH would have to reverse
            # engineer the request from its own output.
            os.path.join(WORLDS_DIR, f"{s}.gen.json"))


# --- hand edits: the game's World Map Editor writes back here -------------
#
# A region can now arrive from two directions: generated here, or authored by
# an admin in something2's World Map Editor and PUT back whole. The two are not
# interchangeable - regenerating a hand-edited region silently discards the
# admin's work - so the fact that hand edits EXIST has to live on disk, not in
# the shape of the spec.
#
# It is its own file rather than a key in `<name>.gen.json`, for two reasons.
# The sidecar means exactly one thing, "how this was generated", and both
# `list_worlds` and `check-artifacts.py` read its PRESENCE as "this region can
# be regenerated"; a sidecar holding a flag but no parameters would be listed
# as editable and then fail on the first PATCH. And a region from before
# sidecars existed has no `.gen.json` at all, yet can still be hand-edited.
EDIT_SUFFIX = ".edit.json"

# Read-compare-write must not interleave with another request's. This is one
# process (uvicorn, no --workers) but FastAPI runs sync handlers in a
# threadpool, so two PUTs for the same region really can overlap: without the
# lock both read the same sha256, both match, and the second silently
# overwrites the first - the lost update If-Match exists to prevent.
_SPEC_LOCK = threading.Lock()


def _edit_path(name: str) -> str:
    map_path, _, _ = _paths(name)
    return map_path[: -len(".map.json")] + EDIT_SUFFIX


def _hand_edit(name: str) -> dict:
    """Is this region hand-edited, and when was it last written?

    Always the same two keys, so no consumer has to branch on whether the
    marker file exists.
    """
    path = _edit_path(name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                marker = json.load(fh)
            if marker.get("hand_edited"):
                return {"hand_edited": True,
                        "edited_at": marker.get("edited_at")}
        except Exception as e:
            # An unreadable marker must NOT be read as "no hand edits" - that
            # is the reading that regenerates over somebody's work.
            logger.warning("unreadable hand-edit marker for %s: %s", name, e)
            return {"hand_edited": True, "edited_at": None}
    return {"hand_edited": False, "edited_at": None}


def _mark_hand_edited(name: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(_edit_path(name), "w", encoding="utf-8") as fh:
        json.dump({"hand_edited": True, "edited_at": stamp}, fh, indent=1)
    return stamp


def _clear_hand_edit(name: str) -> None:
    try:
        os.remove(_edit_path(name))
    except FileNotFoundError:
        pass


def _refuse_if_hand_edited(name: str, discard: bool) -> None:
    """Regenerating over hand edits is refused unless it is asked for by name.

    Both doors come through here - PATCH, and POST with overwrite - because a
    guard on one of them is not a guard. The flag itself is cleared by
    `_write_region`, so every path that rebuilds a region drops it exactly
    once.
    """
    if not _hand_edit(name)["hand_edited"] or discard:
        return
    raise HTTPException(
        status_code=409,
        detail="region has hand edits; pass discard_hand_edits=true "
               "to regenerate")


def _sha256_file(path: str) -> str:
    """The hash of the stored spec's EXACT bytes - the ETag something2 sees."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _etag(sha: str) -> str:
    """The quoted form. An ETag is a quoted string in HTTP; the bare sha is
    still accepted on the way IN, so a caller that stored what it read works
    either way - see `_if_match_sha`."""
    return f'"{sha}"'


def _if_match_sha(header: str | None) -> str | None:
    """The sha256 a caller claims to have last seen, however they quoted it."""
    if header is None:
        return None
    value = header.strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    return value.strip('"').strip() or None


def _write_region(name: str, params: dict, plan_kwargs: dict, note: str) -> dict:
    """Generate, write all three files, and return the response body.

    One function for POST and PATCH so a created region and an edited one can
    never diverge in shape or in what gets written to disk.
    """
    map_path, png_path, gen_path = _paths(name)

    region = world_gen.plan_region(name, **plan_kwargs)
    rep = world_gen.report(region, plan_kwargs.get("target_per_screen"))

    os.makedirs(WORLDS_DIR, exist_ok=True)
    with open(map_path, "w", encoding="utf-8") as fh:
        fh.write(world_gen.to_json(region))
    with open(gen_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1)
    try:
        world_preview.render(region, rep).save(png_path)
    except Exception as e:
        # A missing preview is worth a warning, never a failed generation - the
        # spec is the artefact and it is already on disk.
        logger.warning("preview failed for %s: %s", name, e)

    # A regeneration supersedes whatever was hand-edited here. Cleared in the
    # ONE place every rebuild passes through rather than at each caller.
    _clear_hand_edit(name)

    slug = _slug(name)
    return {
        "name": slug,
        "author": note,
        "params": params,
        "report": rep,
        "spec_url": f"/api/worlds/{slug}",
        "preview_url": f"/api/worlds/{slug}/preview.png",
        "seed_with": f"make seed-map SPEC={slug}",
        "spec": region,
    }


def _parse_plan(blob: str):
    """Read a nested array of biome names out of almost-JSON.

    A 3B model asked for `[["Meadow"]]` will cheerfully answer `[[Meadow]]`,
    which is not JSON. Rather than reject the answer over punctuation, the
    fallback scans each bracketed group for names that ACTUALLY EXIST in the
    biome table - which is the same validation the strict path applies anyway,
    so nothing unsafe gets through a looser reader.
    """
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass

    groups = re.findall(r"\[([^\[\]]*)\]", blob)
    if not groups:
        return None
    out = []
    for g in groups:
        found = [b for b in world_gen.BIOMES if re.search(rf"\b{re.escape(b)}\b", g)]
        if found:
            out.append(found)
    return out or None


def _llm_biome_plan(theme: str | None, count: int) -> tuple[list | None, str]:
    """Ask for a biome per world. Returns (plan, note) - never raises.

    Constrained hard by the prompt and re-validated here, because the failure
    mode of a small instruct model is a confident biome name that does not
    exist. Anything unrecognised is dropped rather than repaired, and a short
    plan simply hands the rest back to the deterministic author.
    """
    # The biome table, with the ONE property the model cannot infer and keeps
    # getting wrong: how much flora a biome carries. Left to itself it pairs
    # Arid Dunes with Frozen Waste because they sound like a journey, and both
    # carry two flora types - so the ground reads bare between the creatures.
    # There is no per-world decoration density in something2 to fix that after
    # the fact, so it has to be avoided at the point of choosing.
    # The model is given the SURFACE/DEEP split, not just the biome names,
    # because left to itself it picks surface biomes for a surface-sounding
    # theme and never reaches the deep ones at all. That validates and reads
    # fine - and it caps the region's creature variety, because the surface
    # biomes admit only the four legacy creatures between them while the deep
    # ones carry the P4 families.
    #
    # A descent is also what keeps a region COHERENT. something2's own warning:
    # a "green river valley" theme spanning Catacombs and Emberdepths will read
    # strangely even though it validates. Deep biomes belong further out, not
    # scattered among meadows.
    surface = ", ".join(world_gen.SURFACE)
    deep = ", ".join(world_gen.DEEP)
    rich = [n for n, b in world_gen.BIOMES.items()
            if len(b["flora"]) >= world_gen.THIN_FLORA_BELOW]

    prompt = (
        f"You are laying out a {count}-world region for a 2D fantasy RPG"
        + (f" themed: {theme}. " if theme else ". ")
        + f"Choose 1-2 biomes for each of the {count} worlds, IN ORDER, as a "
        f"DESCENT: world 1 is the safe surface starting area and the last "
        f"world is the deepest and harshest. "
        f"SURFACE biomes (open air, use these for the first 2-3 worlds): "
        f"{surface}. "
        f"DEEP biomes (underground or ruined, use these for the later worlds): "
        f"{deep}. "
        f"RULES: the first world must be surface only. The last third must be "
        f"deep. Do not give two worlds the same pair. Prefer pairing a biome "
        f"with a plant-rich one ({', '.join(rich[:6])}) so the ground is not "
        f"bare, except for the deliberately barren ones "
        f"({', '.join(sorted(world_gen.FLORA_OPTIONAL)[:3])} and similar), "
        f"which are meant to be empty. "
        f"Reply with ONLY a JSON array of {count} arrays of biome names, "
        f'spelled exactly as above. Example: [["Meadow"],["Deep Forest","Mire"]]'
    )
    import text as text_api
    try:
        # Through the gated brain (decisions/0013): refused at once while
        # another model holds the card, which is a rule fallback, not an error.
        try:
            res = text_api.run_text(
                prompt=prompt, temperature=0.7, max_tokens=400,
                route="internal:worlds", wait_s=LLM_TIMEOUT,
                caller={"principal_name": "worlds (internal)"})
        except text_api.TextRefused as e:
            return None, (f"brain unavailable ({e.reason}: {e.detail}) - "
                          f"biomes chosen deterministically instead")
        model, text = res.get("model"), res.get("text") or ""
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return None, "LLM returned no array at all - fell back to rules"

        raw = _parse_plan(m.group(0))
        if raw is None:
            return None, "LLM's array could not be read - fell back to rules"

        plan, dropped = [], []
        for entry in raw[:count]:
            entry = entry if isinstance(entry, list) else [entry]
            keep = [b for b in entry if b in world_gen.BIOMES]
            dropped += [str(b) for b in entry if b not in world_gen.BIOMES]
            if keep:
                plan.append(keep[:2])
        if not plan:
            return None, "LLM named no biome that exists - fell back to rules"

        note = f"biomes authored by {model} ({len(plan)}/{count} worlds)"
        if dropped:
            note += f"; dropped invented biome(s): {', '.join(sorted(set(dropped)))}"
        return plan, note
    except Exception as e:
        return None, f"LLM call failed ({type(e).__name__}) - fell back to rules"


@router.post("/api/worlds", status_code=201)
def create_world(spec: WorldSpec, authorization: str | None = Header(None)):
    """Generate a region, write it, and hand back the verdict with it."""
    auth.require(authorization, "generate")

    map_path, _, _ = _paths(spec.name)
    if os.path.exists(map_path):
        if not spec.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"{_slug(spec.name)} already exists - pass overwrite to "
                       f"replace it, or PATCH it to change one thing")
        _refuse_if_hand_edited(spec.name, spec.discard_hand_edits)

    plan, note = None, "biomes chosen deterministically"
    if spec.author == "llm":
        plan, note = _llm_biome_plan(spec.theme, spec.worlds)

    params = spec.model_dump(exclude={"overwrite", "discard_hand_edits"})
    # The biome plan is stored, not just used. A PATCH that changes the size
    # must not silently re-roll the biomes, and re-asking the LLM would give a
    # different answer every time - so an edit reuses the plan the region was
    # built with unless it is explicitly re-authored.
    params["biome_plan"] = plan
    return _write_region(spec.name, params, _plan_kwargs(params), note)


def _plan_kwargs(params: dict) -> dict:
    """Stored parameters -> plan_region's arguments. One place, two callers."""
    return {
        "world_count": params["worlds"],
        "target_per_screen": params["target_per_screen"],
        "size": params["size"],
        "chunk_size": params["chunk_size"],
        "biome_cell": params["biome_cell"],
        "biome_plan": params.get("biome_plan"),
    }


class WorldEdit(BaseModel):
    """Every field optional: a PATCH changes what it names and nothing else."""
    worlds: int | None = Field(None, ge=1, le=36)
    target_per_screen: float | None = Field(None, ge=0.5, le=25.0)
    size: int | None = Field(None, ge=32, le=224)
    chunk_size: int | None = Field(None, ge=8, le=64)
    biome_cell: int | None = Field(None, ge=8, le=64)
    theme: str | None = None
    sfx_engine: str | None = Field(None, pattern="^(realistic|retro)$")
    # Re-ask the LLM for biomes. Off by default: an edit to the creature target
    # should not quietly redraw the whole region's character.
    reauthor: bool = False
    # Throw away hand edits made in the game's World Map Editor and rebuild
    # from parameters. Off by default, and a hand-edited region refuses a PATCH
    # without it rather than quietly regenerating over the admin's work.
    discard_hand_edits: bool = False


@router.patch("/api/worlds/{name}")
def edit_world(name: str, edit: WorldEdit,
               authorization: str | None = Header(None)):
    """Change one thing about an existing region and rebuild it.

    Reads the stored generation parameters, applies only the fields the caller
    named, and regenerates. Everything not named is carried over - including
    the biome plan, so raising the creature target does not also re-roll which
    biomes the region is made of.
    """
    auth.require(authorization, "generate")

    map_path, _, gen_path = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")
    if not os.path.exists(gen_path):
        raise HTTPException(
            status_code=409,
            detail=f"{_slug(name)} was generated before edits were supported "
                   f"and has no stored parameters - POST it again with "
                   f"overwrite to make it editable")

    # After the sidecar check, not before: a region that cannot be regenerated
    # at all should say THAT first, rather than naming a flag which would then
    # fail on the next line.
    _refuse_if_hand_edited(name, edit.discard_hand_edits)

    with open(gen_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    changes = edit.model_dump(exclude_none=True,
                              exclude={"reauthor", "discard_hand_edits"})
    params.update(changes)

    note = "edited; biomes carried over"
    if edit.reauthor:
        plan, note = _llm_biome_plan(params.get("theme"), params["worlds"])
        params["biome_plan"] = plan
    elif changes.get("worlds") and params.get("biome_plan"):
        # A region that grew has more worlds than the stored plan covers; the
        # deterministic author fills the tail rather than the plan being
        # stretched over worlds it never chose biomes for.
        note = "edited; stored biomes reused, new worlds chosen deterministically"

    out = _write_region(name, params, _plan_kwargs(params), note)
    out["changed"] = sorted(changes) + (["biome_plan"] if edit.reauthor else [])
    return out


@router.get("/api/worlds")
def list_worlds(authorization: str | None = Header(None)):
    """Every generated region. This is something2's catalogue of what exists."""
    auth.require(authorization, "read")

    items = []
    for path in sorted(glob.glob(os.path.join(WORLDS_DIR, "*.map.json"))):
        name = os.path.basename(path)[: -len(".map.json")]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                region = json.load(fh)
            # Params first, because the report needs the TARGET to say whether
            # the region actually hit it - and the target only exists in the
            # sidecar. `editable` is what a UI needs before offering an edit
            # form: a region generated before the sidecar existed can only be
            # replaced, not patched.
            gen_path = os.path.join(WORLDS_DIR, f"{name}.gen.json")
            params = None
            if os.path.exists(gen_path):
                try:
                    with open(gen_path, "r", encoding="utf-8") as gh:
                        params = json.load(gh)
                except Exception:
                    params = None

            rep = world_gen.report(
                region, (params or {}).get("target_per_screen"))
            items.append({
                "name": name,
                "region": region.get("name"),
                "editable": params is not None,
                "params": params,
                # Whether this spec came back from the game's editor. A UI
                # needs it before offering a regeneration that would discard
                # it, and something2 needs it to know which copy is authored.
                **_hand_edit(name),
                "worlds": rep["totals"]["worlds"],
                "creatures": rep["totals"]["creatures"],
                "mean_per_screen": rep["totals"]["mean_per_screen"],
                "empty_worlds": rep["totals"]["empty_worlds"],
                "ok": rep["ok"],
                "bytes": os.path.getsize(path),
                "created_at": os.path.getmtime(path),
                "spec_url": f"/api/worlds/{name}",
                "preview_url": f"/api/worlds/{name}/preview.png",
            })
        except Exception as e:
            # One unreadable file must not blank the listing.
            items.append({"name": name, "error": str(e)})
    return {"items": items, "total": len(items)}


@router.get("/api/worlds/{name}")
def get_world(name: str, download: bool = False,
              authorization: str | None = Header(None)):
    """The map spec itself - the file `make seed-map` reads.

    Carries an `ETag`: the sha256 of the stored file's EXACT bytes. That is
    what `PUT /api/worlds/{name}/spec` wants back in `If-Match`, so a hand edit
    can only replace the version it was made against.
    """
    auth.require(authorization, "read")

    map_path, _, _ = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")

    etag = _etag(_sha256_file(map_path))
    if download:
        return FileResponse(map_path, media_type="application/json",
                            filename=f"{_slug(name)}.map.json",
                            headers={"ETag": etag})
    with open(map_path, "r", encoding="utf-8") as fh:
        # A response object rather than the bare dict, because the header is
        # half of what this route is now for.
        return JSONResponse(content=json.load(fh), headers={"ETag": etag})


# --- hand-edited specs IN: a PUT, never a PATCH --------------------------

def _validate_posted_spec(body, slug: str) -> list:
    """Refuse a body that is not this region, and say which part is wrong.

    STRUCTURAL ONLY, on purpose. something2's World Map Editor is the authority
    on what a hand-authored region should contain, so a spec that is merely
    thin, crowded or oddly linked is stored and REPORTED - refusing it would
    lose an admin's work over a judgement this service is not entitled to make.
    `world_gen.report` still runs on the way in, and its problems come back
    with the 200.

    What IS refused is a body this service could not serve afterwards: no
    worlds, a world with no key, duplicate keys, a link into a world that does
    not exist. Those turn the region's own /report into a 500 for good.
    """
    errs = []
    if not isinstance(body, dict):
        return ["the body must be a spec object"]

    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        errs.append("name is required")
    elif _slug(name) != slug:
        # Compared as SLUGS, not literally. A region created as "Emerald Reach"
        # is stored at `emerald-reach.map.json` with `"name": "Emerald Reach"`
        # inside it, so a literal comparison would refuse that region its own
        # spec back - which is the round trip this endpoint exists for.
        errs.append(f"name {name!r} is not this region - the path says {slug!r}")

    posted = body.get("worlds")
    if not isinstance(posted, list) or not posted:
        errs.append("worlds must be a non-empty list")
        return errs

    keys = set()
    for i, w in enumerate(posted):
        if not isinstance(w, dict):
            errs.append(f"worlds[{i}] is not an object")
            continue
        key = w.get("key")
        if not isinstance(key, str) or not key:
            errs.append(f"worlds[{i}] has no key")
        elif key in keys:
            errs.append(f"worlds[{i}]: duplicate key {key!r}")
        else:
            keys.add(key)

    links = body.get("links")
    if links is None:
        links = []
    if not isinstance(links, list):
        errs.append("links must be a list")
        return errs

    for i, l in enumerate(links):
        if not isinstance(l, dict):
            errs.append(f"links[{i}] is not an object")
            continue
        if l.get("from") not in keys:
            errs.append(f"links[{i}]: from {l.get('from')!r} is not a world in "
                        f"this region")
        to, to_name = l.get("to"), l.get("to_name")
        if to_name is not None:
            # A cross-region doorway. `to_name` is a world NAME in ANOTHER
            # region's spec, which this file cannot resolve and this service
            # does not try to - checked for being a name, and no further.
            if not isinstance(to_name, str) or not to_name:
                errs.append(f"links[{i}]: to_name must be a world name")
        elif to not in keys:
            errs.append(f"links[{i}]: to {to!r} is not a world in this region, "
                        f"and no to_name was given")
        if l.get("kind") == "edge" and l.get("edge") not in world_gen.EDGES:
            errs.append(f"links[{i}]: kind 'edge' needs an explicit edge, one "
                        f"of {', '.join(world_gen.EDGES)}")
    return errs


@router.put("/api/worlds/{name}/spec")
def put_spec(name: str, body: dict = Body(...),
             if_match: str | None = Header(None, alias="If-Match"),
             authorization: str | None = Header(None)):
    """Store a hand-edited spec as the region's current spec. NO regeneration.

    THE OTHER DIRECTION. Everything else here turns parameters into a spec;
    this takes a spec that was authored in something2's World Map Editor, so
    the box's copy is what the game actually runs. PATCH cannot do it - PATCH
    takes generation parameters and rebuilds, which is precisely what must not
    happen to a map somebody drew.

    Unknown keys are stored unchanged. Their format already carries three
    optional fields this service does not generate and must not drop - a
    world's `graph: [x, y]` (a dragged graph position that wins over
    grid x 220), a link's `kind: "edge"` with an explicit `edge` and no
    grid-delta rule, and a link's `to_name` (a world NAME in another region,
    used instead of a spec-local `to`) - and that list will grow on their side,
    not here. So the body is taken as-is rather than through a model that would
    silently drop whatever it has not been taught.

    `If-Match` is required rather than optional. Two admins editing the same
    region is the ordinary case this is built for, and a blind write loses one
    of them with no trace; a 409 costs a re-read.
    """
    auth.require(authorization, "generate")

    map_path, png_path, _ = _paths(name)
    slug = _slug(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")

    # 428 BEFORE 422: a caller with no If-Match has not read this region yet,
    # so complaining about their body would send them to fix the wrong thing.
    claimed = _if_match_sha(if_match)
    if not claimed:
        raise HTTPException(
            status_code=428,
            detail="If-Match with the sha256 from the last GET is required - "
                   "a blind write would lose whatever is stored")

    errs = _validate_posted_spec(body, slug)
    if errs:
        raise HTTPException(status_code=422, detail=errs)

    try:
        rep = world_gen.report(body)
    except Exception as e:
        # Refused, because the alternative is a stored region whose own /report
        # is a 500 for ever. This is "this service cannot read it", not "it is
        # not good enough" - the quality verdicts ride along with the 200.
        raise HTTPException(
            status_code=422,
            detail=[f"this service cannot read the spec: "
                    f"{type(e).__name__}: {e}"])

    with _SPEC_LOCK:
        current = _sha256_file(map_path)
        if claimed != "*" and claimed != current:
            # NOT an HTTPException: the contract puts `sha256` beside `detail`
            # at the top level, and HTTPException would nest it a level deeper.
            return JSONResponse(
                status_code=409,
                content={"detail": "spec changed on the box",
                         "sha256": current},
                headers={"ETag": _etag(current)})

        # Written whole, then moved into place. A PUT interrupted half way
        # through would otherwise leave the seeder a truncated spec, and the
        # next caller's ETag would be the hash of a file nobody authored.
        #
        # Serialised through `to_json` - the same writer generation uses - so a
        # spec that goes out and comes back unchanged is byte-identical and
        # keeps its ETag, rather than drifting by two spaces of indentation.
        tmp = map_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(world_gen.to_json(body))
        os.replace(tmp, map_path)
        sha = _sha256_file(map_path)
        stamp = _mark_hand_edited(name)

    try:
        world_preview.render(body, rep).save(png_path)
    except Exception as e:
        # The generation rule: the spec is the artefact and it is already on
        # disk, so a preview is worth a warning and never the request.
        logger.warning("preview failed for hand-edited %s: %s", name, e)

    return JSONResponse(
        content={
            "name": slug,
            "sha256": sha,
            "hand_edited": True,
            "edited_at": stamp,
            # The verdicts, not a gate. Stored either way, so a caller can see
            # what it just saved without a second round trip.
            "problems": rep["problems"],
            "spec_url": f"/api/worlds/{slug}",
            "preview_url": f"/api/worlds/{slug}/preview.png",
        },
        headers={"ETag": _etag(sha)})


@router.get("/api/worlds/{name}/report")
def get_report(name: str, authorization: str | None = Header(None)):
    """Is it empty? Answered without downloading the spec to find out."""
    auth.require(authorization, "read")

    map_path, _, _ = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")
    with open(map_path, "r", encoding="utf-8") as fh:
        rep = world_gen.report(json.load(fh))
    # Whether a human drew this rather than the generator. Here as well as in
    # the listing, because this is the route a consumer calls to DECIDE, and
    # "the box generated it" and "an admin authored it" are different answers.
    rep.update(_hand_edit(name))
    return rep


@router.get("/api/worlds/{name}/preview.png")
def get_preview(name: str, authorization: str | None = Header(None)):
    """One screenful per world, so emptiness is visible rather than described."""
    auth.require(authorization, "read")

    map_path, png_path, _ = _paths(name)
    if not os.path.exists(png_path):
        if not os.path.exists(map_path):
            raise HTTPException(status_code=404, detail=f"no region {name!r}")
        # Regenerate rather than 404: the spec is the source of truth and the
        # preview is derived, so a missing PNG is a cache miss, not an error.
        with open(map_path, "r", encoding="utf-8") as fh:
            region = json.load(fh)
        world_preview.render(region).save(png_path)

    with open(png_path, "rb") as fh:
        return Response(content=fh.read(), media_type="image/png")


@router.delete("/api/worlds/{name}")
def delete_world(name: str, authorization: str | None = Header(None)):
    auth.require(authorization, "generate")

    map_path, png_path, gen_path = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")
    removed = []
    # The sidecar goes too. Leaving it behind would make a later region of the
    # same name inherit the deleted one's parameters on its first PATCH - and
    # the hand-edit marker likewise, which would make a freshly generated
    # region refuse its own first PATCH as somebody else's hand edits.
    for p in (map_path, png_path, gen_path, _edit_path(name)):
        try:
            os.remove(p)
            removed.append(os.path.basename(p))
        except FileNotFoundError:
            pass
    return {"deleted": removed}
