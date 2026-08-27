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
import json
import logging
import os
import re

import requests
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import auth
import world_gen
import world_preview

logger = logging.getLogger(__name__)
router = APIRouter()

WORLDS_DIR = os.environ.get("WORLDS_DIR", "/app/images/worlds")
LLM_URL = os.environ.get("LLM_URL", "http://llm-server:8080")

# Short. The LLM is an improvement to a path that already works without it, so
# a slow or absent model must cost a few seconds, never the request.
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
    author: str = Field("rules", pattern="^(rules|llm)$")
    overwrite: bool = False


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


def _write_region(name: str, params: dict, plan_kwargs: dict, note: str) -> dict:
    """Generate, write all three files, and return the response body.

    One function for POST and PATCH so a created region and an edited one can
    never diverge in shape or in what gets written to disk.
    """
    map_path, png_path, gen_path = _paths(name)

    region = world_gen.plan_region(name, **plan_kwargs)
    rep = world_gen.report(region)

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


def _llm_model() -> str | None:
    """Which model to route to.

    llama.cpp runs in ROUTER mode here (`--models-dir`), which means it serves
    several models and picks by the request's `model` field. Omitting it is a
    400, not a default - that is what made the first version of this always
    report the LLM as unavailable while the server was healthy.
    """
    override = os.environ.get("WORLD_LLM_MODEL")
    if override:
        return override
    r = requests.get(f"{LLM_URL}/v1/models", timeout=10)
    data = r.json().get("data") or []
    return data[0]["id"] if data else None


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
    try:
        model = _llm_model()
        if not model:
            return None, ("no text model loaded in llama.cpp - biomes chosen "
                          "deterministically instead")

        # Two attempts, because the router loads models ON DEMAND: the first
        # call after an idle period spends ~13s loading and answers with a body
        # that is not the completion JSON. That is a cold start, not a failure,
        # and retrying once turns it into a 0.4s success. `--sleep-idle-seconds
        # 120` means any generation after two quiet minutes pays it.
        text = None
        for attempt in (1, 2):
            r = requests.post(
                f"{LLM_URL}/v1/chat/completions",
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.7, "max_tokens": 400},
                timeout=LLM_TIMEOUT)
            if r.status_code != 200:
                if attempt == 2:
                    return None, (f"LLM unavailable (HTTP {r.status_code}) - "
                                  f"biomes chosen deterministically instead")
                continue
            try:
                text = r.json()["choices"][0]["message"]["content"]
                break
            except (ValueError, KeyError, IndexError):
                if attempt == 2:
                    return None, ("LLM answered with something that was not a "
                                  "completion - fell back to rules")
        if text is None:
            return None, "LLM did not answer - fell back to rules"
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
    if os.path.exists(map_path) and not spec.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"{_slug(spec.name)} already exists - pass overwrite to "
                   f"replace it, or PATCH it to change one thing")

    plan, note = None, "biomes chosen deterministically"
    if spec.author == "llm":
        plan, note = _llm_biome_plan(spec.theme, spec.worlds)

    params = spec.model_dump(exclude={"overwrite"})
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
    # Re-ask the LLM for biomes. Off by default: an edit to the creature target
    # should not quietly redraw the whole region's character.
    reauthor: bool = False


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

    with open(gen_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    changes = edit.model_dump(exclude_none=True, exclude={"reauthor"})
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
            rep = world_gen.report(region)
            # `editable` is what a UI needs to know before offering an edit
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
            items.append({
                "name": name,
                "region": region.get("name"),
                "editable": params is not None,
                "params": params,
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
    """The map spec itself - the file `make seed-map` reads."""
    auth.require(authorization, "read")

    map_path, _, _ = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")

    if download:
        return FileResponse(map_path, media_type="application/json",
                            filename=f"{_slug(name)}.map.json")
    with open(map_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/api/worlds/{name}/report")
def get_report(name: str, authorization: str | None = Header(None)):
    """Is it empty? Answered without downloading the spec to find out."""
    auth.require(authorization, "read")

    map_path, _, _ = _paths(name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"no region {name!r}")
    with open(map_path, "r", encoding="utf-8") as fh:
        return world_gen.report(json.load(fh))


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
    # same name inherit the deleted one's parameters on its first PATCH.
    for p in (map_path, png_path, gen_path):
        try:
            os.remove(p)
            removed.append(os.path.basename(p))
        except FileNotFoundError:
            pass
    return {"deleted": removed}
