"""Is a turnaround reference framed as a whole figure, judged semantically?

WHY THIS CANNOT BE GEOMETRY (SOMET-696)

`sheet_qc.figure_ratio` compares every action cell to its own direction's
`dir_<d>.png`. When that reference is itself a close-up the ratio cancels - it
scored a chin-to-mid-thigh row at mean 1.014 against the 1.85-1.98 close-up
signature on record. The same shape of blindness as `facing` going dormant on
a weak reference: a reference-relative metric cannot judge the reference.

Five geometric proxies were tried against six hand-labelled references and
none separated them (2026-11-03):

    area x median   aspect   h_frac   bottom-band taper   profile-vs-median

The decisive failures: job `02cb757f` scores the MOST normal of all six on
area (1.04x) and h_frac (0.984) and is visibly cut at the thigh; `h_frac`
saturates near 1.0 because every render is edge to edge; and the taper idea -
a standing figure ends in two feet with air between them, a thigh cut ends in
one solid block - is confounded because in profile the legs overlap anyway.
Worse, `dir_e` aspect spans 0.729-0.904 across six references that are ALL the
same defect, which is exactly why the spread looked like seed variance and
produced a wrong "about 1 in 4" conclusion in this repo.

So the question - "is this figure complete, or cut off?" - is semantic, and it
is asked of a vision model.

JUDGE THE REFERENCE, NOT THE CELL. Eight calls fix eight whole rows; judging
cells instead would be hundreds of calls and could not fix anything anyway,
because every action cell is re-conditioned on `dir_<d>.png` and a redraw
re-inherits its framing.

THE BRAIN AND THE TRANSFORMER MUST NEVER SHARE THE CARD (0013). So this runs
as its own conveyor stage between `turnaround-denoise` and `actions-encode`,
when no transformer is resident, and it judges every direction inside ONE
brain load rather than one load per direction.

VISION IS NOT OPTIONAL AND ITS ABSENCE IS SILENT. The Qwen3VL language GGUF
has no vision tensors; without `--mmproj` the brain answers from the text
alone, fluently, and nothing in the reply says it never saw the picture. Every
entry point here refuses rather than guesses - see `brain_engine.has_vision`.
"""

from __future__ import annotations

import logging
import os

import brain_engine

logger = logging.getLogger(__name__)

BRAIN = os.environ.get("REFERENCE_QC_BRAIN", "qwen3-vl-8b")

# What the model must answer. `lowest_visible` is asked for BESIDE the boolean
# on purpose: a bare yes/no invites a confident yes, while naming the lowest
# body part it can see forces it to look at the bottom of the frame, and it
# also gives a human something to check the verdict against.
SCHEMA = {
    "type": "object",
    "properties": {
        "lowest_visible": {
            "type": "string",
            "enum": ["feet", "ankles", "shins", "knees", "thighs",
                     "hips", "waist", "chest"],
        },
        "head_complete": {"type": "boolean"},
        "whole_figure": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["lowest_visible", "head_complete", "whole_figure", "reason"],
    "additionalProperties": False,
}

PROMPT = (
    "This is a reference render of a single game character, used as the "
    "framing guide for an animation sheet. Judge ONLY how the figure is "
    "framed, not its art style or quality.\n\n"
    "Answer: what is the lowest part of the body visible in the image; "
    "whether the whole head is inside the frame; and whether the entire "
    "figure from head to feet is inside the frame.\n\n"
    "A correctly framed reference shows the complete character including the "
    "feet, with a little empty space around it. A badly framed one is zoomed "
    "in so that the legs or feet are outside the image, or the top of the "
    "head is cut off by the edge. Look carefully at the bottom edge: if the "
    "body continues to the edge and the feet are not visible, the figure is "
    "cut off."
)

# `thighs` and above means the legs are gone. `knees` is the boundary case and
# is treated as cut: a sheet needs feet to stand a character on the ground.
CUT_AT = {"knees", "thighs", "hips", "waist", "chest"}


class VisionUnavailable(RuntimeError):
    """The brain cannot see. Raised instead of returning a cheerful verdict."""


def judge_one(path: str, brain: str = BRAIN, max_tokens: int = 220) -> dict:
    """Judge one reference image. Returns the model's verdict plus `ok`.

    Raises VisionUnavailable when the projector is not provisioned, and
    BrainError on a brain failure - the caller decides whether a sheet should
    proceed unjudged, because that is a policy question, not a QC one.
    """
    if not brain_engine.has_vision(brain):
        raise VisionUnavailable(
            f"brain {brain} has no vision projector on disk; run "
            f"`make provision` for brain-vl-8b-mmproj. Refusing to judge, "
            f"because the language half would answer about an imagined image.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out = brain_engine.complete(
        brain,
        [brain_engine.image_message(PROMPT, [path])],
        schema=SCHEMA, temperature=0.0, max_tokens=max_tokens)
    v = dict(out.get("json") or {})
    if not v:
        raise brain_engine.BrainError(
            "failed", f"no JSON verdict for {os.path.basename(path)}: "
                      f"{(out.get('text') or '')[:200]}")
    # Trust the two signals that required looking at an edge, not the summary
    # boolean: `whole_figure` is the one a model can answer agreeably.
    v["ok"] = bool(v.get("head_complete")) and \
        v.get("lowest_visible") not in CUT_AT
    v["path"] = path
    return v


def judge_references(work: str, directions, brain: str = BRAIN) -> dict:
    """Judge every `dir_<d>.png` in a work directory inside ONE brain load.

    Returns {direction: verdict}. Missing files are skipped rather than
    failed - a job that rendered a subset of directions is legitimate.

    The CALLER is responsible for the card being free. From inside a worker
    job that means `complete_in_job`'s eviction; as a conveyor stage it means
    running between turnaround-denoise and actions-encode.
    """
    out: dict[str, dict] = {}
    try:
        for d in directions:
            p = os.path.join(work, f"dir_{d}.png")
            if not os.path.exists(p):
                continue
            out[d] = judge_one(p, brain=brain)
            logger.info("reference %s: %s (lowest=%s, head=%s)", d,
                        "ok" if out[d]["ok"] else "CUT",
                        out[d].get("lowest_visible"),
                        out[d].get("head_complete"))
    finally:
        # Free the card whatever happened: the next stage wants a transformer
        # and the two cannot coexist.
        brain_engine.stop("reference judging done")
    return out
