"""Compose generated cells into a sprite sheet plus an atlas.

The last stage of the conveyor. Takes a pile of individual frames keyed by
(action, direction, frame index), pixelates them against ONE shared palette, and
lays them out on a grid that divides evenly.

Layout: one row per (action, direction) pair, one column per frame. That is the
RPG-Maker-ish shape of the supplied reference sheets and it is the shape
something2 slices - its provider system takes a flat grid plus a column and row
count, and rejects a sheet that does not divide evenly.

Why the palette is extracted here and not per-cell: consistency. See
`pixelate.pixelate_sheet`. The palette comes from the CONCEPT image when one is
supplied, so every character's sheet is anchored to the art that was approved in
the UI rather than to whichever frame happened to be measured first.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

import numpy as np
from PIL import Image

import pixelate

logger = logging.getLogger(__name__)

# Canonical direction order. Rows are emitted in this order so that a consumer
# can index a direction arithmetically instead of reading the atlas - and so
# that two characters' sheets are row-compatible with each other.
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

# 4-way sheets are still common and cost half the rows. Kept as a named subset
# rather than a flag so the atlas records which convention was used.
DIRECTIONS_4 = ["s", "e", "n", "w"]

# Below this mean per-pixel step, two consecutive frames are the same picture
# as far as a player is concerned. Measured: a clean row's smallest step runs
# 3-15, while the pairs worth seeing sit at 0.18-0.84.
DUPLICATE_STEP = float(os.environ.get("SHEET_DUPLICATE_STEP", "1.0"))


class SheetBuilder:
    """Accumulate frames, then emit a grid PNG and an atlas JSON.

    Usage:
        b = SheetBuilder(cell=(48, 64), frames=6, directions=DIRECTIONS)
        b.add("walk", "s", 0, img)
        ...
        b.save("walk_sheet.png", "walk_sheet.json", concept=concept_img)
    """

    def __init__(self, cell=(48, 64), frames: int = 6, directions=None,
                 n_colors: int = pixelate.DEFAULT_COLORS,
                 alpha_threshold: int = pixelate.DEFAULT_ALPHA_THRESHOLD,
                 margin: int = 1, mirrored=None,
                 pinned_scale: float | None = None):
        self.cell_w, self.cell_h = cell
        self.frames = frames
        self.directions = list(directions or DIRECTIONS)
        self.n_colors = n_colors
        self.alpha_threshold = alpha_threshold
        self.margin = margin
        # {target direction: source direction} for rows that were produced by
        # flipping another row rather than rendered. Recorded in the atlas and
        # nowhere else - the flip itself happens in the caller, because it has
        # to happen before the cells are added or the shared palette and scale
        # would be measured over a half-populated sheet.
        self.mirrored = dict(mirrored or {})
        # The scale a previous build of this same sheet settled on. See
        # cell_images: it is what keeps a one-frame redraw from resizing
        # every other cell.
        self.pinned_scale = float(pinned_scale) if pinned_scale else None
        # Ordered so rows come out in insertion order of the ACTION, with
        # directions in canonical order inside each action.
        self._cells: dict[tuple[str, str, int], Image.Image] = {}
        self._actions: "OrderedDict[str, None]" = OrderedDict()

    def add(self, action: str, direction: str, frame: int, img: Image.Image):
        # A mirror SOURCE is accepted even when it is not an emitted row. A
        # sheet can ask for `w` without asking for `e`, and `w` is built by
        # flipping `e` - so the source has to be fittable without becoming a
        # row of its own. Without this the request renders a source nothing
        # consumes and emits an empty west row.
        if (direction not in self.directions
                and direction not in self.mirrored.values()):
            raise KeyError(f"{direction!r} not in {self.directions} "
                           f"and not a mirror source of any of them")
        if not 0 <= frame < self.frames:
            raise IndexError(f"frame {frame} out of range 0..{self.frames - 1}")
        self._actions[action] = None
        self._cells[(action, direction, frame)] = img

    # --- layout ---------------------------------------------------------

    def rows(self):
        """The (action, direction) pair for each row, in emission order."""
        return [(a, d) for a in self._actions for d in self.directions]

    def missing(self):
        """Every cell that was never supplied. Empty list means complete.

        A mirrored row counts as supplied when the row it is flipped from is:
        its cells are produced in `cell_images`, from the finished source
        cell, and never passed to `add`.
        """
        return [(a, d, f)
                for a, d in self.rows()
                for f in range(self.frames)
                if (a, d, f) not in self._cells
                and (a, self.mirrored.get(d), f) not in self._cells]

    # --- output ---------------------------------------------------------

    def cell_images(self, concept: Image.Image | None = None,
                    allow_missing: bool = False):
        """Return {(action, direction, frame): finished cell_w x cell_h RGBA}.

        The finished cells, palette-locked and fitted at the one shared scale -
        exactly what `build` pastes into the grid. Split out so the conveyor
        can write each cell to its own PNG and then assemble the sheet from
        those same objects: a sheet that is composed from the per-frame files
        cannot disagree with them, which is the property the redraw endpoint
        depends on.
        """
        gaps = self.missing()
        if gaps and not allow_missing:
            shown = ", ".join(f"{a}/{d}/{f}" for a, d, f in gaps[:6])
            more = f" (+{len(gaps) - 6} more)" if len(gaps) > 6 else ""
            raise ValueError(
                f"{len(gaps)} cell(s) never supplied: {shown}{more}. "
                f"Pass allow_missing=True to emit them as transparent holes."
            )

        rows = self.rows()
        if not rows:
            raise ValueError("no frames were added")

        # One palette for the whole character. The concept image wins when
        # given: it is the art the user approved, and anchoring to it keeps two
        # sheets for the same character (walk, attack) sharing a palette even
        # though they are built in separate jobs.
        source = concept if concept is not None else self._collage()
        palette = pixelate.extract_palette(source, self.n_colors,
                                           self.alpha_threshold)

        # That anchoring is an asymmetry - a palette measured on ONE artifact
        # and applied to others - so it was checked against the failure the map
        # path hit, where a DECLARED terrain colour sat 77.7 Lab from the
        # painted water, further away than a mid-grey, and quantisation handed
        # the entire sea to stone. The declared colour captured 0.0% and the
        # only symptom was a coverage warning that read like a nag.
        #
        # Measured across every sheet in images/: no entry is displaced that
        # way. Every entry that captures pixels sits at mean 0.0 Lab from them,
        # because `extract_palette` median-cuts the art itself rather than
        # naming colours - a derived palette cannot be far from the pixels it
        # came from. Sixteen entries across six sheets do capture 0.0%, but
        # they are near-duplicates crowded out by a neighbour 0.4-1.9 Lab away
        # (one sheet asks for 256 entries and has seven near-identical dark
        # maroons), which wastes palette slots and displaces nothing.
        #
        # So the hazard is real but lives where colours are DECLARED, and
        # nothing in this pipeline declares one. If a caller ever passes a
        # hand-written palette to `pixelate`, this check is the one to redo.

        # One scale for the whole character, for the same reason
        # pixelate_sheet computes one: a per-cell fit makes the sprite pulse.
        #
        # It is the MINIMUM fit scale over every cell, so the widest pose
        # still fits inside the cell - which also means one bloated cell
        # shrinks all of the others. That is why the quality checks measure
        # the figure BEFORE the fit: after it, a close-up has already
        # contaminated every cell it is being compared against.
        fit = None
        for img in self._cells.values():
            box = pixelate._alpha_bbox(img, self.alpha_threshold)
            if box is None:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            s = min((self.cell_w - 2 * self.margin) / w,
                    (self.cell_h - 2 * self.margin) / h)
            fit = s if fit is None else min(fit, s)
        if fit is None:
            raise ValueError("no supplied cell has any opaque pixels")

        # A pinned scale keeps a REDRAW from resizing the whole sheet.
        #
        # Re-assembling after one frame changed would otherwise recompute the
        # minimum over a set with one new member, and a redrawn frame that is
        # a little wider than the old one rescales all 336 cells. "Replacing
        # one frame changes only that cell" would then be false in the most
        # confusing possible way: every sprite in the game shifts because one
        # was repaired.
        #
        # Only honoured when it is not LARGER than the fit the cells actually
        # need, because a pin that is too big pastes a sprite wider than the
        # cell and `Image.paste` silently clips it.
        scale = fit
        self._refit = False
        if self.pinned_scale:
            # Tolerance, then clamp. The pin is read back from the atlas,
            # where it is rounded to six decimals, so a pin that came from
            # THIS EXACT set of cells reads very slightly larger than the
            # scale it was derived from. Compared exactly, re-composing an
            # unchanged sheet declared a refit and rescaled nothing - a
            # warning about a resize that had not happened.
            #
            # The tolerance is ABSOLUTE, because the error it covers is: six
            # decimal places is at most 5e-7 either way, whatever the
            # magnitude. A relative 1e-6 was tried and still fired at a scale
            # of 0.121, where the rounding error is 1.65e-6 of the value.
            #
            # The clamp keeps the no-clipping guarantee the comparison exists
            # for.
            if self.pinned_scale <= fit + 1e-6:
                scale = min(self.pinned_scale, fit)
            else:
                # The sheet DOES rescale in this case, and that is the lesser
                # evil: honouring the pin would paste a sprite wider than the
                # cell and `Image.paste` would clip it silently. One scale per
                # sheet is the goal; a clipped sprite is a defect. The caller
                # is told, through atlas["refit"], so "only that cell changed"
                # is never claimed when it is false.
                self._refit = True
                logger.warning(
                    "pinned scale %.5f does not fit these cells (they need "
                    "%.5f); recomputing, so every cell in this sheet changes "
                    "size. A redrawn frame is wider than anything in the "
                    "original build.", self.pinned_scale, fit)

        # Every supplied cell, not just the ones that are rows: a mirror
        # source may have been supplied without being emitted.
        out = {}
        for key, img in self._cells.items():
            out[key] = pixelate.pixelate(
                img, self.cell_w, self.cell_h, palette=palette,
                alpha_threshold=self.alpha_threshold,
                scale=scale, margin=self.margin)

        # Mirrored rows are flipped AFTER the fit, from the finished cell.
        #
        # Flipping the 512px render and fitting it separately does NOT give a
        # mirror image: `pixelate` crops to the alpha bbox and centres with
        # `(cell_w - width) // 2`, so an odd remainder rounds the sprite one
        # pixel to the left in BOTH cells and the pair is off by one. Measured
        # on synthetic cells - every frame of the flipped row differed from
        # the mirror of its source. In game that is a sprite that jumps a
        # pixel sideways when the character turns around.
        #
        # Flipping the finished cell is a pure transpose of a 48x64 grid, so
        # the two rows are exact mirrors by construction.
        for target, src in self.mirrored.items():
            for action, direction in rows:
                if direction != target:
                    continue
                for f in range(self.frames):
                    base = out.get((action, src, f))
                    if base is not None:
                        out[(action, target, f)] = base.transpose(
                            Image.FLIP_LEFT_RIGHT)

        self._palette = palette
        self._scale = scale
        # Only the emitted rows. A mirror source that is not itself a row was
        # fitted above so its flip could be derived, and must not reach the
        # grid or the frames directory.
        wanted = {(a, d) for a, d in rows}
        return {k: v for k, v in out.items() if (k[0], k[1]) in wanted}

    def build(self, concept: Image.Image | None = None,
              allow_missing: bool = False, quality=None, cells=None):
        """Return (sheet_image, atlas_dict).

        A missing cell is an error by default. Silently emitting a transparent
        hole produces a sheet that passes every structural check and shows a
        character vanishing for one frame in game - the kind of defect that is
        found by a player, not by a test.

        `cells` lets a caller pass finished cells back in - the ones it just
        wrote to disk - so the sheet is demonstrably the concatenation of the
        per-frame files rather than a second rendering of them.
        """
        if cells is None:
            cells = self.cell_images(concept=concept,
                                     allow_missing=allow_missing)
        rows = self.rows()
        palette = getattr(self, "_palette", None)
        if palette is None:
            raise ValueError("cell_images must run before build")

        sheet = Image.new(
            "RGBA",
            (self.frames * self.cell_w, len(rows) * self.cell_h),
            (0, 0, 0, 0),
        )

        atlas_rows = []
        for r, (action, direction) in enumerate(rows):
            for f in range(self.frames):
                img = cells.get((action, direction, f))
                if img is None:
                    continue
                sheet.paste(img, (f * self.cell_w, r * self.cell_h))
            entry: dict = {
                "row": r, "action": action, "direction": direction,
                # The directory the per-frame PNGs live in, so a consumer
                # that wants one cell does not have to slice the grid.
                "frames_dir": f"{action}_{direction}"}
            motion = self.row_motion(cells, action, direction)
            if motion:
                entry["motion"] = motion
            src = self.mirrored.get(direction)
            if src:
                # Named rather than implied. something2 mirrors in game and is
                # unaffected, but a consumer that does NOT mirror has to be
                # able to see that this row's handedness is reversed - a
                # one-handed sword is in the other hand here than in the row it
                # was flipped from.
                entry["mirrored_from"] = src
                entry["handedness"] = "swapped"
            atlas_rows.append(entry)

        atlas: dict = {
            "cell": {"w": self.cell_w, "h": self.cell_h},
            "grid": {"columns": self.frames, "rows": len(rows)},
            "directions": self.directions,
            "frames_per_action": self.frames,
            "palette": [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette],
            "rows": atlas_rows,
            # something2 reads a flat grid and slices it itself; these are the
            # exact values to type into its provider form.
            "something2": {"sprite_sheet": "flat",
                           "columns": self.frames,
                           "rows": len(rows)},
            # The one scale every cell was fitted at. Recorded so a later
            # re-assembly - a redraw - can pin it and leave the other cells
            # exactly as they were.
            "scale": round(float(getattr(self, "_scale", 0.0)), 6),
        }
        if getattr(self, "_refit", False):
            # A redrawn frame was wider than anything in the original build,
            # so every cell was rescaled. A consumer that caches cells has to
            # know the whole sheet moved, not one cell.
            atlas["refit"] = True
        if self.mirrored:
            atlas["mirrored"] = dict(self.mirrored)
        if quality is not None:
            # Additive, and present even when every frame passed: a consumer
            # cannot tell "nothing was checked" from "nothing failed" if the
            # key only appears on failure.
            atlas["quality"] = quality
        return sheet, atlas

    def _collage(self):
        """All cells side by side, only ever used to sample a palette from."""
        imgs = list(self._cells.values())
        w = sum(i.width for i in imgs)
        h = max(i.height for i in imgs)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x = 0
        for i in imgs:
            out.paste(i, (x, 0))
            x += i.width
        return out

    def row_motion(self, cells: dict, action: str, direction: str) -> dict:
        """How much this row actually moves, frame to frame.

        REPORTED, NOT CHECKED, and that distinction is the whole point.

        The per-frame quality checks all score a cell against a REFERENCE, so
        a row of six near-identical frames passes every one of them. This is
        the measurement that would notice, and it deliberately does not feed
        the redraw loop: motion is a property of a ROW, and a row that does
        not move needs different pose text, not another seed. Re-rolling a
        frame cannot fix a prompt that asks for a pose the camera cannot see.

        Measured on the 7-action Cultist sheet, mean step by view family,
        excluding idle and sway which are meant to be still:

            side 19.08    front 14.84    back 12.20

        and the two that gave this away: `walk n` at 4.40 - a walk cycle
        barely moving - and `attack n` with consecutive frames 0.39 apart.
        Both are back rows, and every action's `back` pose list was a
        verbatim copy of its `front` list at the time.

        `duplicates` counts consecutive pairs under DUPLICATE_STEP. Some are
        structural rather than a defect: the conveyor renders one image per
        distinct (direction, pose) and copies it, so a pose list that repeats
        an entry produces byte-identical frames by construction. Either way
        it is a frame of the animation that does not animate, which is worth
        seeing.
        """
        frames = [cells.get((action, direction, f)) for f in range(self.frames)]
        frames = [f for f in frames if f is not None]
        if len(frames) < 2:
            return {}
        arr = [np.asarray(f, dtype=np.float32) for f in frames]
        # Wraps from the last frame back to the first: these play as loops,
        # so the closing step is as real as the others.
        steps = [float(np.abs(arr[i] - arr[(i + 1) % len(arr)]).mean())
                 for i in range(len(arr))]
        return {
            "mean_step": round(float(np.mean(steps)), 2),
            "min_step": round(min(steps), 2),
            "duplicates": sum(1 for s in steps if s < DUPLICATE_STEP),
        }

    def write_frames(self, root: str, cells: dict) -> dict:
        """Write every finished cell to `root/<action>_<direction>/<n>.png`.

        Returns {row_name: [relative paths]} for the atlas. The layout is the
        owner's: a sheet IS a folder of per-frame images, and sheet.png is
        assembled from them. That is what makes a single-frame redraw possible
        without regenerating anything else - replace one PNG, re-assemble, and
        every other cell is byte-identical because it was never touched.
        """
        index = {}
        for (action, direction, f), img in sorted(cells.items()):
            row = f"{action}_{direction}"
            os.makedirs(os.path.join(root, row), exist_ok=True)
            rel = os.path.join(row, f"{f}.png")
            img.save(os.path.join(root, rel))
            index.setdefault(row, []).append(rel.replace(os.sep, "/"))
        return index

    def save(self, png_path: str, atlas_path: str | None = None,
             concept: Image.Image | None = None, allow_missing: bool = False,
             frames_root: str | None = None, quality=None):
        cells = self.cell_images(concept=concept, allow_missing=allow_missing)
        frames_index = None
        if frames_root:
            frames_index = self.write_frames(frames_root, cells)
        sheet, atlas = self.build(concept=concept, allow_missing=allow_missing,
                                  quality=quality, cells=cells)
        if frames_index is not None:
            atlas["frames"] = frames_index
        sheet.save(png_path)

        atlas_path = atlas_path or os.path.splitext(png_path)[0] + ".json"
        atlas["image"] = os.path.basename(png_path)
        with open(atlas_path, "w") as f:
            json.dump(atlas, f, indent=2)

        size_mb = os.path.getsize(png_path) / 1024 ** 2
        # something2 caps a sheet at 32 MB. A palette-locked 48px sheet is tiny
        # (tens of KB), so breaching this means something upstream is wrong -
        # most likely the pixelation stage was skipped.
        if size_mb > 32:
            raise ValueError(
                f"{png_path} is {size_mb:.1f} MB; something2 rejects anything "
                f"over 32 MB. A pixelated sheet this size means the palette "
                f"lock did not run."
            )
        return sheet, atlas
