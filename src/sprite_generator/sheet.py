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
import os
from collections import OrderedDict

from PIL import Image

import pixelate

# Canonical direction order. Rows are emitted in this order so that a consumer
# can index a direction arithmetically instead of reading the atlas - and so
# that two characters' sheets are row-compatible with each other.
DIRECTIONS = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

# 4-way sheets are still common and cost half the rows. Kept as a named subset
# rather than a flag so the atlas records which convention was used.
DIRECTIONS_4 = ["s", "e", "n", "w"]


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
                 margin: int = 1):
        self.cell_w, self.cell_h = cell
        self.frames = frames
        self.directions = list(directions or DIRECTIONS)
        self.n_colors = n_colors
        self.alpha_threshold = alpha_threshold
        self.margin = margin
        # Ordered so rows come out in insertion order of the ACTION, with
        # directions in canonical order inside each action.
        self._cells: dict[tuple[str, str, int], Image.Image] = {}
        self._actions: "OrderedDict[str, None]" = OrderedDict()

    def add(self, action: str, direction: str, frame: int, img: Image.Image):
        if direction not in self.directions:
            raise KeyError(f"{direction!r} not in {self.directions}")
        if not 0 <= frame < self.frames:
            raise IndexError(f"frame {frame} out of range 0..{self.frames - 1}")
        self._actions[action] = None
        self._cells[(action, direction, frame)] = img

    # --- layout ---------------------------------------------------------

    def rows(self):
        """The (action, direction) pair for each row, in emission order."""
        return [(a, d) for a in self._actions for d in self.directions]

    def missing(self):
        """Every cell that was never supplied. Empty list means complete."""
        return [(a, d, f)
                for a, d in self.rows()
                for f in range(self.frames)
                if (a, d, f) not in self._cells]

    # --- output ---------------------------------------------------------

    def build(self, concept: Image.Image | None = None,
              allow_missing: bool = False):
        """Return (sheet_image, atlas_dict).

        A missing cell is an error by default. Silently emitting a transparent
        hole produces a sheet that passes every structural check and shows a
        character vanishing for one frame in game - the kind of defect that is
        found by a player, not by a test.
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

        # One scale for the whole character, for the same reason
        # pixelate_sheet computes one: a per-cell fit makes the sprite pulse.
        scale = None
        for img in self._cells.values():
            box = pixelate._alpha_bbox(img, self.alpha_threshold)
            if box is None:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            s = min((self.cell_w - 2 * self.margin) / w,
                    (self.cell_h - 2 * self.margin) / h)
            scale = s if scale is None else min(scale, s)
        if scale is None:
            raise ValueError("no supplied cell has any opaque pixels")

        sheet = Image.new(
            "RGBA",
            (self.frames * self.cell_w, len(rows) * self.cell_h),
            (0, 0, 0, 0),
        )

        atlas_rows = []
        for r, (action, direction) in enumerate(rows):
            for f in range(self.frames):
                img = self._cells.get((action, direction, f))
                if img is None:
                    continue
                sheet.paste(
                    pixelate.pixelate(img, self.cell_w, self.cell_h,
                                      palette=palette,
                                      alpha_threshold=self.alpha_threshold,
                                      scale=scale, margin=self.margin),
                    (f * self.cell_w, r * self.cell_h),
                )
            atlas_rows.append({"row": r, "action": action,
                               "direction": direction})

        atlas = {
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
        }
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

    def save(self, png_path: str, atlas_path: str | None = None,
             concept: Image.Image | None = None, allow_missing: bool = False):
        sheet, atlas = self.build(concept=concept, allow_missing=allow_missing)
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
