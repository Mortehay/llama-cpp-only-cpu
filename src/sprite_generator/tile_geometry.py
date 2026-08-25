"""The shape of an isometric ground tile. PIL only - no FastAPI, no database.

Split out of `tiles.py` so the WORKER can import it without dragging in a
FastAPI router, pydantic models and the auth module. `jobs.py` already records
the mirror-image rule - the API process must never import tasks.py, because
that pulls in torch - and this is the same boundary from the other side.

It also removed a failure: the worker imported `tiles` lazily inside the task
and got `ModuleNotFoundError: No module named 'tiles'`, while the identical
import worked from a shell in the same container with the same working
directory. Rather than keep guessing at the ambient sys.path of a Celery
prefork child, the geometry now lives in a module tasks.py imports at module
scope, alongside `core_models` - which has always resolved correctly.

A tile is a SHAPE, not a small sprite. A sprite is a subject with a silhouette;
a tile is a rhombus that must tessellate with its neighbours exactly, or the
ground shows seams. So the model never decides the outline: it paints texture,
and the rhombus is applied afterwards at the ratio the world actually uses.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

# Classic 2:1 dimetric - the projection most 2D "isometric" games actually use,
# and a 26.6 degree camera. Only a default: a measured tile overrides it.
DEFAULT_RATIO = 2.0
DEFAULT_TILE_W = 64


def diamond_mask(width: int, height: int) -> Image.Image:
    """An 'L' mask holding the tile rhombus.

    The four points land exactly on the edge midpoints, which is what makes
    adjacent tiles interlock without a seam. Drawn with width-1/height-1 so the
    points sit ON the last pixel rather than one past it - an off-by-one here
    leaves a transparent hairline down two sides of every tile, invisible on
    one tile and obvious across a field of them.
    """
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(
        [(width // 2, 0), (width - 1, height // 2),
         (width // 2, height - 1), (0, height // 2)],
        fill=255)
    return mask


def cut_tile(image: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop `image` to the tile aspect, resize, and mask to a rhombus."""
    target = width / height
    w, h = image.size
    if w / h > target:
        new_w = int(h * target)
        box = ((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h)
    else:
        new_h = int(w / target)
        box = (0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h)

    # LANCZOS on the way down; NEAREST would alias the texture badly. The
    # pixelation stage afterwards is what re-establishes a hard grid.
    cropped = image.crop(box).resize((width, height), Image.LANCZOS)
    out = cropped.convert("RGBA")
    out.putalpha(diamond_mask(width, height))
    return out


def tile_size_for(ratio: float, width: int = DEFAULT_TILE_W) -> tuple[int, int]:
    """(w, h) for a projection ratio, with an even height.

    Even because the rhombus's side vertices sit at height // 2; an odd height
    puts them half a pixel off centre and the tile stops tessellating cleanly.
    """
    height = max(2, int(round(width / max(ratio, 0.01))))
    if height % 2:
        height += 1
    return width, height
