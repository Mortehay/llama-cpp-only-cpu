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

    Built row by row rather than as a polygon, because tessellation is a
    property of the ROWS and polygon rasterisation does not respect it. Tiles
    are laid on a (width/2, height/2) lattice, so a row and the row half a tile
    below it must together span exactly one tile:

        w(y) + w(y + height/2) == width      for every y

    Each row is measured from its own midline (y + 0.5), which makes the top
    and bottom halves exact mirrors and gives a mask of exactly
    width*height/2 opaque pixels.

    The previous polygon put its vertices at (width-1, height/2) and
    (width/2, height-1) to keep them "on the last pixel". That made the diamond
    asymmetric - measured row widths 1,5,9,13,16,20,24,28,32,28,23,... summing
    to 248 of the required 256 - so w(0) + w(8) was 33, not 32, and a field of
    tiles showed transparent pinholes along every other diagonal. Invisible on
    one tile, obvious across a map.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    centre = width / 2

    for y in range(height):
        rows_from_edge = (y + 0.5) if y < height / 2 else (height - y - 0.5)
        half_w = rows_from_edge * width / height
        x0 = int(round(centre - half_w))
        x1 = int(round(centre + half_w))
        if x1 > x0:
            draw.line([(max(0, x0), y), (min(width, x1) - 1, y)], fill=255)

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
