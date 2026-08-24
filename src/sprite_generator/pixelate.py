"""Turn a soft, anti-aliased render into genuine pixel art.

Deterministic: no model, no GPU, no network. This is the stage the conveyor has
never had, and its absence is why output looks like *a painting of* pixel art
rather than pixel art - see `.ai/decisions/0005`.

Three things separate the two, and all three are arithmetic:

1.  **Resolution.** A sprite cell is 32-64px. Generating at 512 and saving at
    512 means every "pixel" is really a 10x10 smooth gradient.
2.  **Palette.** Real pixel art draws from a small fixed set of colours. A
    diffusion sample uses tens of thousands.
3.  **Alpha.** A sprite edge is opaque or absent. Anti-aliased edges leave a
    fringe that reads as a halo once the sprite is composited over a game tile.

The palette is also where most of the *consistency* win lives, which is not
obvious. At 48px a hue shift between two cells is what reads as "a different
character" far more than a silhouette difference does. Extracting ONE palette
and snapping every cell to it removes most visible drift for free, before any
VRAM is spent on identity preservation.

Two rules that are easy to get wrong when doing this per-cell:

-   **One scale for the whole sheet, not one per cell.** Fitting each cell to
    its own bounding box makes the character grow and shrink between frames.
    `pixelate_sheet` measures every cell first and applies a single factor.
-   **Bottom-anchor, do not centre.** Sprites are ground-anchored. Centring on
    the bounding box makes the character bob vertically as limbs extend.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Alpha at or above this becomes fully opaque; below it becomes fully
# transparent. There is no middle: a partially transparent pixel is what
# produces the halo this module exists to remove.
DEFAULT_ALPHA_THRESHOLD = 128

# Palette size. 16-32 is the range the supplied reference sheets sit in.
DEFAULT_COLORS = 24

# Fraction of the sprite's height above its bottom edge that counts as "shins",
# used as the width reference when detecting a ground patch. 15-35% clears the
# feet and the ground blob while staying below the hips on every character
# tested here.
SHIN_BAND = (0.15, 0.35)


# --- colour -------------------------------------------------------------

def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) float CIELAB (D65).

    Nearest-colour matching is done in Lab rather than RGB because RGB distance
    does not track how different two colours look. Snapping in RGB reliably
    picks a wrong-but-numerically-close colour for skin and shadow tones, which
    is exactly where a sprite's identity lives.
    """
    arr = rgb.astype(np.float64) / 255.0

    # sRGB -> linear
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    # linear RGB -> XYZ (sRGB primaries, D65)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T

    # Normalise by the D65 white point
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    return np.stack([
        116.0 * f[..., 1] - 16.0,
        500.0 * (f[..., 0] - f[..., 1]),
        200.0 * (f[..., 1] - f[..., 2]),
    ], axis=-1)


def extract_palette(img: Image.Image,
                    n_colors: int = DEFAULT_COLORS,
                    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD) -> np.ndarray:
    """Median-cut the OPAQUE pixels of `img` into an (n, 3) uint8 palette.

    Transparent and near-transparent pixels are excluded deliberately. Including
    them spends palette entries on the background and on the anti-aliased fringe
    between background and character - typically a third of the budget on
    colours that will not survive the alpha threshold anyway.
    """
    rgba = np.asarray(img.convert("RGBA"))
    opaque = rgba[..., 3] >= alpha_threshold
    pixels = rgba[opaque][:, :3]

    if pixels.size == 0:
        raise ValueError("image has no pixels at or above the alpha threshold")

    # Median cut wants an image, not a pixel list. Reshaping into a tall strip
    # is equivalent for quantisation purposes and avoids resampling.
    strip = Image.fromarray(pixels.reshape(-1, 1, 3), mode="RGB")
    n = min(n_colors, len(np.unique(pixels.reshape(-1, 3), axis=0)))
    quantised = strip.quantize(colors=n, method=Image.Quantize.MEDIANCUT)

    raw = quantised.getpalette()[: n * 3]
    return np.array(raw, dtype=np.uint8).reshape(-1, 3)


def snap_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Replace every pixel with its nearest palette entry, matched in Lab."""
    flat = rgb.reshape(-1, 3)
    lab_px = _srgb_to_lab(flat)
    lab_pal = _srgb_to_lab(palette)

    # Chunked so a 1024x1024 frame does not allocate an N x P float64 matrix in
    # one go - that is 8 bytes * 1M * 24, which is survivable, but the same code
    # runs over a whole sheet.
    out = np.empty(len(flat), dtype=np.int64)
    step = 65536
    for i in range(0, len(flat), step):
        d = lab_px[i:i + step, None, :] - lab_pal[None, :, :]
        out[i:i + step] = np.einsum("ijk,ijk->ij", d, d).argmin(axis=1)

    return palette[out].reshape(rgb.shape)


# --- background ---------------------------------------------------------

def strip_ground_patch(img: Image.Image, width_ratio: float = 1.8,
                       max_band_frac: float = 0.25) -> Image.Image:
    """Delete a ground/shadow patch fused to the bottom of a sprite.

    Moved here from tasks.py, which now imports it, because it is pure
    PIL/numpy geometry and belongs beside the other cleanup stages - and
    because the 2D conveyor needs it OUTSIDE the Celery worker, where importing
    tasks would drag in celery, psycopg2, torch and a CUDA availability check.

    Why it is needed at all, for the Qwen route: the concept image has NO ground
    patch, but `fal/...-Multiple-Angles-LoRA` was trained on Gaussian-Splatting
    renders of real captured scenes, which always have a floor. It carries that
    prior and invents a dirt patch under the character in every direction.
    Suppressing it by prompt was tried and abandoned - see ADR 0005 - because
    the extra tokens cost more VRAM than the fix was worth.

    Detection is by WIDTH, not colour. Colour would have to know this particular
    ground is brown, which does not generalise; a stone slab or a pool of shadow
    is not. Width does: feet are narrow and the thing they stand on spreads out.
    Measured on the zombie core, legs run ~99px and the patch peaks at 319px - a
    3.2x step no part of a character body produces.

    KNOWN GAP, carried over verbatim: on a STOCKY character with an arm held
    out, the body median inflates and the patch ratio falls just under
    width_ratio, so the patch is missed - and lowering the threshold then walks
    up through the boots and amputates the feet. The rule assumes a body whose
    reference width is dominated by the torso and whose feet are clearly
    narrower. True for slim characters, false for stocky ones.

    Use on rest-pose turnarounds. Be careful applying it to ACTION frames: a
    wide attack stance or a flared robe genuinely is wider at the bottom, and
    clipping that is worse than leaving the patch.
    """
    arr = np.array(img.convert("RGBA"))
    opaque = arr[:, :, 3] > 0
    rows = np.where(opaque.any(axis=1))[0]
    if rows.size == 0:
        return img

    top, bot = int(rows[0]), int(rows[-1])
    height = bot - top + 1
    widths = opaque.sum(axis=1).astype(float)

    # Reference width, measured on the SHINS - not the whole body.
    #
    # The original rule used the body median, and that is what made this fail on
    # front and back views: those present the shoulders at full span, which
    # inflates the median until the patch no longer clears width_ratio. Lowering
    # the ratio to catch it then let the hysteresis walk run up through the
    # boots and amputate the legs (measured 2026-08-23, both failure modes).
    #
    # The shins are the right thing to compare against, because the question is
    # "is this wider than the thing standing on it" - and what stands on the
    # ground is legs, not shoulders. A shin band is narrow from EVERY angle,
    # which is exactly the property the body median lacks.
    lo = bot - int(height * SHIN_BAND[1])
    hi = bot - int(height * SHIN_BAND[0])
    shins = widths[max(lo, top):max(hi, top + 1)]
    shins = shins[shins > 0]
    if shins.size == 0:
        # Fall back to the old whole-body reference rather than refusing: a
        # sprite too short to have a shin band is not necessarily broken.
        shins = widths[top:max(bot + 1 - int(height * 0.25), top + 1)]
        shins = shins[shins > 0]
    if shins.size == 0:
        return img
    reference = float(np.median(shins))
    if reference <= 0:
        return img

    band = np.arange(max(bot - int(height * max_band_frac), top), bot + 1)
    wide = band[widths[band] > reference * width_ratio]
    if wide.size == 0:
        return img

    # Cut from the TOPMOST offending row, not upward from the bottom: the patch
    # tapers to a few pixels of grass at its lowest rows, so a scan starting at
    # the bottom edge stops on the first narrow row and removes nothing.
    cut = int(wide.min())

    # Then walk further up with a LOWER threshold. One threshold cannot do both
    # jobs: the high one must be high enough that a wide stance is not mistaken
    # for ground, but the patch tapers rather than stopping abruptly, and
    # cutting only above the high threshold leaves that taper behind - which
    # then reads as a fresh shadow. Legs are NARROWER than the body median, so
    # they cannot sustain this walk; the taper can.
    floor = max(top, bot - int(height * max_band_frac))
    while cut - 1 >= floor and widths[cut - 1] > reference * 1.05:
        cut -= 1

    arr[cut:bot + 1, :, 3] = 0
    return Image.fromarray(arr)


def key_background(img: Image.Image, tolerance: int = 24,
                   min_enclosed: int = 200) -> Image.Image:
    """Flood-fill the background to transparent, starting from the corners.

    Only for images that arrive without usable alpha. A corner flood fill is
    used rather than "every pixel close to white": the latter also eats white
    highlights INSIDE the sprite - teeth, eyes, armour specular - and leaves
    holes that are invisible at 512px and obvious at 48px.
    """
    rgba = np.asarray(img.convert("RGBA")).copy()
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.int16)

    visited = np.zeros((h, w), dtype=bool)
    stack = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    seeds = [rgb[y, x] for y, x in stack]

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        if not any(np.abs(rgb[y, x] - s).max() <= tolerance for s in seeds):
            continue
        visited[y, x] = True
        stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

    rgba[visited, 3] = 0

    # Second pass: ENCLOSED background, which the flood fill structurally cannot
    # reach.
    #
    # A walking sprite traps a wedge of background between its legs; a raised
    # arm traps another under the armpit. Neither touches the border, so a
    # corner flood leaves them opaque and they show up as a solid black slab in
    # the middle of the sprite - measured on frame 3 of the walk cycle.
    #
    # Clearing every near-background pixel instead would be simpler and wrong:
    # this character's outline and its darkest shading sit close to black, and a
    # global threshold punches holes through them. So only whole CONNECTED
    # REGIONS above a size floor are removed - a trapped wedge is large and
    # contiguous, shading is small and scattered.
    if min_enclosed > 0:
        try:
            from scipy import ndimage
        except ImportError:
            return Image.fromarray(rgba, mode="RGBA")

        near_bg = np.zeros((h, w), dtype=bool)
        for s in seeds:
            near_bg |= (np.abs(rgb - s).max(axis=2) <= tolerance)
        enclosed = near_bg & ~visited
        labels, count = ndimage.label(enclosed)
        if count:
            sizes = ndimage.sum(enclosed, labels, range(1, count + 1))
            for idx, size in enumerate(sizes, start=1):
                if size >= min_enclosed:
                    rgba[labels == idx, 3] = 0

    return Image.fromarray(rgba, mode="RGBA")


# --- the pixelation itself ----------------------------------------------

def _alpha_bbox(img: Image.Image, alpha_threshold: int):
    """Bounding box of pixels at or above the alpha threshold, or None."""
    alpha = np.asarray(img.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha >= alpha_threshold)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pixelate(img: Image.Image,
             cell_w: int,
             cell_h: int,
             palette: np.ndarray | None = None,
             n_colors: int = DEFAULT_COLORS,
             alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
             scale: float | None = None,
             margin: int = 1) -> Image.Image:
    """Render one sprite into an exactly `cell_w` x `cell_h` RGBA pixel-art cell.

    `scale` pins the downsample factor. Pass the value `sheet_scale()` returned
    so that every cell in a sheet is reduced identically; leave it None to fit
    this sprite alone, which is only correct for a single standalone asset.
    """
    src = img.convert("RGBA")
    box = _alpha_bbox(src, alpha_threshold)
    if box is None:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    src = src.crop(box)

    if palette is None:
        palette = extract_palette(src, n_colors, alpha_threshold)

    if scale is None:
        scale = min((cell_w - 2 * margin) / src.width,
                    (cell_h - 2 * margin) / src.height)

    tw = max(1, int(round(src.width * scale)))
    th = max(1, int(round(src.height * scale)))

    # BOX (area average) rather than LANCZOS or BICUBIC. The sharpening filters
    # ring at high-contrast edges, and at 48px a ring is a stray pixel of a
    # colour that is in neither the character nor the background.
    small = src.resize((tw, th), Image.Resampling.BOX)

    arr = np.asarray(small).copy()
    opaque = arr[..., 3] >= alpha_threshold

    arr[..., 3] = np.where(opaque, 255, 0)
    # Zero the colour under transparent pixels too. Left alone it is the
    # averaged edge colour, and some engines and atlas packers bleed it back in
    # when sampling with filtering enabled.
    arr[~opaque, :3] = 0
    arr[opaque, :3] = snap_to_palette(arr[..., :3], palette)[opaque]

    small = Image.fromarray(arr, mode="RGBA")

    # Re-crop AFTER thresholding, not before.
    #
    # The bbox was measured on the full-resolution source. Downscaling averages
    # a faint bottom row - a shadow, a toe, the soft edge of a boot - down below
    # the alpha threshold, which then erases it. Aligning the pre-threshold
    # rectangle therefore lands the character 1-2px high in some cells and not
    # others, and a baseline that moves by a pixel between frames reads as the
    # sprite bobbing during playback. Measure what actually survived.
    box = _alpha_bbox(small, alpha_threshold)
    if box is None:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    small = small.crop(box)

    # Centre horizontally, anchor to the bottom vertically. See module docstring.
    cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    cell.paste(small,
               ((cell_w - small.width) // 2, cell_h - margin - small.height),
               small)
    return cell


def _slice_grid(sheet: Image.Image, cols: int, rows: int):
    """Yield (col, row, sub-image). Requires an even division, like something2."""
    if sheet.width % cols or sheet.height % rows:
        raise ValueError(
            f"sheet {sheet.width}x{sheet.height} does not divide evenly into "
            f"{cols}x{rows} - something2 rejects such a sheet outright"
        )
    cw, ch = sheet.width // cols, sheet.height // rows
    for r in range(rows):
        for c in range(cols):
            yield c, r, sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))


def pixelate_sheet(sheet: Image.Image,
                   cols: int,
                   rows: int,
                   cell_w: int,
                   cell_h: int,
                   palette: np.ndarray | None = None,
                   n_colors: int = DEFAULT_COLORS,
                   alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
                   margin: int = 1,
                   strip_ground: bool = False,
                   ground_ratio: float = 1.8) -> Image.Image:
    """Pixelate every cell of a grid sheet with a shared palette and scale.

    The shared pair is the whole point. A per-cell palette and a per-cell scale
    each independently reintroduce the drift this stage is meant to remove.
    """
    cells = list(_slice_grid(sheet, cols, rows))

    if strip_ground:
        # Per cell, BEFORE the palette is sampled. Sampling first would spend
        # entries on dirt and shadow colours that are about to be deleted, and
        # the palette is only 24 wide.
        #
        # Note this runs per DIRECTION, and the threshold bites unevenly across
        # them: front and back views present the body at its widest across the
        # shoulders, which inflates the reference width and pushes the patch
        # ratio under the cut. At 1.8 those two cells kept their patches while
        # the six turned cells lost theirs.
        cells = [(c, r, strip_ground_patch(sub, width_ratio=ground_ratio))
                 for c, r, sub in cells]

    if palette is None:
        palette = extract_palette(sheet, n_colors, alpha_threshold)

    # One scale for the sheet: the tightest fit any single cell needs.
    scale = None
    for _, _, sub in cells:
        box = _alpha_bbox(sub, alpha_threshold)
        if box is None:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        s = min((cell_w - 2 * margin) / w, (cell_h - 2 * margin) / h)
        scale = s if scale is None else min(scale, s)
    if scale is None:
        raise ValueError("no cell in the sheet has any opaque pixels")

    out = Image.new("RGBA", (cols * cell_w, rows * cell_h), (0, 0, 0, 0))
    for c, r, sub in cells:
        out.paste(
            pixelate(sub, cell_w, cell_h, palette=palette,
                     alpha_threshold=alpha_threshold, scale=scale,
                     margin=margin),
            (c * cell_w, r * cell_h),
        )
    return out


# --- CLI ----------------------------------------------------------------

def _parse_pair(text: str, what: str):
    try:
        a, b = text.lower().split("x")
        return int(a), int(b)
    except Exception:
        raise SystemExit(f"--{what} wants WxH, e.g. 4x2 (got {text!r})")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--grid", default="1x1", help="columns x rows, e.g. 4x2")
    p.add_argument("--cell", default="48x48", help="output cell size, e.g. 48x48")
    p.add_argument("--colors", type=int, default=DEFAULT_COLORS)
    p.add_argument("--alpha-threshold", type=int, default=DEFAULT_ALPHA_THRESHOLD)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--key", action="store_true",
                   help="flood-fill the background to transparent first "
                        "(for sources saved without alpha)")
    p.add_argument("--key-tolerance", type=int, default=24)
    p.add_argument("--strip-ground", action="store_true",
                   help="delete the ground/shadow patch fused under each "
                        "sprite. Qwen's angles LoRA invents one in every "
                        "direction; keying cannot remove it because it is "
                        "joined to the feet, not to the background")
    p.add_argument("--ground-ratio", type=float, default=1.8,
                   help="how much wider than the body a row must be to count "
                        "as ground (default 1.8). Lower catches the front and "
                        "back views, whose wide shoulders inflate the "
                        "reference; too low walks up into the boots")
    p.add_argument("--preview-scale", type=int, default=0,
                   help="also write <dst>.preview.png upscaled N times with "
                        "NEAREST, for inspecting a 48px sheet on a real screen")
    a = p.parse_args(argv)

    cols, rows = _parse_pair(a.grid, "grid")
    cw, ch = _parse_pair(a.cell, "cell")

    img = Image.open(a.src)
    if a.key:
        img = key_background(img, a.key_tolerance)

    out = pixelate_sheet(img, cols, rows, cw, ch,
                         n_colors=a.colors,
                         alpha_threshold=a.alpha_threshold,
                         margin=a.margin,
                         strip_ground=a.strip_ground,
                         ground_ratio=a.ground_ratio)
    out.save(a.dst)

    pal = extract_palette(out, a.colors, a.alpha_threshold)
    print(f"{a.src} {img.width}x{img.height} -> {a.dst} {out.width}x{out.height} "
          f"({cols}x{rows} cells of {cw}x{ch}, {len(pal)} colours)")

    if a.preview_scale > 1:
        # NEAREST, so the preview shows the actual pixels rather than a
        # smoothed guess at them. Any other filter defeats the purpose.
        dst = f"{a.dst}.preview.png"
        out.resize((out.width * a.preview_scale, out.height * a.preview_scale),
                   Image.Resampling.NEAREST).save(dst)
        print(f"preview -> {dst} ({a.preview_scale}x NEAREST)")


if __name__ == "__main__":
    main()
