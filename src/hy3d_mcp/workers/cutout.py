"""Key a concept image out to a centered square RGBA PNG.

Runs under the worker interpreter (HY3D_PY), which must have numpy + PIL
(scipy optional, used for interior hole-fill and component filtering;
rembg for the fallback key). Prints one JSON line on stdout as its result
contract: {"png_path", "opaque_pct", "method"} on success, {"error": ...}
on refusal.

Two keys, tried in order:

1. **Corner sampling.** Free, exact, and correct for a single object on a
   plain background — the case the pipeline is documented for. It refuses
   busy inputs rather than shredding them.
2. **rembg / u2net.** A segmentation model, so it handles the concept art
   people actually have: painted scenes, textured grounds, cast shadows.
   Slower, needs a 176MB weight file, and keeps stray objects the corner
   key would never have reached — hence the largest-component filter.

Both keys then run that component filter, at different thresholds; see
``largest_component`` for why the corner key needs one at all.

Downstream rationale: Hunyuan3D (and TRELLIS before it) skips its gated
background-removal model whenever the input already carries real
transparency, so keying here removes the only gated-weights dependency in
the pipeline — and, unlike the engine's own pass, frames the result.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# RGB-space distances from the sampled background color that map to alpha 0
# and alpha 255. The soft ramp keeps anti-aliased edges clean.
D_LO, D_HI = 12.0, 40.0
CORNER = 24
MARGIN = 32
# Per-channel corner-patch std above this means the corners disagree — the
# input isn't a plain-background concept and keying it would shred it.
CORNER_SPREAD_MAX = 28.0
# Below this the key has eaten the subject rather than the background.
OPAQUE_MIN_PCT = 0.5
# Corner-key islands smaller than this share of the subject are frame
# furniture, not part of it. Measured across four multiview strips: the
# border rules ran 0.7-1.4%, and everything else the key found was single
# pixels. 5% leaves 3.5x headroom over the largest observed stray while
# staying well under any real detached part.
CORNER_KEEP_PCT = 5.0


def emit(payload: dict) -> None:
    print(json.dumps(payload))


def key_corners(a: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Alpha from a plain-background key, or (None, why it refused)."""
    c = CORNER
    corners = np.concatenate([
        a[:c, :c].reshape(-1, 3), a[:c, -c:].reshape(-1, 3),
        a[-c:, :c].reshape(-1, 3), a[-c:, -c:].reshape(-1, 3),
    ])
    spread = float(np.std(corners, axis=0).mean())
    if spread > CORNER_SPREAD_MAX:
        return None, ("corner patches disagree (per-channel std %.1f > %.1f) — "
                      "this doesn't look like a single object on a plain "
                      "background" % (spread, CORNER_SPREAD_MAX))
    bg = np.median(corners, axis=0)

    dist = np.linalg.norm(a - bg, axis=-1)
    alpha = np.clip((dist - D_LO) / (D_HI - D_LO), 0.0, 1.0)

    # Fill interior: pixels well inside the silhouette should be opaque even
    # if their color happens to sit near the background.
    try:
        from scipy.ndimage import binary_fill_holes
        solid = binary_fill_holes(alpha > 0.6)
        alpha = np.maximum(alpha, solid.astype(np.float32))
    except ImportError:
        pass

    opaque_pct = 100.0 * float((alpha > 0.5).mean())
    if opaque_pct < OPAQUE_MIN_PCT:
        return None, ("key found almost nothing opaque (%.2f%%) — the "
                      "background sample probably matched the subject"
                      % opaque_pct)
    return alpha, None


def key_rembg(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Alpha and RGB from u2net segmentation. Raises RuntimeError if unusable.

    rembg composites nothing and returns the source pixels with an alpha
    channel, so the RGB it hands back is the input's own — worth taking
    rather than re-reading the file, since rembg also normalises exotic
    input modes on the way through.
    """
    try:
        from rembg import new_session, remove
    except ImportError as e:
        raise RuntimeError(
            "the corner key refused this image and rembg is not available to "
            "fall back on (%r missing from %s) — install it with: uv pip "
            "install --python %s rembg onnxruntime"
            % (getattr(e, "name", "rembg"), sys.executable, sys.executable))
    with Image.open(path) as im:
        # u2net weights land in ~/.u2net on first use: a silent 176MB
        # download, which is why the installer prefetches them.
        cut = remove(im.convert("RGB"), session=new_session("u2net"))
    cut = cut.convert("RGBA")
    arr = np.asarray(cut).astype(np.float32)
    return arr[..., 3] / 255.0, arr[..., :3]


def largest_component(alpha: np.ndarray,
                      keep_pct: float | None = None) -> tuple[np.ndarray, int]:
    """Drop stray islands. Returns (alpha, blobs dropped).

    u2net segments *subjects*, plural: on garden concept art it keeps the
    lantern and also a loose rock and part of a cast shadow. Each arrives as
    its own island, and every island becomes geometry — a rock floating
    beside the model.

    The corner key needs this too, which the earlier docstring here denied
    ("it cannot reach past the background it sampled"). It can, whenever
    something in the frame is a different colour from the corner it sampled
    and survives the key on its own. The case that proved it: a multiview
    sheet cut into strips, where three of four strips carried a 1-2px
    full-height border rule. The rule keyed as its own island, became a
    vertical slab standing behind the figure, and fused into the mesh — by
    which point no mesh-side filter can separate it, because it is no
    longer a separate component.

    Which islands to drop differs by path, so the threshold does too.
    ``keep_pct=None`` keeps only the biggest blob: that is the rembg
    contract, where a second *subject* is exactly what should go. A float
    keeps every blob at least that percentage of the biggest, which is what
    the corner path wants — its strays are frame furniture, measured at
    0.7-1.4% of the subject across the four views above, while a genuinely
    detached second part of a subject is far larger. Erring toward keeping
    is right here: a kept rule is visible in the preview, a dropped arm is
    not.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        return alpha, 0
    lab, n = label(alpha > 0.5)
    if n <= 1:
        return alpha, 0
    # Bin 0 is background, so the subject is the largest of bins 1..n.
    sizes = np.bincount(lab.ravel())[1:]
    biggest = int(sizes.max())
    if keep_pct is None:
        keep = np.array([int(np.argmax(sizes)) + 1])
    else:
        keep = np.where(sizes >= biggest * keep_pct / 100.0)[0] + 1
    mask = np.isin(lab, keep)
    return np.where(mask, alpha, 0.0).astype(np.float32), n - len(keep)


def frame_square(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Crop to the alpha bbox with margin, then pad square.

    The generator frames its latent around the image, so a centered square
    subject uses the resolution instead of wasting it on empty background.
    This is what the engine's own rembg pass does not do.
    """
    rgba = np.dstack([rgb, alpha[..., None] * 255.0]).astype(np.uint8)
    out = Image.fromarray(rgba, "RGBA")
    ys, xs = np.where(alpha > 0.02)
    m = MARGIN
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, rgb.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, rgb.shape[1])
    out = out.crop((int(x0), int(y0), int(x1), int(y1)))
    side = max(out.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(out, ((side - out.width) // 2, (side - out.height) // 2))
    return sq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--method", choices=["auto", "corner", "rembg"],
                    default="auto",
                    help="auto tries the corner key and falls back to rembg")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    rgb = np.asarray(img).astype(np.float32)

    alpha = None
    refusal = None
    method = args.method
    dropped = 0
    if args.method in ("auto", "corner"):
        alpha, refusal = key_corners(rgb)
        method = "corner"
        if alpha is not None:
            # See largest_component: the corner key does reach past its own
            # background, and a border rule it keeps becomes a slab fused
            # into the mesh. Generous threshold — this is only meant to
            # catch frame furniture, not to pick between subjects.
            alpha, dropped = largest_component(alpha, keep_pct=CORNER_KEEP_PCT)
            if dropped:
                method = "corner+component-filter"

    if alpha is None and args.method in ("auto", "rembg"):
        try:
            alpha, rgb = key_rembg(args.input)
        except RuntimeError as e:
            emit({"error": "%s%s" % (refusal + "; " if refusal else "", e)})
            sys.exit(2)
        alpha, dropped = largest_component(alpha)
        method = "rembg+largest-component"

    if alpha is None:
        # --method corner, and it refused. Say so plainly: the caller asked
        # for exactly this key and is entitled to the reason, not a silent
        # substitution.
        emit({"error": "%s — crop or regenerate the concept, or re-run with "
                       "method=auto to fall back to rembg" % refusal})
        sys.exit(2)

    # Full-frame, and only to catch an empty key: frame_square would raise on
    # a bbox of nothing.
    raw_pct = 100.0 * float((alpha > 0.5).mean())
    if raw_pct < OPAQUE_MIN_PCT:
        emit({"error": "%s key found almost nothing opaque (%.2f%%) — nothing "
                       "usable to generate from" % (method, raw_pct)})
        sys.exit(2)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sq = frame_square(rgb, alpha)
    sq.save(dst)

    # Reported after the recrop, because that is the image the generator sees.
    # Pre-crop the number measures composition rather than the key: a subject
    # keyed perfectly but occupying a corner of a wide scene reads as 5%, and
    # the crop has already fixed exactly that.
    opaque_pct = 100.0 * float((np.asarray(sq)[..., 3] > 127).mean())

    payload = {"png_path": str(dst), "opaque_pct": round(opaque_pct, 1),
               "method": method}
    if refusal and method != "corner":
        payload["note"] = "corner key declined (%s), used rembg" % refusal
    if dropped:
        payload["components_dropped"] = dropped
    emit(payload)


if __name__ == "__main__":
    main()
