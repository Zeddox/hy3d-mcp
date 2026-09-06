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


def largest_component(alpha: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only the biggest connected blob. Returns (alpha, blobs dropped).

    u2net segments *subjects*, plural: on garden concept art it keeps the
    lantern and also a loose rock and part of a cast shadow. Each arrives as
    its own island, and every island becomes geometry — a rock floating
    beside the model. The corner key never needs this; it cannot reach past
    the background it sampled.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        return alpha, 0
    lab, n = label(alpha > 0.5)
    if n <= 1:
        return alpha, 0
    # Bin 0 is background, so the subject is the largest of bins 1..n.
    sizes = np.bincount(lab.ravel())
    keep = int(np.argmax(sizes[1:])) + 1
    return np.where(lab == keep, alpha, 0.0).astype(np.float32), n - 1


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
    if args.method in ("auto", "corner"):
        alpha, refusal = key_corners(rgb)
        method = "corner"

    dropped = 0
    if alpha is None and args.method in ("auto", "rembg"):
        if args.method == "corner":
            pass
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

    opaque_pct = 100.0 * float((alpha > 0.5).mean())
    if opaque_pct < OPAQUE_MIN_PCT:
        emit({"error": "%s key found almost nothing opaque (%.2f%%) — nothing "
                       "usable to generate from" % (method, opaque_pct)})
        sys.exit(2)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame_square(rgb, alpha).save(dst)

    payload = {"png_path": str(dst), "opaque_pct": round(opaque_pct, 1),
               "method": method}
    if refusal and method != "corner":
        payload["note"] = "corner key declined (%s), used rembg" % refusal
    if dropped:
        payload["components_dropped"] = dropped
    emit(payload)


if __name__ == "__main__":
    main()
