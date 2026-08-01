"""Key a plain-background concept image out to RGBA.

Runs under the worker interpreter (HY3D_PY), which must have numpy + PIL
(scipy optional, used for interior hole-fill). Prints one JSON line on
stdout as its result contract: {"png_path", "opaque_pct"} on success,
{"error": ...} on refusal.

Downstream rationale: Hunyuan3D (and TRELLIS before it) skips its gated
background-removal model whenever the input already carries real
transparency, so a clean local cutout removes the only gated-weights
dependency in the pipeline.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    a = np.asarray(img).astype(np.float32)

    # Background color = median of the four corner patches
    c = CORNER
    corners = np.concatenate([
        a[:c, :c].reshape(-1, 3), a[:c, -c:].reshape(-1, 3),
        a[-c:, :c].reshape(-1, 3), a[-c:, -c:].reshape(-1, 3),
    ])
    spread = float(np.std(corners, axis=0).mean())
    if spread > CORNER_SPREAD_MAX:
        print(json.dumps({
            "error": "corner patches disagree (per-channel std %.1f > %.1f) — "
                     "this doesn't look like a single object on a plain "
                     "background; crop or regenerate the concept instead"
                     % (spread, CORNER_SPREAD_MAX),
        }))
        sys.exit(2)
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
    if opaque_pct < 0.5:
        print(json.dumps({
            "error": "key found almost nothing opaque (%.2f%%) — background "
                     "sample probably matched the subject" % opaque_pct,
        }))
        sys.exit(2)

    rgba = np.dstack([a, alpha[..., None] * 255.0]).astype(np.uint8)
    out = Image.fromarray(rgba, "RGBA")

    # Crop to the alpha bbox with margin, then pad square: the generator
    # frames its latent around the image, so a centered square subject uses
    # the resolution instead of wasting it on empty background.
    ys, xs = np.where(alpha > 0.02)
    m = MARGIN
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, a.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, a.shape[1])
    out = out.crop((x0, y0, x1, y1))
    side = max(out.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(out, ((side - out.width) // 2, (side - out.height) // 2))
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sq.save(dst)

    print(json.dumps({"png_path": str(dst), "opaque_pct": round(opaque_pct, 1)}))


if __name__ == "__main__":
    main()
