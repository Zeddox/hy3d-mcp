"""Game-look texture pass for a generated GLB. Geometry untouched.

Runs under the worker interpreter (HY3D_PY), which must have cv2, numpy,
trimesh and PIL. Prints one JSON line on stdout:
{"glb_path", "accent_coverage_pct"} on success, {"error": ...} otherwise.

Two texture operations, proven on the AEGIS fleet:
1. Tone the albedo: midtone-darkening gamma, contrast around mid-grey,
   saturation lift so accents read.
2. Extract saturated accents (redness = R - max(G,B)) plus blackhat panel
   seams into a dedicated glTF emissive texture. Accents then glow at full
   strength in-engine while the hull carries no emission floor — the flat
   albedo-as-emission trick washes the whole hull out.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

PINSTRIPE = np.array([0.95, 0.10, 0.08], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--tone-gamma", type=float, default=1.35)
    ap.add_argument("--contrast", type=float, default=1.30)
    ap.add_argument("--saturation", type=float, default=1.35)
    ap.add_argument("--no-accent-emissive", action="store_true")
    ap.add_argument("--seam-pinstripes", type=float, default=0.58,
                    help="strength 0-1; 0 disables seam extraction")
    ap.add_argument("--seam-halo", type=float, default=0.10)
    args = ap.parse_args()

    scene = trimesh.load(args.input, force="scene")
    if not scene.geometry:
        print(json.dumps({"error": "no geometry in %s" % args.input}))
        sys.exit(2)
    mesh = max(scene.geometry.values(), key=lambda g: len(g.faces))
    mat = mesh.visual.material
    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if tex is None:
        print(json.dumps({"error": "no albedo texture found — was the model "
                                   "generated with paint enabled?"}))
        sys.exit(2)

    rgb = np.asarray(tex.convert("RGB")).astype(np.float32) / 255.0

    # Accent mask from the ORIGINAL colors, before toning shifts them.
    emissive = np.zeros_like(rgb)
    mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    if not args.no_accent_emissive:
        redness = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])
        mask = np.clip((redness - 0.06) * 5.0, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
        emissive = rgb * mask[..., None]

    # Seam pinstriping: the bake paints panel seams as thin dark lines; a
    # blackhat lifts exactly those, joining the emissive map as tracery that
    # draws the hull's shape on a dark screen.
    if args.seam_pinstripes > 0.0:
        gray = rgb.mean(axis=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        seams = np.clip((blackhat - 0.05) * 6.0, 0.0, 1.0)
        seams = cv2.GaussianBlur(seams, (0, 0), 0.8)
        emissive = np.clip(
            emissive + PINSTRIPE * (seams * args.seam_pinstripes)[..., None],
            0.0, 1.0)
        # A wide blur of the seams joins the emissive faintly — soft bloom
        # around each line sells self-illumination without paying for GI.
        if args.seam_halo > 0.0:
            halo = cv2.GaussianBlur(seams, (0, 0), 6.0)
            emissive = np.clip(
                emissive + PINSTRIPE * (halo * args.seam_halo)[..., None],
                0.0, 1.0)
    emissive8 = (emissive * 255.0).astype(np.uint8)

    # Tone: darken midtones, push contrast around mid-grey, lift saturation.
    toned = np.power(rgb, args.tone_gamma)
    toned = np.clip((toned - 0.42) * args.contrast + 0.34, 0.0, 1.0)
    luma = toned.mean(axis=-1, keepdims=True)
    toned = np.clip(luma + (toned - luma) * args.saturation, 0.0, 1.0)
    albedo8 = (toned * 255.0).astype(np.uint8)

    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(albedo8, "RGB"),
            emissiveTexture=Image.fromarray(emissive8, "RGB"),
            emissiveFactor=[1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))
    print(json.dumps({
        "glb_path": str(dst),
        "accent_coverage_pct": round(100.0 * float((mask > 0.5).mean()), 1),
    }))


if __name__ == "__main__":
    main()
