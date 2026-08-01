"""Offscreen renders of a GLB — a quick look without opening an engine.

Runs under the worker interpreter (HY3D_PY); needs trimesh + numpy, and
pyrender for the actual raster. Prints one JSON line on stdout:
{"png_paths": [...]} on success, {"error": ...} otherwise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

VIEWS = {
    "iso": (1.0, 0.6, 1.0),
    "front": (0.0, 0.05, 1.0),
    "back": (0.0, 0.05, -1.0),
    "top": (0.0, 1.0, 0.001),
    "side": (1.0, 0.05, 0.0),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--views", default="iso",
                    help="comma-separated subset of %s" % ",".join(VIEWS))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    try:
        import numpy as np
        import trimesh
        import pyrender
    except ImportError as e:
        # uv, not `python -m pip`: uv-created venvs ship without pip. Name the
        # real interpreter too, so the command is runnable as printed.
        print(json.dumps({
            "error": "missing %r in the worker venv — install it with: "
                     "uv pip install --python %s %s"
                     % (e.name, sys.executable, e.name),
        }))
        sys.exit(2)

    names = [v.strip() for v in args.views.split(",") if v.strip()]
    bad = [v for v in names if v not in VIEWS]
    if bad:
        print(json.dumps({"error": "unknown views %s (valid: %s)"
                          % (bad, sorted(VIEWS))}))
        sys.exit(2)

    tm = trimesh.load(args.input, force="scene")
    bounds = tm.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extent = float(np.linalg.norm(bounds[1] - bounds[0]))

    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0],
                           ambient_light=[0.25, 0.25, 0.28])
    for geom in tm.geometry.values():
        scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False))

    cam = pyrender.PerspectiveCamera(yfov=math.radians(40.0))
    key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5)
    renderer = pyrender.OffscreenRenderer(args.size, args.size)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    def look_at(direction):
        eye = center + np.asarray(direction) / np.linalg.norm(direction) * extent * 1.4
        fwd = center - eye
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(fwd, up))) > 0.99:
            up = np.array([0.0, 0.0, -1.0])
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -fwd
        pose[:3, 3] = eye
        return pose

    from PIL import Image
    paths = []
    for name in names:
        pose = look_at(VIEWS[name])
        cam_node = scene.add(cam, pose=pose)
        # Key light rides the camera so every view is lit from where it looks.
        key_node = scene.add(key, pose=pose)
        color, _ = renderer.render(scene)
        out = outdir / ("%s-%s.png" % (stem, name))
        Image.fromarray(color).save(out)
        paths.append(str(out))
        scene.remove_node(cam_node)
        scene.remove_node(key_node)
    renderer.delete()

    print(json.dumps({"png_paths": paths}))


if __name__ == "__main__":
    main()
