"""Render turntable previews of a GLB so shape quality can be judged by eye.

Headless via EGL. Untextured meshes are rendered with a neutral material and
normal-driven shading, which is the point: it shows silhouette, surface
detail and fused geometry rather than hiding them behind concept-art colour.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
from PIL import Image


def render(path, views=4, size=512, distance=2.0, height=0.85):
    import pyrender
    mesh = trimesh.load(str(path), force="mesh")
    # Normalize into a unit box so every model frames identically -- otherwise
    # a size difference reads as a quality difference.
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))

    scene = pyrender.Scene(bg_color=[0.10, 0.10, 0.12, 1.0],
                           ambient_light=[0.25, 0.25, 0.28])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False,
              material=pyrender.MetallicRoughnessMaterial(
                  baseColorFactor=[0.78, 0.76, 0.72, 1.0],
                  metallicFactor=0.0, roughnessFactor=0.75)))

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.5)
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
    r = pyrender.OffscreenRenderer(size, size)

    frames = []
    for i in range(views):
        th = 2 * np.pi * i / views
        eye = np.array([distance * np.sin(th), height, distance * np.cos(th)])
        # Standard lookAt. Building the basis as (right, up, -forward) from a
        # cross with world-up gives a determinant of -1 -- a reflection, not a
        # rotation -- which renders view 0 by luck and blanks the rest.
        z = eye / np.linalg.norm(eye)              # camera looks down -Z
        x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
        nc = scene.add(cam, pose=pose)
        nk = scene.add(key, pose=pose)
        lp = pose.copy()
        lp[:3, 3] = eye + x * 2.0 - np.array([0.0, 1.5, 0.0])
        nf = scene.add(fill, pose=lp)
        color, _ = r.render(scene)
        frames.append(color)
        scene.remove_node(nc); scene.remove_node(nk); scene.remove_node(nf)
    r.delete()
    return np.hstack(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("glb", nargs="+")
    ap.add_argument("-o", "--output", default="preview.png")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--distance", type=float, default=2.0,
                    help="camera distance in unit-box radii; lower crops in "
                         "close, which is how you judge surface relief")
    ap.add_argument("--height", type=float, default=0.85)
    args = ap.parse_args()

    rows = [render(g, args.views, args.size, args.distance, args.height)
            for g in args.glb]
    Image.fromarray(np.vstack(rows)).save(args.output)
    print(f"{args.output}  ({len(rows)} row(s) x {args.views} views)")


if __name__ == "__main__":
    main()
