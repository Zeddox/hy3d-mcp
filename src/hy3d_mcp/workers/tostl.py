"""Convert a generated GLB into an STL a slicer will accept.

Runs under the worker interpreter (HY3D_PY), which must have numpy +
trimesh. Prints one JSON line on stdout as its result contract.

Two conversions matter and neither is optional:

* **Units.** The engine emits a mesh normalized to roughly a unit box. STL
  carries no units and every slicer reads it as millimetres, so an
  unscaled export arrives as a 2mm trinket. Scale to a real height.
* **Up axis.** glTF is Y-up; slicers are Z-up. Skipping this lands the
  model on its side on the build plate.

Also drops the mesh onto z=0 so it sits on the plate rather than floating,
and reports the manifold checks that decide whether the file slices at all.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("glb")
    ap.add_argument("output")
    ap.add_argument("--height", type=float, default=120.0,
                    help="target height in mm along the print Z axis")
    ap.add_argument("--min-wall", type=float, default=0.8,
                    help="warn when scaled features fall under this thickness "
                         "in mm (roughly two perimeters of a 0.4mm nozzle)")
    args = ap.parse_args()

    mesh = trimesh.load(args.glb, force="mesh")
    checks = {
        "watertight": bool(mesh.is_watertight),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "enclosed_volume": bool(mesh.is_volume),
        "single_body": bool(mesh.body_count == 1),
    }

    # glTF Y-up -> slicer Z-up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_scale(args.height / mesh.extents[2])
    # Sit on the build plate, then centre X/Y now that Z is pinned.
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
    c = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-c[0], -c[1], 0.0])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    x, y, z = (float(v) for v in mesh.extents)
    # Enclosed volume against the bounding box is the one number that
    # separates a solid from a thin hollow shell with the same silhouette.
    # Renders cannot show the difference; this can.
    fill_pct = 100.0 * float(mesh.volume) / float(np.prod(mesh.extents))
    genus = int((2 - mesh.euler_number) // 2)

    result = {
        "stl_path": str(out),
        "bytes": out.stat().st_size,
        "size_mm": [round(x, 1), round(y, 1), round(z, 1)],
        "volume_cm3": round(float(mesh.volume) / 1000.0, 1),
        "bbox_fill_pct": round(fill_pct, 1),
        "genus": genus,
        "checks": checks,
        "printable": all(checks.values()),
    }

    warnings = []
    if not all(checks.values()):
        warnings.append("failed %s — repair before slicing"
                        % ", ".join(k for k, v in checks.items() if not v))
    # The shortest edges are a cheap proxy for the finest detail present. This
    # does not measure wall thickness, but a model whose detail is already
    # sub-nozzle at this scale will lose it in the slice.
    p1 = float(np.percentile(mesh.edges_unique_length, 1))
    result["finest_detail_mm"] = round(p1, 2)
    if p1 < args.min_wall:
        warnings.append("finest detail ~%.2fmm is under the %.1fmm min-wall "
                        "guide; scale up or expect it to be dropped"
                        % (p1, args.min_wall))
    if genus:
        warnings.append("genus %d — the surface has %d tunnel(s) through it; "
                        "slices fine, but check it is intentional" % (genus, genus))
    if fill_pct < 15.0:
        warnings.append("encloses only %.1f%% of its bounding box — this is a "
                        "thin hollow shell, and its walls may be under the "
                        "nozzle minimum even though the silhouette looks right"
                        % fill_pct)
    if warnings:
        result["warnings"] = warnings

    print(json.dumps(result))


if __name__ == "__main__":
    main()
