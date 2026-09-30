from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

HERE = Path(__file__).resolve().parent
V7_PATH = HERE / "depth_sdf_reconstruct_v7.py"

spec = importlib.util.spec_from_file_location("depth_v7_base", V7_PATH)
v7 = importlib.util.module_from_spec(spec)
sys.modules["depth_v7_base"] = v7
assert spec and spec.loader
spec.loader.exec_module(v7)


def fill_sparse_depth(zbuf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(zbuf) & mask
    if not valid.any():
        raise RuntimeError("empty base depth projection")
    inds = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = zbuf[tuple(inds)].astype(np.float32)
    return filled * mask.astype(np.float32)


def anchored_calibrate_depth(mesh, ds, depth: np.ndarray, mask: np.ndarray, view: str, diag: Path):
    rel_depth = v7.v3.normalize_inverse_depth(depth, mask)
    calib = v7.generic_camera_calibration(mesh, ds, mask, view)
    zbuf = v7.sample_surface_zbuffer(mesh, calib, mask.shape)
    valid = np.isfinite(zbuf) & mask
    valid &= ndimage.binary_erosion(mask, iterations=3)
    if int(valid.sum()) < 500:
        valid = np.isfinite(zbuf) & mask

    a, b, rmse, inlier_ratio, inliers = v7.robust_affine_fit(rel_depth[valid], zbuf[valid])
    affine_metric = (a * rel_depth + b).astype(np.float32)
    base_metric = fill_sparse_depth(zbuf, mask)

    low = ndimage.gaussian_filter(rel_depth, sigma=10.0)
    high = (rel_depth - low) * mask.astype(np.float32)
    hp = np.abs(high[mask])
    hp95 = float(np.percentile(hp, 95)) if hp.size else 1.0
    maxdim = max(ds.width, ds.depth, ds.height)
    detail_scale = (0.024 * maxdim) / max(hp95, 1e-6)
    detail = np.clip(high * detail_scale, -0.032 * maxdim, 0.032 * maxdim)
    anchored = (base_metric + detail).astype(np.float32)

    fit_rel = math.exp(-((rmse / max(0.032 * maxdim, 1e-6)) ** 2))
    raw_rel = float(np.clip(fit_rel * (0.45 + 0.55 * inlier_ratio), 0.0, 1.0))

    if view in v7.ORTHO_VIEWS:
        reliability = max(raw_rel, 0.28)
    elif view == "iso_right_bottom":
        reliability = max(raw_rel, 0.07)
    else:
        reliability = min(raw_rel, 0.02)

    affine_mix = float(np.clip((raw_rel - 0.18) / 0.55, 0.0, 0.55))
    metric = ((1.0 - affine_mix) * anchored + affine_mix * affine_metric).astype(np.float32)
    metric = np.clip(
        metric,
        calib["depth_min"] - 0.045 * maxdim,
        calib["depth_max"] + 0.045 * maxdim,
    ) * mask.astype(np.float32)

    vis = np.zeros_like(metric, dtype=np.float32)
    vals = metric[mask]
    if vals.size:
        lo, hi = np.percentile(vals, [2, 98])
        vis[mask] = np.clip((vals - lo) / max(float(hi - lo), 1e-6), 0, 1)
    Image.fromarray((vis * 255).astype(np.uint8)).save(diag / f"{view}_metric_depth_v7_refined.png")

    return metric, rel_depth, calib, {
        "a": float(a),
        "b": float(b),
        "rmse": float(rmse),
        "inlier_ratio": float(inlier_ratio),
        "inliers": int(inliers),
        "raw_affine_reliability": raw_rel,
        "reliability": float(reliability),
        "affine_mix": affine_mix,
        "detail_scale": float(detail_scale),
        "mode": "base-anchored high-pass learned detail",
        "ppu": float(calib["ppu"]),
        "cx": float(calib["cx"]),
        "cy": float(calib["cy"]),
    }


_original_poisson = v7.poisson_reconstruct


def repaired_poisson(points, normals, base_mesh, ds, diag: Path, depth: int = 9):
    mesh = _original_poisson(points, normals, base_mesh, ds, diag, depth=depth)
    try:
        from pymeshfix import MeshFix
        fixer = MeshFix(mesh.vertices, mesh.faces)
        fixer.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        repaired = trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=True)
        comps = repaired.split(only_watertight=False)
        if len(comps) > 1:
            repaired = max(comps, key=lambda m: len(m.faces))
        repaired.fill_holes()
        repaired.remove_unreferenced_vertices()
        if repaired.volume < 0:
            repaired.invert()
        repaired.export(diag / "poisson_meshfix_repaired_v7.stl")
        print(f"[v7 refined] meshfix watertight={repaired.is_watertight} faces={len(repaired.faces)}")
        return repaired
    except Exception as exc:
        print(f"[v7 refined] meshfix failed, using Poisson mesh: {exc}")
        return mesh


v7.calibrate_depth_to_mesh = anchored_calibrate_depth
v7.poisson_reconstruct = repaired_poisson


if __name__ == "__main__":
    v7.main()
