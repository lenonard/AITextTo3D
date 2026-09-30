from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "depth_sdf_reconstruct.py"
V4_PATH = HERE / "depth_sdf_reconstruct_v4.py"
V5_PATH = HERE / "depth_sdf_reconstruct_v5.py"
V6_PATH = HERE / "depth_sdf_reconstruct_v6.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


v3 = _load_module("depth_v3_v8", V3_PATH)
v4 = _load_module("depth_v4_v8", V4_PATH)
v5 = _load_module("depth_v5_v8", V5_PATH)
v6 = _load_module("depth_v6_v8", V6_PATH)

ORTHO_VIEWS = ("front", "back", "left", "right")
ISO_VIEWS = ("iso_left_top", "iso_right_bottom")

AXIS_OUT = {
    "front": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "back": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
}


def masked_gaussian(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(arr.astype(np.float32) * m, sigma=sigma)
    den = ndimage.gaussian_filter(m, sigma=sigma)
    return (num / np.maximum(den, 1e-5)).astype(np.float32)


def detail_band(surface: np.ndarray, mask: np.ndarray, maxdim: float):
    # V7 failed because raw metric depth / Poisson fusion turned view disagreement
    # into surface noise and changed the body shape.  V8 only keeps a stable
    # band-pass component from the already aligned orthographic depth surfaces.
    s2 = masked_gaussian(surface, mask, 2.0)
    s7 = masked_gaussian(surface, mask, 7.0)
    fine = surface - s2
    medium = s2 - s7
    band = (0.18 * fine + 0.82 * medium).astype(np.float32)

    # Never create ridges along silhouette boundaries.
    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    edge_fade = np.clip(dist / 14.0, 0.0, 1.0)
    edge_fade = edge_fade * edge_fade * (3.0 - 2.0 * edge_fade)
    band *= edge_fade

    cap = 0.0080 * maxdim
    band = np.clip(band, -cap, cap)
    conf = np.sqrt(np.clip(dist / max(float(dist.max()), 1e-6), 0.0, 1.0))
    conf *= edge_fade
    return band.astype(np.float32), conf.astype(np.float32)


def project_vertices(ds, verts: np.ndarray, mask: np.ndarray, view: str):
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    if view == "front":
        return v3.world_to_pixel_front(x, z, mask, ds.height)
    if view == "back":
        return v3.world_to_pixel_front(-x, z, mask, ds.height)
    if view == "left":
        return v3.world_to_pixel_side(-y, z, mask, ds.height)
    if view == "right":
        return v3.world_to_pixel_side(y, z, mask, ds.height)
    raise ValueError(view)


def sample_detail_for_vertices(ds, verts, normals, masks, bands, confs, maxdim):
    n = len(verts)
    candidates = np.full((len(ORTHO_VIEWS), n), np.nan, dtype=np.float64)
    weights = np.zeros((len(ORTHO_VIEWS), n), dtype=np.float64)

    for vi, view in enumerate(ORTHO_VIEWS):
        px, py = project_vertices(ds, verts, masks[view], view)
        hp = v3.sample_bilinear(bands[view], px, py).astype(np.float64)
        cf = v3.sample_bilinear(confs[view], px, py).astype(np.float64)
        inside = v3.sample_bilinear(masks[view].astype(np.float32), px, py) > 0.75
        facing = np.clip(normals @ AXIS_OUT[view], 0.0, 1.0)
        w = cf * (facing ** 2.2)
        good = inside & (facing > 0.16) & (w > 0.025)
        # A change in axial surface depth moves the surface along the view's
        # outward axis. Project that change onto the vertex normal.
        candidates[vi, good] = hp[good] * facing[good]
        weights[vi, good] = w[good]

    with np.errstate(all="ignore"):
        med = np.nanmedian(candidates, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    # Reject only strong cross-view disagreement. This is the key safeguard
    # against the noisy rippled surface seen in V7.
    agree = np.isfinite(candidates) & (
        np.abs(candidates - med[None, :]) <= (0.0045 * maxdim)
    )
    weights *= agree
    vals = np.nan_to_num(candidates, nan=0.0)
    wsum = weights.sum(axis=0)
    disp = np.zeros(n, dtype=np.float64)
    ok = wsum > 1e-8
    disp[ok] = (vals[:, ok] * weights[:, ok]).sum(axis=0) / wsum[ok]

    # Confidence rises only when at least one view sees this point well.
    support = np.sum(weights > 0.025, axis=0)
    confidence = np.clip(wsum / 1.15, 0.0, 1.0)
    confidence *= np.where(support >= 2, 1.0, 0.72)
    return disp, confidence, support


def build_v6_reference(ds, args, diag: Path):
    depths = v4.cache_depths(ds, diag, args.model, Path(args.depth_cache))
    surfaces, masks = v3.make_surface_maps(ds, depths, diag)

    # Preserve copies because alignment functions mutate their inputs.
    surfaces = {k: v.copy() for k, v in surfaces.items()}
    masks = {k: v.copy() for k, v in masks.items()}
    surfaces = v5.conservative_cross_view_align(ds, surfaces, masks, diag)
    surfaces, ortho_reg = v5.conservative_point_register(ds, surfaces, masks, diag)

    sdf0, axes = v3.build_sdf(
        ds,
        surfaces,
        masks,
        resolution=tuple(args.resolution),
        trunc_fraction=0.034,
    )
    sdf, reproj = v5.boundary_only_reprojection_refine(
        ds, sdf0, masks, axes, iterations=args.refine_iterations
    )
    base_mesh = v3.mesh_from_sdf(sdf, axes)
    if base_mesh.volume < 0:
        base_mesh.invert()
    base_mesh.fill_holes()
    base_mesh.remove_unreferenced_vertices()

    # Keep the V6 diagonal-view refinement because that version had a much more
    # faithful body shape than V7. Bad isometric registration is ignored by V6's
    # reliability gate.
    iso_depths = v6.cache_iso_depths(ds, diag, args.model, Path(args.iso_depth_cache))
    ref_pts = base_mesh.vertices
    if len(ref_pts) > 45000:
        rng = np.random.default_rng(1234)
        ref_pts = ref_pts[rng.choice(len(ref_pts), 45000, replace=False)]

    maxdim = max(ds.width, ds.depth, ds.height)
    iso_data = {}
    iso_reg = {}
    for view in ISO_VIEWS:
        raw_pts, conf, calib = v6.build_iso_cloud(ds, base_mesh, iso_depths[view], view, stride=3)
        reg_pts, reg = v6.conservative_trimmed_icp(raw_pts, ref_pts, maxdim, iterations=5)
        reliability = v6.registration_reliability(reg, maxdim)
        iso_data[view] = {
            "points": reg_pts,
            "confidence": conf,
            "camera": v6.normalize(v6.ISO_CAMERA_DIRS[view]),
            "reliability": reliability,
        }
        iso_reg[view] = {
            "reliability": float(reliability),
            "registration": reg,
            "ppu": float(calib["ppu"]),
        }

    refined, iso_stats = v6.refine_mesh_from_iso(
        base_mesh, iso_data, maxdim, iterations=args.iso_refine_iterations
    )
    refined.fill_holes()
    refined.remove_unreferenced_vertices()
    if refined.volume < 0:
        refined.invert()
    refined.export(diag / "v6_reference_shape.stl")
    return refined, surfaces, masks, {
        "ortho_registration": ortho_reg,
        "v5_reprojection": reproj,
        "iso_registration": iso_reg,
        "iso_refine_stats": iso_stats,
    }


def conservative_detail_refine(mesh, ds, surfaces, masks, diag: Path, iterations: int = 2):
    refined = mesh.copy()
    base_vertices = refined.vertices.copy()
    base_normals = refined.vertex_normals.copy()
    maxdim = max(ds.width, ds.depth, ds.height)

    bands = {}
    confs = {}
    for view in ORTHO_VIEWS:
        bands[view], confs[view] = detail_band(surfaces[view], masks[view], maxdim)
        hp = bands[view]
        vis = np.zeros_like(hp, dtype=np.float32)
        vals = hp[masks[view]]
        scale = max(float(np.percentile(np.abs(vals), 98)), 1e-6) if vals.size else 1.0
        vis[masks[view]] = np.clip(0.5 + 0.5 * hp[masks[view]] / scale, 0.0, 1.0)
        Image.fromarray((vis * 255).astype(np.uint8)).save(diag / f"{view}_detail_band_v8.png")

    history = []
    for it in range(iterations):
        verts = refined.vertices.copy()
        normals = refined.vertex_normals.copy()
        disp, conf, support = sample_detail_for_vertices(
            ds, verts, normals, masks, bands, confs, maxdim
        )
        cap_iter = 0.0048 * maxdim
        disp = np.clip(disp, -cap_iter, cap_iter)
        step = (0.46 if it == 0 else 0.28) * conf * disp
        refined.vertices = verts + step[:, None] * normals

        # Very light fairing only; unlike V7 this never reconstructs a noisy
        # point cloud surface from scratch.
        trimesh.smoothing.filter_taubin(
            refined, lamb=0.075, nu=-0.076, iterations=1
        )

        # Hard shape lock: after smoothing, retain ONLY a small normal offset
        # from the V6 reference. No tangential drift is allowed.
        delta = refined.vertices - base_vertices
        scalar = np.sum(delta * base_normals, axis=1)
        total_cap = 0.0095 * maxdim
        scalar = np.clip(scalar, -total_cap, total_cap)
        refined.vertices = base_vertices + scalar[:, None] * base_normals

        history.append({
            "iteration": int(it + 1),
            "supported_vertices": int(np.sum(conf > 0.05)),
            "two_view_supported": int(np.sum(support >= 2)),
            "mean_abs_step": float(np.mean(np.abs(step))),
            "p95_abs_step": float(np.percentile(np.abs(step), 95)),
            "max_abs_total_offset": float(np.max(np.abs(scalar))),
        })

    refined.fill_holes()
    refined.remove_unreferenced_vertices()
    if refined.volume < 0:
        refined.invert()
    return refined, history


def shape_guard(base_mesh, candidate, ds, masks):
    base_iou = {v: v6.silhouette_iou_mesh(base_mesh, ds, masks[v], v) for v in ORTHO_VIEWS}
    cand_iou = {v: v6.silhouette_iou_mesh(candidate, ds, masks[v], v) for v in ORTHO_VIEWS}
    base_ext = base_mesh.bounds[1] - base_mesh.bounds[0]
    cand_ext = candidate.bounds[1] - candidate.bounds[0]
    ratio = cand_ext / np.maximum(base_ext, 1e-8)

    # Detail is allowed to change silhouettes only fractionally.
    bad_iou = any(cand_iou[v] < base_iou[v] - 0.010 for v in ORTHO_VIEWS)
    bad_bbox = bool(np.any(ratio < 0.975) or np.any(ratio > 1.025))
    return (not bad_iou and not bad_bbox), {
        "base_iou": base_iou,
        "candidate_iou": cand_iou,
        "bbox_ratio": [float(x) for x in ratio],
        "bad_iou": bool(bad_iou),
        "bad_bbox": bool(bad_bbox),
    }


def build(args):
    ds = v3.load_dataset(args.dataset_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics)
    diag.mkdir(parents=True, exist_ok=True)

    base_mesh, surfaces, masks, base_info = build_v6_reference(ds, args, diag)
    candidate, detail_history = conservative_detail_refine(
        base_mesh,
        ds,
        surfaces,
        masks,
        diag,
        iterations=args.detail_iterations,
    )

    ok, guard = shape_guard(base_mesh, candidate, ds, masks)
    final_mesh = candidate
    fallback = False
    if not ok:
        # Retry at half detail amplitude before falling back completely.
        delta = candidate.vertices - base_mesh.vertices
        half = base_mesh.copy()
        half.vertices = base_mesh.vertices + 0.5 * delta
        half.fill_holes()
        half.remove_unreferenced_vertices()
        ok2, guard2 = shape_guard(base_mesh, half, ds, masks)
        guard["half_detail_guard"] = guard2
        if ok2:
            final_mesh = half
        else:
            final_mesh = base_mesh.copy()
            fallback = True

    final_mesh.fill_holes()
    final_mesh.remove_unreferenced_vertices()
    if final_mesh.volume < 0:
        final_mesh.invert()
    final_mesh.export(out)

    # PLY is convenient for inspecting normals/surface quality in Blender/MeshLab.
    final_mesh.export(diag / "cat_sitting_depth_sdf_v8.ply")

    report = {
        "pipeline": "V8: stable V6 learned-depth SDF + reliability-gated isometric refinement -> masked multi-scale orthographic depth band-pass -> visibility weighted normal-only detail -> cross-view disagreement rejection -> strict V6 shape lock -> silhouette/bbox guard -> STL",
        "model": args.model,
        "resolution": list(args.resolution),
        "vertices": int(len(final_mesh.vertices)),
        "faces": int(len(final_mesh.faces)),
        "watertight": bool(final_mesh.is_watertight),
        "is_volume": bool(final_mesh.is_volume),
        "fallback_to_v6": bool(fallback),
        "bbox": {
            "xlen": float(final_mesh.bounds[1, 0] - final_mesh.bounds[0, 0]),
            "ylen": float(final_mesh.bounds[1, 1] - final_mesh.bounds[0, 1]),
            "zlen": float(final_mesh.bounds[1, 2] - final_mesh.bounds[0, 2]),
        },
        "detail_history": detail_history,
        "shape_guard": guard,
        "v6_reference": base_info,
    }
    out.with_suffix(".build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--diagnostics", default="output/depth_sdf_v8_diagnostics")
    ap.add_argument("--depth-cache", default=".depth-cache/v5")
    ap.add_argument("--iso-depth-cache", default=".depth-cache/v6_iso")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--resolution", nargs=3, type=int, default=[176, 184, 248])
    ap.add_argument("--refine-iterations", type=int, default=2)
    ap.add_argument("--iso-refine-iterations", type=int, default=2)
    ap.add_argument("--detail-iterations", type=int, default=2)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
