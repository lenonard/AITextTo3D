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
from scipy.spatial import cKDTree
import trimesh

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "depth_sdf_reconstruct.py"
V4_PATH = HERE / "depth_sdf_reconstruct_v4.py"
V5_PATH = HERE / "depth_sdf_reconstruct_v5.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


v3 = _load_module("depth_v3", V3_PATH)
v4 = _load_module("depth_v4", V4_PATH)
v5 = _load_module("depth_v5", V5_PATH)

ORTHO_VIEWS = ("front", "back", "left", "right")
ISO_VIEWS = ("iso_left_top", "iso_right_bottom")

ISO_CAMERA_DIRS = {
    "iso_left_top": np.array([-1.0, -1.0, 0.78], dtype=np.float32),
    "iso_right_bottom": np.array([1.0, -1.0, -0.62], dtype=np.float32),
}


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def camera_frame(view: str):
    c = normalize(ISO_CAMERA_DIRS[view].copy())
    zup = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(zup, c))
    up = normalize(np.cross(c, right))
    return c, right, up


def cache_iso_depths(ds, diag: Path, model_id: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {v: cache_dir / f"{v}.npy" for v in ISO_VIEWS}
    if all(p.exists() for p in files.values()):
        print("[v6] iso depth cache hit")
        return {v: np.load(files[v]).astype(np.float32) for v in ISO_VIEWS}

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(torch.device("cpu")).eval()
    out = {}
    for view in ISO_VIEWS:
        image = v3.read_rgb(v3.render_path(ds.root, view))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        out[view] = depth
        np.save(files[view], depth)
        mask = v3.read_mask(ds.root / "masks" / f"{view}_mask.png")
        rel = v3.normalize_inverse_depth(depth, mask)
        Image.fromarray((rel * 255).astype(np.uint8)).save(diag / f"{view}_learned_depth_v6.png")
    return out


def mask_bbox(mask: np.ndarray):
    y, x = np.where(mask)
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def calibrate_iso_camera(mesh: trimesh.Trimesh, ds, mask: np.ndarray, view: str):
    c, right, up = camera_frame(view)
    origin = np.array([0.0, 0.0, 0.5 * ds.height], dtype=np.float32)
    rel = mesh.vertices.astype(np.float32) - origin[None, :]
    pu = rel @ right
    pv = rel @ up
    pd = rel @ c

    u0, u1 = np.percentile(pu, [0.5, 99.5])
    v0, v1 = np.percentile(pv, [0.5, 99.5])
    d0, d1 = np.percentile(pd, [0.5, 99.5])
    x0, y0, x1, y1 = mask_bbox(mask)
    bw = max(float(x1 - x0), 1.0)
    bh = max(float(y1 - y0), 1.0)
    ppu_u = bw / max(float(u1 - u0), 1e-6)
    ppu_v = bh / max(float(v1 - v0), 1e-6)
    ppu = float(math.sqrt(max(ppu_u * ppu_v, 1e-8)))
    cx = 0.5 * (x0 + x1) - 0.5 * (u0 + u1) * ppu
    cy = 0.5 * (y0 + y1) + 0.5 * (v0 + v1) * ppu
    return {
        "camera": c,
        "right": right,
        "up": up,
        "origin": origin,
        "ppu": ppu,
        "cx": float(cx),
        "cy": float(cy),
        "depth_center": 0.5 * float(d0 + d1),
        "depth_half": 0.5 * float(d1 - d0),
        "projected_range": {
            "u": [float(u0), float(u1)],
            "v": [float(v0), float(v1)],
            "d": [float(d0), float(d1)],
        },
    }


def build_iso_cloud(ds, mesh, depth: np.ndarray, view: str, stride: int = 3):
    mask = v3.read_mask(ds.root / "masks" / f"{view}_mask.png")
    rel_depth = v3.normalize_inverse_depth(depth, mask)
    calib = calibrate_iso_camera(mesh, ds, mask, view)

    yy, xx = np.where(mask)
    take = np.arange(0, len(xx), stride)
    xx = xx[take].astype(np.float32)
    yy = yy[take].astype(np.float32)
    rel = rel_depth[yy.astype(np.int32), xx.astype(np.int32)]

    u = (xx - calib["cx"]) / calib["ppu"]
    v = (calib["cy"] - yy) / calib["ppu"]
    d = calib["depth_center"] + calib["depth_half"] * (0.06 + 0.94 * rel)

    pts = (
        calib["origin"][None, :]
        + u[:, None] * calib["right"][None, :]
        + v[:, None] * calib["up"][None, :]
        + d[:, None] * calib["camera"][None, :]
    ).astype(np.float32)

    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    dist /= max(float(dist.max()), 1e-6)
    conf_img = np.sqrt(dist) * (0.45 + 0.55 * rel_depth)
    conf = conf_img[yy.astype(np.int32), xx.astype(np.int32)].astype(np.float32)
    return pts, conf, calib


def kabsch_transform(src: np.ndarray, dst: np.ndarray):
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    X = src - cs
    Y = dst - cd
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    denom = float((X * X).sum())
    scale = float(S.sum() / max(denom, 1e-8))
    scale = float(np.clip(scale, 0.975, 1.025))
    t = cd - scale * (R @ cs)
    return scale, R.astype(np.float32), t.astype(np.float32)


def rotation_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    angle = math.acos(float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
    return math.degrees(angle)


def conservative_trimmed_icp(points: np.ndarray, reference: np.ndarray, maxdim: float, iterations: int = 5):
    src = points.astype(np.float32).copy()
    tree = cKDTree(reference)
    total_R = np.eye(3, dtype=np.float32)
    total_s = 1.0
    total_t = np.zeros(3, dtype=np.float32)
    history = []

    for _ in range(iterations):
        d, idx = tree.query(src, k=1)
        cutoff = min(float(np.percentile(d, 62)), 0.11 * maxdim)
        keep = d <= max(cutoff, 1e-6)
        if int(keep.sum()) < 40:
            break
        a = src[keep]
        b = reference[idx[keep]]
        s, R, t = kabsch_transform(a, b)

        angle = rotation_angle_deg(R)
        if angle > 4.0:
            alpha = 4.0 / max(angle, 1e-6)
            M = (1.0 - alpha) * np.eye(3) + alpha * R
            U, _, Vt = np.linalg.svd(M)
            R = (U @ Vt).astype(np.float32)
        tnorm = float(np.linalg.norm(t))
        if tnorm > 0.035 * maxdim:
            t *= (0.035 * maxdim / tnorm)

        src = (s * (src @ R.T) + t[None, :]).astype(np.float32)
        total_t = (s * (R @ total_t) + t).astype(np.float32)
        total_R = (R @ total_R).astype(np.float32)
        total_s *= s
        history.append({
            "median_nn": float(np.median(d[keep])),
            "trimmed_points": int(keep.sum()),
            "step_scale": float(s),
            "step_rotation_deg": float(rotation_angle_deg(R)),
            "step_translation": [float(x) for x in t],
        })
    final_d, _ = tree.query(src, k=1)
    return src, {
        "history": history,
        "final_median_nn": float(np.median(final_d)),
        "final_p80_nn": float(np.percentile(final_d, 80)),
        "total_scale": float(total_s),
        "total_rotation_deg": float(rotation_angle_deg(total_R)),
        "total_translation": [float(x) for x in total_t],
    }


def registration_reliability(reg: dict, maxdim: float) -> float:
    med = float(reg.get("final_median_nn", maxdim))
    rot = float(reg.get("total_rotation_deg", 180.0))
    scale = float(reg.get("total_scale", 1.0))
    a = math.exp(-((med / max(0.035 * maxdim, 1e-6)) ** 2))
    b = math.exp(-((rot / 7.0) ** 2))
    c = math.exp(-((math.log(max(scale, 1e-6)) / 0.13) ** 2))
    r = a * b * c
    if rot > 12.0 or scale < 0.90 or scale > 1.10:
        r *= 0.10
    return float(np.clip(r, 0.0, 1.0))


def refine_mesh_from_iso(mesh: trimesh.Trimesh, iso_data, maxdim: float, iterations: int = 2):
    refined = mesh.copy()
    stats = []
    for _ in range(iterations):
        refined.remove_unreferenced_vertices()
        normals = refined.vertex_normals.copy()
        verts = refined.vertices.copy()
        accum = np.zeros_like(verts, dtype=np.float64)
        wsum = np.zeros(len(verts), dtype=np.float64)
        iter_stat = {}

        for view, data in iso_data.items():
            reliability = float(data.get("reliability", 1.0))
            if reliability < 0.05:
                iter_stat[view] = {
                    "supported_vertices": 0,
                    "median_nn_supported": None,
                    "reliability": reliability,
                    "used": False,
                }
                continue
            pts = data["points"]
            conf = data["confidence"]
            c = data["camera"]
            tree = cKDTree(pts)
            d, idx = tree.query(verts, k=1)
            facing = np.clip(normals @ c, 0.0, 1.0)
            sigma = 0.032 * maxdim
            dist_w = np.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)
            c_w = conf[idx]
            w = dist_w * (facing ** 1.4) * c_w * reliability
            good = (d < 0.065 * maxdim) & (facing > 0.10) & (w > 0.025)

            target = pts[idx]
            disp = target - verts
            dn = np.sum(disp * normals, axis=1)
            dn = np.clip(dn, -0.010 * maxdim, 0.010 * maxdim)
            ndisp = dn[:, None] * normals
            accum[good] += ndisp[good] * w[good, None]
            wsum[good] += w[good]
            iter_stat[view] = {
                "supported_vertices": int(good.sum()),
                "median_nn_supported": float(np.median(d[good])) if good.any() else None,
                "reliability": reliability,
                "used": True,
            }

        good = wsum > 1e-8
        delta = np.zeros_like(verts)
        delta[good] = accum[good] / wsum[good, None]
        verts[good] += 0.34 * delta[good]
        refined.vertices = verts
        trimesh.smoothing.filter_taubin(refined, lamb=0.22, nu=-0.23, iterations=1)
        stats.append(iter_stat)
    return refined, stats


def silhouette_iou_mesh(mesh: trimesh.Trimesh, ds, mask: np.ndarray, view: str):
    verts = mesh.vertices
    if view == "front":
        x0,y0,x1,y1 = v3.bbox(mask); ppu=(y1-y0)/ds.height; cx=.5*(x0+x1)
        px=cx+verts[:,0]*ppu; py=y1-verts[:,2]*ppu
    elif view == "back":
        x0,y0,x1,y1 = v3.bbox(mask); ppu=(y1-y0)/ds.height; cx=.5*(x0+x1)
        px=cx-verts[:,0]*ppu; py=y1-verts[:,2]*ppu
    elif view == "left":
        x0,y0,x1,y1 = v3.bbox(mask); ppu=(y1-y0)/ds.height; cx=.5*(x0+x1)
        px=cx-verts[:,1]*ppu; py=y1-verts[:,2]*ppu
    else:
        x0,y0,x1,y1 = v3.bbox(mask); ppu=(y1-y0)/ds.height; cx=.5*(x0+x1)
        px=cx+verts[:,1]*ppu; py=y1-verts[:,2]*ppu
    pred=np.zeros(mask.shape,bool)
    xi=np.clip(np.round(px).astype(int),0,mask.shape[1]-1); yi=np.clip(np.round(py).astype(int),0,mask.shape[0]-1)
    pred[yi,xi]=True
    pred=ndimage.binary_dilation(pred,iterations=3)
    pred=ndimage.binary_fill_holes(pred)
    inter=np.logical_and(pred,mask).sum(); union=np.logical_or(pred,mask).sum()
    return float(inter/max(union,1))


def build(args):
    ds = v3.load_dataset(args.dataset_root)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics); diag.mkdir(parents=True, exist_ok=True)

    depths = v4.cache_depths(ds, diag, args.model, Path(args.depth_cache))
    surfaces, masks = v3.make_surface_maps(ds, depths, diag)
    surfaces = v5.conservative_cross_view_align(ds, surfaces, masks, diag)
    surfaces, ortho_reg = v5.conservative_point_register(ds, surfaces, masks, diag)
    sdf0, axes = v3.build_sdf(ds, surfaces, masks, resolution=tuple(args.resolution), trunc_fraction=0.034)
    sdf, reproj = v5.boundary_only_reprojection_refine(ds, sdf0, masks, axes, iterations=args.refine_iterations)
    base_mesh = v3.mesh_from_sdf(sdf, axes)
    if base_mesh.volume < 0: base_mesh.invert()
    base_mesh.fill_holes(); base_mesh.remove_unreferenced_vertices()
    base_mesh.export(diag / "base_v5_before_iso.stl")

    iso_depths = cache_iso_depths(ds, diag, args.model, Path(args.iso_depth_cache))
    ref_pts = base_mesh.vertices
    if len(ref_pts) > 45000:
        rng = np.random.default_rng(1234)
        ref_pts = ref_pts[rng.choice(len(ref_pts), 45000, replace=False)]
    maxdim = max(ds.width, ds.depth, ds.height)
    iso_data = {}
    iso_reg = {}
    colors = {"iso_left_top": [255, 80, 220, 255], "iso_right_bottom": [80, 220, 255, 255]}

    for view in ISO_VIEWS:
        raw_pts, conf, calib = build_iso_cloud(ds, base_mesh, iso_depths[view], view, stride=3)
        reg_pts, reg = conservative_trimmed_icp(raw_pts, ref_pts, maxdim, iterations=5)
        reliability = registration_reliability(reg, maxdim)
        iso_data[view] = {
            "points": reg_pts,
            "confidence": conf,
            "camera": normalize(ISO_CAMERA_DIRS[view]),
            "reliability": reliability,
        }
        iso_reg[view] = {"reliability": reliability, "calibration": {
            "ppu": calib["ppu"],
            "cx": calib["cx"],
            "cy": calib["cy"],
            "depth_center": calib["depth_center"],
            "depth_half": calib["depth_half"],
            "projected_range": calib["projected_range"],
        }, "registration": reg}
        trimesh.points.PointCloud(
            reg_pts,
            colors=np.tile(np.array(colors[view], dtype=np.uint8), (len(reg_pts),1)),
        ).export(diag / f"{view}_registered_points_v6.ply")

    (diag / "iso_registration_v6.json").write_text(json.dumps(iso_reg, indent=2), encoding="utf-8")

    refined, refine_stats = refine_mesh_from_iso(base_mesh, iso_data, maxdim, iterations=args.iso_refine_iterations)
    refined.fill_holes(); refined.remove_unreferenced_vertices()
    if refined.volume < 0: refined.invert()
    refined.export(out)

    ortho_iou = {v: silhouette_iou_mesh(refined, ds, masks[v], v) for v in ORTHO_VIEWS}
    report = {
        "pipeline": "V6: V5 orthographic learned-depth SDF base -> Depth Anything on 2 isometric views -> calibrated diagonal point clouds -> reliability-gated trimmed ICP -> normal-only visibility/confidence weighted mesh refinement -> solid STL",
        "model": args.model,
        "resolution": list(args.resolution),
        "vertices": int(len(refined.vertices)),
        "faces": int(len(refined.faces)),
        "watertight": bool(refined.is_watertight),
        "is_volume": bool(refined.is_volume),
        "bbox": {
            "xlen": float(refined.bounds[1,0]-refined.bounds[0,0]),
            "ylen": float(refined.bounds[1,1]-refined.bounds[0,1]),
            "zlen": float(refined.bounds[1,2]-refined.bounds[0,2]),
        },
        "ortho_registration": ortho_reg,
        "iso_registration": iso_reg,
        "iso_refine_stats": refine_stats,
        "orthographic_vertex_projection_iou": ortho_iou,
        "v5_reprojection_iterations": reproj,
    }
    out.with_suffix(".build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--diagnostics", default="output/depth_sdf_v6_diagnostics")
    ap.add_argument("--depth-cache", default=".depth-cache/v5")
    ap.add_argument("--iso-depth-cache", default=".depth-cache/v6_iso")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--resolution", nargs=3, type=int, default=[160, 168, 224])
    ap.add_argument("--refine-iterations", type=int, default=2)
    ap.add_argument("--iso-refine-iterations", type=int, default=2)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
