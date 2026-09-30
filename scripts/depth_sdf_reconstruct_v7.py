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
V6_PATH = HERE / "depth_sdf_reconstruct_v6.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


v3 = _load_module("depth_v3_v7", V3_PATH)
v4 = _load_module("depth_v4_v7", V4_PATH)
v5 = _load_module("depth_v5_v7", V5_PATH)
v6 = _load_module("depth_v6_v7", V6_PATH)

ORTHO_VIEWS = ("front", "back", "left", "right")
ISO_VIEWS = ("iso_left_top", "iso_right_bottom")
ALL_VIEWS = ORTHO_VIEWS + ISO_VIEWS

VIEW_DIRS = {
    "front": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "back": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "iso_left_top": v6.ISO_CAMERA_DIRS["iso_left_top"].astype(np.float32),
    "iso_right_bottom": v6.ISO_CAMERA_DIRS["iso_right_bottom"].astype(np.float32),
}


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def camera_frame(view: str):
    c = normalize(VIEW_DIRS[view].copy())
    zup = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(zup, c))
    up = normalize(np.cross(c, right))
    return c, right, up


def generic_camera_calibration(mesh: trimesh.Trimesh, ds, mask: np.ndarray, view: str):
    c, right, up = camera_frame(view)
    origin = np.array([0.0, 0.0, 0.5 * ds.height], dtype=np.float32)
    rel = mesh.vertices.astype(np.float32) - origin[None, :]
    pu = rel @ right
    pv = rel @ up
    pd = rel @ c

    u0, u1 = np.percentile(pu, [0.4, 99.6])
    v0, v1 = np.percentile(pv, [0.4, 99.6])
    d0, d1 = np.percentile(pd, [0.4, 99.6])
    x0, y0, x1, y1 = v3.bbox(mask)
    ppu_u = max(float(x1 - x0), 1.0) / max(float(u1 - u0), 1e-6)
    ppu_v = max(float(y1 - y0), 1.0) / max(float(v1 - v0), 1e-6)
    ppu = float(math.sqrt(max(ppu_u * ppu_v, 1e-8)))
    cx = 0.5 * (x0 + x1) - 0.5 * float(u0 + u1) * ppu
    cy = 0.5 * (y0 + y1) + 0.5 * float(v0 + v1) * ppu
    return {
        "camera": c,
        "right": right,
        "up": up,
        "origin": origin,
        "ppu": ppu,
        "cx": float(cx),
        "cy": float(cy),
        "depth_min": float(d0),
        "depth_max": float(d1),
    }


def robust_affine_fit(x: np.ndarray, y: np.ndarray):
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good].astype(np.float64)
    y = y[good].astype(np.float64)
    if len(x) < 200:
        raise RuntimeError("too few depth correspondences")
    keep = np.ones(len(x), dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(6):
        A = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        sol, _, _, _ = np.linalg.lstsq(A, y[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = y - (a * x + b)
        med = float(np.median(r[keep]))
        mad = 1.4826 * float(np.median(np.abs(r[keep] - med))) + 1e-8
        limit = max(2.8 * mad, float(np.percentile(np.abs(r[keep] - med), 70)))
        new_keep = np.abs(r - med) <= max(limit, 1e-8)
        if int(new_keep.sum()) < 200 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    residual = y - (a * x + b)
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return a, b, rmse, float(keep.mean()), int(keep.sum())


def sample_surface_zbuffer(mesh: trimesh.Trimesh, calib, mask_shape, sample_count=220000):
    rng = np.random.default_rng(777)
    pts, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    rel = pts.astype(np.float32) - calib["origin"][None, :]
    u = rel @ calib["right"]
    v = rel @ calib["up"]
    d = rel @ calib["camera"]
    px = np.round(calib["cx"] + u * calib["ppu"]).astype(np.int32)
    py = np.round(calib["cy"] - v * calib["ppu"]).astype(np.int32)
    h, w = mask_shape
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[valid]; py = py[valid]; d = d[valid]
    flat = py.astype(np.int64) * w + px.astype(np.int64)
    z = np.full(h * w, -np.inf, dtype=np.float32)
    np.maximum.at(z, flat, d.astype(np.float32))
    return z.reshape(h, w)


def calibrate_depth_to_mesh(mesh, ds, depth: np.ndarray, mask: np.ndarray, view: str, diag: Path):
    rel_depth = v3.normalize_inverse_depth(depth, mask)
    calib = generic_camera_calibration(mesh, ds, mask, view)
    zbuf = sample_surface_zbuffer(mesh, calib, mask.shape)
    valid = np.isfinite(zbuf) & mask
    er = ndimage.binary_erosion(mask, iterations=3)
    valid &= er
    if int(valid.sum()) < 500:
        valid = np.isfinite(zbuf) & mask
    a, b, rmse, inlier_ratio, inliers = robust_affine_fit(rel_depth[valid], zbuf[valid])
    metric = (a * rel_depth + b).astype(np.float32)
    maxdim = max(ds.width, ds.depth, ds.height)
    fit_rel = math.exp(-((rmse / max(0.028 * maxdim, 1e-6)) ** 2))
    reliability = float(np.clip(fit_rel * (0.45 + 0.55 * inlier_ratio), 0.0, 1.0))

    dmin, dmax = calib["depth_min"], calib["depth_max"]
    metric = np.clip(metric, dmin - 0.055 * maxdim, dmax + 0.055 * maxdim)
    metric *= mask.astype(np.float32)

    vis = np.zeros_like(metric, dtype=np.float32)
    if mask.any():
        vals = metric[mask]
        lo, hi = np.percentile(vals, [2, 98])
        vis[mask] = np.clip((vals - lo) / max(float(hi - lo), 1e-6), 0, 1)
    Image.fromarray((vis * 255).astype(np.uint8)).save(diag / f"{view}_metric_depth_v7.png")

    return metric, rel_depth, calib, {
        "a": float(a),
        "b": float(b),
        "rmse": rmse,
        "inlier_ratio": inlier_ratio,
        "inliers": inliers,
        "reliability": reliability,
        "ppu": float(calib["ppu"]),
        "cx": float(calib["cx"]),
        "cy": float(calib["cy"]),
    }


def depth_normals(metric: np.ndarray, mask: np.ndarray, calib):
    sm = ndimage.gaussian_filter(metric, sigma=0.65)
    gy, gx = np.gradient(sm)
    du = gx * float(calib["ppu"])
    dv = -gy * float(calib["ppu"])
    c = calib["camera"]
    r = calib["right"]
    u = calib["up"]
    tu = r[None, None, :] + du[:, :, None] * c[None, None, :]
    tv = u[None, None, :] + dv[:, :, None] * c[None, None, :]
    n = np.cross(tu, tv)
    nlen = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.maximum(nlen, 1e-8)
    dots = np.sum(n * c[None, None, :], axis=2)
    n[dots < 0] *= -1.0
    n *= mask[:, :, None].astype(np.float32)
    return n.astype(np.float32)


def metric_depth_cloud(ds, metric, rel_depth, mask, calib, reliability: float, stride: int = 2):
    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    dist /= max(float(dist.max()), 1e-6)
    normals = depth_normals(metric, mask, calib)
    yy, xx = np.where(mask)
    take = np.arange(0, len(xx), stride)
    xx = xx[take].astype(np.int32)
    yy = yy[take].astype(np.int32)

    U = (xx.astype(np.float32) - float(calib["cx"])) / float(calib["ppu"])
    V = (float(calib["cy"]) - yy.astype(np.float32)) / float(calib["ppu"])
    D = metric[yy, xx]
    pts = (
        calib["origin"][None, :]
        + U[:, None] * calib["right"][None, :]
        + V[:, None] * calib["up"][None, :]
        + D[:, None] * calib["camera"][None, :]
    ).astype(np.float32)
    nrms = normals[yy, xx]
    conf = (
        (0.18 + 0.82 * np.sqrt(dist[yy, xx]))
        * (0.55 + 0.45 * rel_depth[yy, xx])
        * float(reliability)
    ).astype(np.float32)
    return pts, nrms, conf


def build_v6_base(ds, args, diag: Path):
    depths = v4.cache_depths(ds, diag, args.model, Path(args.depth_cache))
    surfaces, masks = v3.make_surface_maps(ds, depths, diag)
    surfaces = v5.conservative_cross_view_align(ds, surfaces, masks, diag)
    surfaces, ortho_reg = v5.conservative_point_register(ds, surfaces, masks, diag)
    sdf0, axes = v3.build_sdf(ds, surfaces, masks, resolution=tuple(args.base_resolution), trunc_fraction=0.034)
    sdf, reproj = v5.boundary_only_reprojection_refine(ds, sdf0, masks, axes, iterations=args.refine_iterations)
    base_mesh = v3.mesh_from_sdf(sdf, axes)
    if base_mesh.volume < 0:
        base_mesh.invert()
    base_mesh.fill_holes(); base_mesh.remove_unreferenced_vertices()

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
            "camera": normalize(VIEW_DIRS[view]),
            "reliability": reliability,
        }
        iso_reg[view] = {"reliability": reliability, "registration": reg}
    refined, refine_stats = v6.refine_mesh_from_iso(base_mesh, iso_data, maxdim, iterations=args.iso_refine_iterations)
    refined.fill_holes(); refined.remove_unreferenced_vertices()
    if refined.volume < 0:
        refined.invert()
    refined.export(diag / "base_v6_initialization.stl")
    return refined, depths, iso_depths, masks, {
        "ortho_registration": ortho_reg,
        "iso_registration": iso_reg,
        "iso_refine_stats": refine_stats,
        "v5_reprojection": reproj,
    }


def fuse_metric_views(base_mesh, ds, depth_maps, masks, diag: Path, max_points=320000):
    base_tree = cKDTree(base_mesh.vertices)
    maxdim = max(ds.width, ds.depth, ds.height)
    all_points = []
    all_normals = []
    all_conf = []
    fits = {}

    for view in ALL_VIEWS:
        metric, rel_depth, calib, fit = calibrate_depth_to_mesh(
            base_mesh, ds, depth_maps[view], masks[view], view, diag
        )
        pts, nrms, conf = metric_depth_cloud(
            ds, metric, rel_depth, masks[view], calib, fit["reliability"], stride=2
        )
        d, _ = base_tree.query(pts, k=1)
        base_gate = np.exp(-0.5 * (d / max(0.042 * maxdim, 1e-6)) ** 2).astype(np.float32)
        conf *= base_gate
        good = (conf > 0.055) & (d < 0.10 * maxdim) & (np.linalg.norm(nrms, axis=1) > 0.6)
        pts = pts[good]; nrms = nrms[good]; conf = conf[good]
        fits[view] = dict(fit)
        fits[view]["accepted_points"] = int(len(pts))
        if len(pts):
            all_points.append(pts); all_normals.append(nrms); all_conf.append(conf)

    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    confidence = np.concatenate(all_conf, axis=0)

    if len(points) > max_points:
        rng = np.random.default_rng(2027)
        p = confidence.astype(np.float64)
        p = p / max(float(p.sum()), 1e-12)
        idx = rng.choice(len(points), size=max_points, replace=False, p=p)
        points = points[idx]; normals = normals[idx]; confidence = confidence[idx]

    (diag / "joint_depth_calibration_v7.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    colors = np.zeros((len(points), 4), dtype=np.uint8)
    c = np.clip(confidence, 0, 1)
    colors[:, 0] = (255 * (1.0 - c)).astype(np.uint8)
    colors[:, 1] = (255 * c).astype(np.uint8)
    colors[:, 2] = 160
    colors[:, 3] = 255
    trimesh.points.PointCloud(points, colors=colors).export(diag / "metric_fused_points_v7.ply")
    return points, normals, confidence, fits


def poisson_reconstruct(points, normals, base_mesh, ds, diag: Path, depth: int = 9):
    import open3d as o3d

    maxdim = max(ds.width, ds.depth, ds.height)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    voxel = maxdim / 230.0
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=28, std_ratio=2.2)
    pcd.normalize_normals()
    o3d.io.write_point_cloud(str(diag / "poisson_input_v7.ply"), pcd, write_ascii=False)

    omesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=1.035, linear_fit=True
    )
    if len(omesh.triangles) > 430000:
        omesh = omesh.simplify_quadric_decimation(target_number_of_triangles=430000)
    omesh = omesh.filter_smooth_taubin(number_of_iterations=2)
    verts = np.asarray(omesh.vertices).astype(np.float64)
    faces = np.asarray(omesh.triangles).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: len(m.faces))

    tree = cKDTree(base_mesh.vertices)
    d, idx = tree.query(mesh.vertices, k=1)
    nearest = base_mesh.vertices[idx]
    delta = mesh.vertices - nearest
    cap = 0.050 * maxdim
    scale = np.minimum(1.0, cap / np.maximum(d, 1e-8))
    mesh.vertices = nearest + delta * scale[:, None]

    bb = base_mesh.bounds
    margin = 0.035 * maxdim
    lo = bb[0] - margin
    hi = bb[1] + margin
    mesh.vertices = np.minimum(np.maximum(mesh.vertices, lo[None, :]), hi[None, :])

    mesh.fill_holes(); mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    mesh.export(diag / "poisson_constrained_v7.stl")
    return mesh


def restore_detail(mesh: trimesh.Trimesh, points: np.ndarray, confidence: np.ndarray, ds, iterations=2):
    maxdim = max(ds.width, ds.depth, ds.height)
    tree = cKDTree(points)
    refined = mesh.copy()
    for _ in range(iterations):
        refined.remove_unreferenced_vertices()
        normals = refined.vertex_normals.copy()
        d, idx = tree.query(refined.vertices, k=5)
        if idx.ndim == 1:
            idx = idx[:, None]; d = d[:, None]
        w = 1.0 / np.maximum(d, 1e-5)
        w *= confidence[idx]
        target = np.sum(points[idx] * w[:, :, None], axis=1) / np.maximum(w.sum(axis=1)[:, None], 1e-8)
        disp = target - refined.vertices
        dn = np.sum(disp * normals, axis=1)
        confv = np.max(confidence[idx], axis=1)
        good = (np.min(d, axis=1) < 0.032 * maxdim) & (confv > 0.12)
        dn = np.clip(dn, -0.0055 * maxdim, 0.0055 * maxdim)
        step = 0.50 * confv * dn
        refined.vertices[good] += step[good, None] * normals[good]
        trimesh.smoothing.filter_taubin(refined, lamb=0.16, nu=-0.165, iterations=1)
    refined.fill_holes(); refined.remove_unreferenced_vertices()
    if refined.volume < 0:
        refined.invert()
    return refined


def build(args):
    ds = v3.load_dataset(args.dataset_root)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics); diag.mkdir(parents=True, exist_ok=True)

    base_mesh, ortho_depths, iso_depths, ortho_masks, base_info = build_v6_base(ds, args, diag)
    masks = {v: v3.read_mask(ds.root / "masks" / f"{v}_mask.png") for v in ALL_VIEWS}
    depth_maps = dict(ortho_depths)
    depth_maps.update(iso_depths)

    points, normals, confidence, fits = fuse_metric_views(
        base_mesh, ds, depth_maps, masks, diag, max_points=args.max_points
    )
    poisson = poisson_reconstruct(
        points, normals, base_mesh, ds, diag, depth=args.poisson_depth
    )
    final_mesh = restore_detail(
        poisson, points, confidence, ds, iterations=args.detail_iterations
    )

    bbase = base_mesh.bounds
    bfinal = final_mesh.bounds
    base_ext = bbase[1] - bbase[0]
    final_ext = bfinal[1] - bfinal[0]
    ratios = final_ext / np.maximum(base_ext, 1e-8)
    fallback = bool(np.any(ratios < 0.82) or np.any(ratios > 1.18) or len(final_mesh.faces) < 15000)
    if fallback:
        print("[v7] Poisson/detail result failed geometry gate; falling back to V6 base mesh")
        final_mesh = base_mesh.copy()

    final_mesh.fill_holes(); final_mesh.remove_unreferenced_vertices()
    if final_mesh.volume < 0:
        final_mesh.invert()
    final_mesh.export(out)

    report = {
        "pipeline": "V7: V6 initialization -> joint affine calibration of Depth Anything maps against current mesh -> 6-view metric point+normal fusion -> confidence filtering -> Poisson surface reconstruction -> V6-bounded geometry gate -> high-frequency normal detail restoration -> STL",
        "model": args.model,
        "base_resolution": list(args.base_resolution),
        "poisson_depth": int(args.poisson_depth),
        "fused_points": int(len(points)),
        "vertices": int(len(final_mesh.vertices)),
        "faces": int(len(final_mesh.faces)),
        "watertight": bool(final_mesh.is_watertight),
        "is_volume": bool(final_mesh.is_volume),
        "fallback_to_v6": fallback,
        "bbox": {
            "xlen": float(final_mesh.bounds[1,0]-final_mesh.bounds[0,0]),
            "ylen": float(final_mesh.bounds[1,1]-final_mesh.bounds[0,1]),
            "zlen": float(final_mesh.bounds[1,2]-final_mesh.bounds[0,2]),
        },
        "joint_depth_calibration": fits,
        "v6_initialization": base_info,
    }
    out.with_suffix(".build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--diagnostics", default="output/depth_sdf_v7_diagnostics")
    ap.add_argument("--depth-cache", default=".depth-cache/v5")
    ap.add_argument("--iso-depth-cache", default=".depth-cache/v6_iso")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--base-resolution", nargs=3, type=int, default=[168, 176, 240])
    ap.add_argument("--refine-iterations", type=int, default=2)
    ap.add_argument("--iso-refine-iterations", type=int, default=2)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--max-points", type=int, default=320000)
    ap.add_argument("--detail-iterations", type=int, default=2)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
