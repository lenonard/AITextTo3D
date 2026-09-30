from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "depth_sdf_reconstruct.py"
if not V3_PATH.exists():
    V3_PATH = Path("/mnt/data/depth_sdf_reconstruct.py")
spec = importlib.util.spec_from_file_location("depth_v3", V3_PATH)
v3 = importlib.util.module_from_spec(spec)
sys.modules["depth_v3"] = v3
assert spec and spec.loader
spec.loader.exec_module(v3)

VIEWS = ("front", "back", "left", "right")


def cache_depths(ds, diag: Path, model_id: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {v: cache_dir / f"{v}.npy" for v in VIEWS}
    if all(p.exists() for p in files.values()):
        depths = {v: np.load(files[v]).astype(np.float32) for v in VIEWS}
        print("[v4] depth cache hit")
        return depths
    depths = v3.infer_depths(ds, diag, model_id)
    for v in VIEWS:
        np.save(files[v], depths[v].astype(np.float32))
    return depths


def silhouette_width_by_z(mask: np.ndarray, height: float, n=256):
    x0, y0, x1, y1 = v3.bbox(mask)
    ppu = (y1 - y0) / height
    out = np.zeros(n, np.float32)
    for i in range(n):
        z = height * i / max(n - 1, 1)
        yy = int(np.clip(round(y1 - z * ppu), 0, mask.shape[0] - 1))
        xs = np.where(mask[yy])[0]
        if xs.size:
            out[i] = (xs.max() - xs.min()) / max(ppu, 1e-6)
    return ndimage.gaussian_filter1d(out, 2.0)


def robust_axial_width(a: np.ndarray, m: np.ndarray, n=256):
    h = a.shape[0]
    out = np.zeros(n, np.float32)
    for i in range(n):
        yy = int(np.clip(round(i * (h - 1) / max(n - 1, 1)), 0, h - 1))
        vals = a[yy][m[yy]]
        if vals.size:
            out[i] = 2.0 * np.percentile(vals, 84)
    return ndimage.gaussian_filter1d(out, 2.0)


def apply_row_scale(a: np.ndarray, scale: np.ndarray, m: np.ndarray):
    row = np.interp(np.arange(a.shape[0]), np.linspace(0, a.shape[0]-1, len(scale)), scale)
    return (a * row[:, None] * m.astype(np.float32)).astype(np.float32)


def cross_view_align(ds, surfaces, masks, diag: Path):
    target_depth = 0.5 * (silhouette_width_by_z(masks["left"], ds.height) + silhouette_width_by_z(masks["right"], ds.height))
    target_width = 0.5 * (silhouette_width_by_z(masks["front"], ds.height) + silhouette_width_by_z(masks["back"], ds.height))
    pred_depth = 0.5 * (robust_axial_width(surfaces["front"], masks["front"]) + robust_axial_width(surfaces["back"], masks["back"]))
    pred_width = 0.5 * (robust_axial_width(surfaces["left"], masks["left"]) + robust_axial_width(surfaces["right"], masks["right"]))
    sfb = np.ones_like(target_depth)
    slr = np.ones_like(target_width)
    ok = (pred_depth > 1e-5) & (target_depth > 1e-5)
    sfb[ok] = np.clip(target_depth[ok] / pred_depth[ok], 0.72, 1.35)
    ok = (pred_width > 1e-5) & (target_width > 1e-5)
    slr[ok] = np.clip(target_width[ok] / pred_width[ok], 0.72, 1.35)
    sfb = ndimage.gaussian_filter1d(sfb, 5.0)
    slr = ndimage.gaussian_filter1d(slr, 5.0)
    for v in ("front", "back"):
        surfaces[v] = apply_row_scale(surfaces[v], sfb, masks[v])
    for v in ("left", "right"):
        surfaces[v] = apply_row_scale(surfaces[v], slr, masks[v])
    data = {"fb_scale": sfb.tolist(), "lr_scale": slr.tolist(), "target_depth": target_depth.tolist(), "target_width": target_width.tolist()}
    (diag / "cross_view_alignment.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return surfaces


def view_cloud(ds, a, m, view, stride=5):
    x0, y0, x1, y1 = v3.bbox(m)
    ppu = (y1-y0) / ds.height
    cx = .5 * (x0+x1)
    yy, xx = np.where(m)
    take = np.arange(0, len(xx), stride)
    xx = xx[take].astype(np.float32); yy = yy[take].astype(np.float32)
    z = (y1 - yy) / ppu
    av = a[yy.astype(int), xx.astype(int)]
    if view == "front": x=(xx-cx)/ppu; y=-av
    elif view == "back": x=-(xx-cx)/ppu; y=av
    elif view == "left": y=-(xx-cx)/ppu; x=-av
    else: y=(xx-cx)/ppu; x=av
    return np.stack([x,y,z], 1)


def registration_score(points: np.ndarray, ortho_points: np.ndarray, view: str):
    if view in ("front", "back"):
        tree = cKDTree(ortho_points[:, [1,2]])
        d, _ = tree.query(points[:, [1,2]], k=1)
    else:
        tree = cKDTree(ortho_points[:, [0,2]])
        d, _ = tree.query(points[:, [0,2]], k=1)
    return float(np.median(np.clip(d, 0, np.percentile(d, 90))))


def point_cloud_register(ds, surfaces, masks, diag: Path):
    params = {}
    for view in VIEWS:
        ortho = ("left", "right") if view in ("front", "back") else ("front", "back")
        ortho_pts = np.concatenate([view_cloud(ds, surfaces[o], masks[o], o, 8) for o in ortho], 0)
        half = (ds.depth if view in ("front", "back") else ds.width) * 0.5
        base = surfaces[view]
        best = (1e9, 1.0, 0.0, 0.0)
        for gain in np.linspace(0.97, 1.03, 7):
            for off in np.linspace(-0.012*half, 0.012*half, 5):
                trial = np.clip(gain * base + off, 0, half) * masks[view]
                pts = view_cloud(ds, trial, masks[view], view, 8)
                nn = registration_score(pts, ortho_pts, view)
                penalty = 0.15 * half * abs(gain - 1.0) + 0.20 * abs(off)
                objective = nn + penalty
                if objective < best[0]: best = (objective, float(gain), float(off), float(nn))
        _, gain, off, nn = best
        surfaces[view] = np.clip(gain * base + off, 0, half) * masks[view]
        params[view] = {"gain": gain, "offset": off, "median_nn": nn, "regularized_objective": best[0]}
    (diag / "point_registration.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return surfaces, params


def silhouette_volumes(ds, masks, axes):
    xs, ys, zs = axes
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pxf, pyf = v3.world_to_pixel_front(X, Z, masks["front"], ds.height)
    pxb, pyb = v3.world_to_pixel_front(-X, Z, masks["back"], ds.height)
    pxl, pyl = v3.world_to_pixel_side(-Y, Z, masks["left"], ds.height)
    pxr, pyr = v3.world_to_pixel_side(Y, Z, masks["right"], ds.height)
    vols = {
        "front": v3.sample_bilinear(masks["front"].astype(np.float32), pxf, pyf) > 0.45,
        "back": v3.sample_bilinear(masks["back"].astype(np.float32), pxb, pyb) > 0.45,
        "left": v3.sample_bilinear(masks["left"].astype(np.float32), pxl, pyl) > 0.45,
        "right": v3.sample_bilinear(masks["right"].astype(np.float32), pxr, pyr) > 0.45,
    }
    allowed = vols["front"] & vols["back"] & vols["left"] & vols["right"]
    return vols, allowed


def reprojection_refine(ds, sdf: np.ndarray, masks, axes, iterations=3):
    occ = sdf >= 0
    vols, allowed = silhouette_volumes(ds, masks, axes)
    metrics=[]
    for _ in range(iterations):
        occ &= allowed
        need = np.zeros_like(occ)
        projs = {"front": occ.any(1), "back": occ.any(1), "left": occ.any(0), "right": occ.any(0)}
        targets = {"front": vols["front"].any(1), "back": vols["back"].any(1), "left": vols["left"].any(0), "right": vols["right"].any(0)}
        errs={}
        for v in VIEWS:
            miss = targets[v] & ~projs[v]
            extra = projs[v] & ~targets[v]
            errs[v] = {"missing": int(miss.sum()), "extra": int(extra.sum())}
            if v in ("front","back"):
                need |= np.repeat(miss[:,None,:], occ.shape[1], axis=1)
            else:
                need |= np.repeat(miss[None,:,:], occ.shape[0], axis=0)
        grow = ndimage.binary_dilation(occ, iterations=1) & allowed & need
        occ |= grow
        occ = ndimage.binary_closing(occ, iterations=1)
        occ = ndimage.binary_fill_holes(occ)
        metrics.append(errs)
    dist_in = ndimage.distance_transform_edt(occ)
    dist_out = ndimage.distance_transform_edt(~occ)
    return (dist_in - dist_out).astype(np.float32), metrics


def build(args):
    ds = v3.load_dataset(args.dataset_root)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics); diag.mkdir(parents=True, exist_ok=True)
    depths = cache_depths(ds, diag, args.model, Path(args.depth_cache))
    surfaces, masks = v3.make_surface_maps(ds, depths, diag)
    surfaces = cross_view_align(ds, surfaces, masks, diag)
    surfaces, reg = point_cloud_register(ds, surfaces, masks, diag)
    pts, cols = v3.make_point_cloud(ds, surfaces, masks, stride=4)
    trimesh.points.PointCloud(pts, colors=cols).export(diag / "registered_points_v4.ply")
    sdf0, axes = v3.build_sdf(ds, surfaces, masks, resolution=tuple(args.resolution), trunc_fraction=0.030)
    np.savez_compressed(diag / "pre_reprojection_sdf.npz", sdf=sdf0, x=axes[0], y=axes[1], z=axes[2])
    sdf, metrics = reprojection_refine(ds, sdf0, masks, axes, iterations=args.refine_iterations)
    np.savez_compressed(diag / "post_reprojection_sdf.npz", sdf=sdf, x=axes[0], y=axes[1], z=axes[2])
    (diag / "reprojection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mesh = v3.mesh_from_sdf(sdf, axes)
    if mesh.volume < 0: mesh.invert()
    mesh.fill_holes(); mesh.remove_unreferenced_vertices(); mesh.export(out)
    bb=mesh.bounds
    report={"pipeline":"V4: cached learned depth -> cross-view depth alignment -> point-cloud registration -> SDF -> silhouette reprojection optimization -> mesh",
            "model":args.model,"resolution":args.resolution,"vertices":int(len(mesh.vertices)),"faces":int(len(mesh.faces)),
            "watertight":bool(mesh.is_watertight),"is_volume":bool(mesh.is_volume),"registration":reg,
            "bbox":{"xlen":float(bb[1,0]-bb[0,0]),"ylen":float(bb[1,1]-bb[0,1]),"zlen":float(bb[1,2]-bb[0,2])}}
    out.with_suffix(".build.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("-o","--output",required=True)
    ap.add_argument("--diagnostics",default="output/depth_sdf_v4_diagnostics")
    ap.add_argument("--depth-cache",default=".depth-cache/v4")
    ap.add_argument("--model",default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--resolution",nargs=3,type=int,default=[152,160,216])
    ap.add_argument("--refine-iterations",type=int,default=5)
    args=ap.parse_args(); build(args)

if __name__=="__main__": main()
