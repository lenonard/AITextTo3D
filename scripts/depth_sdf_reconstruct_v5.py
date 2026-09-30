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
V4_PATH = HERE / "depth_sdf_reconstruct_v4.py"

spec3 = importlib.util.spec_from_file_location("depth_v3", V3_PATH)
v3 = importlib.util.module_from_spec(spec3)
sys.modules["depth_v3"] = v3
assert spec3 and spec3.loader
spec3.loader.exec_module(v3)

spec4 = importlib.util.spec_from_file_location("depth_v4", V4_PATH)
v4 = importlib.util.module_from_spec(spec4)
sys.modules["depth_v4"] = v4
assert spec4 and spec4.loader
spec4.loader.exec_module(v4)

VIEWS = ("front", "back", "left", "right")


def conservative_cross_view_align(ds, surfaces, masks, diag: Path):
    """Align thickness profiles, but keep learned depth close to its original shape."""
    target_depth = 0.5 * (
        v4.silhouette_width_by_z(masks["left"], ds.height)
        + v4.silhouette_width_by_z(masks["right"], ds.height)
    )
    target_width = 0.5 * (
        v4.silhouette_width_by_z(masks["front"], ds.height)
        + v4.silhouette_width_by_z(masks["back"], ds.height)
    )
    pred_depth = 0.5 * (
        v4.robust_axial_width(surfaces["front"], masks["front"])
        + v4.robust_axial_width(surfaces["back"], masks["back"])
    )
    pred_width = 0.5 * (
        v4.robust_axial_width(surfaces["left"], masks["left"])
        + v4.robust_axial_width(surfaces["right"], masks["right"])
    )

    sfb = np.ones_like(target_depth)
    slr = np.ones_like(target_width)
    ok = (pred_depth > 1e-5) & (target_depth > 1e-5)
    sfb[ok] = np.clip(target_depth[ok] / pred_depth[ok], 0.88, 1.18)
    ok = (pred_width > 1e-5) & (target_width > 1e-5)
    slr[ok] = np.clip(target_width[ok] / pred_width[ok], 0.88, 1.18)

    # Pull profile correction toward identity: V5 preserves learned geometry first.
    sfb = 1.0 + 0.55 * (sfb - 1.0)
    slr = 1.0 + 0.55 * (slr - 1.0)
    sfb = ndimage.gaussian_filter1d(sfb, 6.0)
    slr = ndimage.gaussian_filter1d(slr, 6.0)

    for view in ("front", "back"):
        surfaces[view] = v4.apply_row_scale(surfaces[view], sfb, masks[view])
    for view in ("left", "right"):
        surfaces[view] = v4.apply_row_scale(surfaces[view], slr, masks[view])

    (diag / "cross_view_alignment_v5.json").write_text(
        json.dumps(
            {
                "fb_scale": sfb.tolist(),
                "lr_scale": slr.tolist(),
                "target_depth": target_depth.tolist(),
                "target_width": target_width.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return surfaces


def conservative_point_register(ds, surfaces, masks, diag: Path):
    """Regularize each view with only small gain/offset changes."""
    params = {}
    for view in VIEWS:
        ortho = ("left", "right") if view in ("front", "back") else ("front", "back")
        ortho_pts = np.concatenate(
            [v4.view_cloud(ds, surfaces[o], masks[o], o, 8) for o in ortho], axis=0
        )
        half = (ds.depth if view in ("front", "back") else ds.width) * 0.5
        base = surfaces[view]
        best = (1e9, 1.0, 0.0, 0.0)

        for gain in np.linspace(0.985, 1.015, 7):
            for off in np.linspace(-0.006 * half, 0.006 * half, 5):
                trial = np.clip(gain * base + off, 0, half) * masks[view]
                pts = v4.view_cloud(ds, trial, masks[view], view, 8)
                nn = v4.registration_score(pts, ortho_pts, view)
                # Stronger regularization than V4.
                penalty = 0.55 * half * abs(gain - 1.0) + 0.80 * abs(off)
                objective = nn + penalty
                if objective < best[0]:
                    best = (objective, float(gain), float(off), float(nn))

        _, gain, off, nn = best
        surfaces[view] = np.clip(gain * base + off, 0, half) * masks[view]
        params[view] = {
            "gain": gain,
            "offset": off,
            "median_nn": nn,
            "regularized_objective": best[0],
        }

    (diag / "point_registration_v5.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return surfaces, params


def _projection_from_occ(occ: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        return occ.any(axis=1)
    if view == "back":
        return occ[:, ::-1, :].any(axis=1)
    if view == "left":
        return occ.any(axis=0)
    if view == "right":
        return occ[::-1, :, :].any(axis=0)
    raise ValueError(view)


def _resize_bool(mask: np.ndarray, shape) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 127


def _edge_band(mask: np.ndarray, pixels: int = 10) -> np.ndarray:
    er = ndimage.binary_erosion(mask, iterations=1)
    edge = mask ^ er
    dist = ndimage.distance_transform_edt(~edge)
    return np.exp(-0.5 * (dist / max(float(pixels), 1.0)) ** 2).astype(np.float32)


def _sample_boundary_weight(ds, masks, axes):
    """3-D weight: high only where at least one source silhouette is near its boundary."""
    xs, ys, zs = axes
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    pxf, pyf = v3.world_to_pixel_front(X, Z, masks["front"], ds.height)
    pxb, pyb = v3.world_to_pixel_front(-X, Z, masks["back"], ds.height)
    pxl, pyl = v3.world_to_pixel_side(-Y, Z, masks["left"], ds.height)
    pxr, pyr = v3.world_to_pixel_side(Y, Z, masks["right"], ds.height)

    wf = v3.sample_bilinear(_edge_band(masks["front"]), pxf, pyf)
    wb = v3.sample_bilinear(_edge_band(masks["back"]), pxb, pyb)
    wl = v3.sample_bilinear(_edge_band(masks["left"]), pxl, pyl)
    wr = v3.sample_bilinear(_edge_band(masks["right"]), pxr, pyr)
    return np.maximum.reduce([wf, wb, wl, wr]).astype(np.float32)


def boundary_only_reprojection_refine(ds, sdf: np.ndarray, masks, axes, iterations=2):
    """
    Preserve the learned SDF globally. Silhouette errors can only move the zero-set
    in a narrow band near the current surface and near source silhouette boundaries.
    """
    refined = sdf.astype(np.float32).copy()
    voxel = min(float(np.mean(np.diff(a))) for a in axes)
    maxdim = max(ds.width, ds.depth, ds.height)
    boundary_w = _sample_boundary_weight(ds, masks, axes)
    metrics = []

    # SDF sign used by the V3/V4 pipeline: positive = inside.
    for _ in range(iterations):
        occ = refined >= 0
        shell = np.abs(refined) <= max(2.2 * voxel, 0.010 * maxdim)
        delta = np.zeros_like(refined, dtype=np.float32)
        errs = {}

        for view in VIEWS:
            pred = _projection_from_occ(occ, view)
            target = _resize_bool(masks[view], pred.shape)
            missing = target & ~pred
            extra = pred & ~target
            errs[view] = {"missing": int(missing.sum()), "extra": int(extra.sum())}

            if view in ("front", "back"):
                add_ray = np.repeat(missing[:, None, :], refined.shape[1], axis=1)
                sub_ray = np.repeat(extra[:, None, :], refined.shape[1], axis=1)
                if view == "back":
                    add_ray = add_ray[:, ::-1, :]
                    sub_ray = sub_ray[:, ::-1, :]
            else:
                add_ray = np.repeat(missing[None, :, :], refined.shape[0], axis=0)
                sub_ray = np.repeat(extra[None, :, :], refined.shape[0], axis=0)
                if view == "right":
                    add_ray = add_ray[::-1, :, :]
                    sub_ray = sub_ray[::-1, :, :]

            # Missing silhouette -> expand: increase positive-inside SDF.
            delta[add_ray & shell] += 0.85 * voxel
            # Extra silhouette -> shrink: reduce SDF.
            delta[sub_ray & shell] -= 1.00 * voxel

        local = boundary_w * np.exp(
            -0.5 * (np.abs(refined) / max(2.5 * voxel, 1e-6)) ** 2
        )
        refined = refined + delta * local.astype(np.float32)
        refined = ndimage.gaussian_filter(refined, sigma=(0.40, 0.40, 0.35))
        metrics.append(errs)

    return refined.astype(np.float32), metrics


def build(args):
    ds = v3.load_dataset(args.dataset_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics)
    diag.mkdir(parents=True, exist_ok=True)

    depths = v4.cache_depths(ds, diag, args.model, Path(args.depth_cache))
    surfaces, masks = v3.make_surface_maps(ds, depths, diag)

    # Keep a snapshot of the original learned surface maps for diagnostics.
    for view in VIEWS:
        half = (ds.depth if view in ("front", "back") else ds.width) * 0.5
        Image.fromarray(
            (np.clip(surfaces[view] / max(half, 1e-6), 0, 1) * 255).astype(np.uint8)
        ).save(diag / f"{view}_axial_depth_original_v5.png")

    surfaces = conservative_cross_view_align(ds, surfaces, masks, diag)
    surfaces, reg = conservative_point_register(ds, surfaces, masks, diag)

    pts, cols = v3.make_point_cloud(ds, surfaces, masks, stride=4)
    trimesh.points.PointCloud(pts, colors=cols).export(diag / "registered_points_v5.ply")

    sdf0, axes = v3.build_sdf(
        ds,
        surfaces,
        masks,
        resolution=tuple(args.resolution),
        trunc_fraction=0.034,
    )
    np.savez_compressed(
        diag / "pre_boundary_refine_sdf_v5.npz",
        sdf=sdf0,
        x=axes[0],
        y=axes[1],
        z=axes[2],
    )

    sdf, metrics = boundary_only_reprojection_refine(
        ds, sdf0, masks, axes, iterations=args.refine_iterations
    )
    np.savez_compressed(
        diag / "post_boundary_refine_sdf_v5.npz",
        sdf=sdf,
        x=axes[0],
        y=axes[1],
        z=axes[2],
    )
    (diag / "reprojection_metrics_v5.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    mesh = v3.mesh_from_sdf(sdf, axes)
    if mesh.volume < 0:
        mesh.invert()
    mesh.fill_holes()
    mesh.remove_unreferenced_vertices()
    mesh.export(out)

    bb = mesh.bounds
    report = {
        "pipeline": "V5: cached Depth Anything -> conservative cross-view alignment -> regularized point registration -> learned SDF fusion -> boundary-only silhouette reprojection refinement -> mesh",
        "model": args.model,
        "resolution": args.resolution,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "registration": reg,
        "bbox": {
            "xlen": float(bb[1, 0] - bb[0, 0]),
            "ylen": float(bb[1, 1] - bb[0, 1]),
            "zlen": float(bb[1, 2] - bb[0, 2]),
        },
    }
    out.with_suffix(".build.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--diagnostics", default="output/depth_sdf_v5_diagnostics")
    ap.add_argument("--depth-cache", default=".depth-cache/v5")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--resolution", nargs=3, type=int, default=[152, 160, 216])
    ap.add_argument("--refine-iterations", type=int, default=2)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
