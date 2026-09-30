from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
import trimesh

VIEWS = ("front", "back", "left", "right")

@dataclass
class Dataset:
    root: Path
    width: float
    depth: float
    height: float


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)
    dims = json.loads((root / "metadata" / "dimensions.json").read_text(encoding="utf-8"))
    return Dataset(root=root, width=float(dims["width"]), depth=float(dims["depth"]), height=float(dims["height"]))


def read_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return m > 127


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def infer_depths(ds: Dataset, out_dir: Path, model_id: str) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    result: Dict[str, np.ndarray] = {}
    for view in VIEWS:
        image = read_rgb(ds.root / "renders" / f"{view}.png")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        result[view] = depth

        mask = read_mask(ds.root / "masks" / f"{view}_mask.png")
        vis = normalize_inverse_depth(depth, mask)
        Image.fromarray((vis * 255).astype(np.uint8), mode="L").save(out_dir / f"{view}_learned_depth.png")
    return result


def bbox(mask: np.ndarray) -> Tuple[int,int,int,int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise ValueError("empty mask")
    return int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())


def normalize_inverse_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = ndimage.binary_closing(mask, iterations=2)
    eroded = ndimage.binary_erosion(mask, iterations=max(2, int(min(mask.shape) * 0.004)))
    boundary = mask & ~eroded
    vals = depth[mask]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.float32)

    q10, q95 = np.percentile(vals, [10, 95])
    denom = max(float(q95 - q10), 1e-6)
    learned = np.clip((depth - q10) / denom, 0.0, 1.0)

    if boundary.any():
        bmed = float(np.median(depth[boundary]))
        learned2 = np.clip((depth - bmed) / max(float(q95 - bmed), 1e-6), 0.0, 1.0)
        interior = ndimage.binary_erosion(mask, iterations=max(8, int(min(mask.shape)*0.02)))
        if interior.any() and float(np.median(depth[interior])) > bmed:
            learned = 0.35 * learned + 0.65 * learned2

    dt = ndimage.distance_transform_edt(mask).astype(np.float32)
    if dt.max() > 0:
        prior = np.sqrt(dt / dt.max())
    else:
        prior = dt
    fused = 0.82 * learned + 0.18 * prior
    fused *= mask.astype(np.float32)
    fused = ndimage.gaussian_filter(fused, sigma=1.1)
    fused *= mask.astype(np.float32)
    return np.clip(fused, 0.0, 1.0)


def sample_bilinear(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    good = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    x0c = np.clip(x0, 0, w-1); x1c = np.clip(x1, 0, w-1)
    y0c = np.clip(y0, 0, h-1); y1c = np.clip(y1, 0, h-1)
    wx = x - x0; wy = y - y0
    out = ((1-wx)*(1-wy)*arr[y0c,x0c] + wx*(1-wy)*arr[y0c,x1c] + (1-wx)*wy*arr[y1c,x0c] + wx*wy*arr[y1c,x1c])
    return np.where(good, out, 0.0)


def world_to_pixel_front(x, z, mask, H):
    x0,y0,x1,y1 = bbox(mask)
    ppu = (y1-y0) / H
    cx = 0.5*(x0+x1)
    px = cx + x*ppu
    py = y1 - z*ppu
    return px, py


def world_to_pixel_side(y, z, mask, H):
    x0,y0,x1,y1 = bbox(mask)
    ppu = (y1-y0) / H
    cx = 0.5*(x0+x1)
    px = cx + y*ppu
    py = y1 - z*ppu
    return px, py


def make_surface_maps(ds: Dataset, depths: Dict[str,np.ndarray], out_dir: Path):
    surfaces = {}
    masks = {}
    for view in VIEWS:
        mask = read_mask(ds.root / "masks" / f"{view}_mask.png")
        masks[view] = mask
        rel = normalize_inverse_depth(depths[view], mask)
        half = (ds.depth if view in ("front", "back") else ds.width) * 0.5
        axial = half * (0.08 + 0.92 * rel)
        axial *= mask.astype(np.float32)
        surfaces[view] = axial
        vis = axial / max(half, 1e-6)
        Image.fromarray((np.clip(vis,0,1)*255).astype(np.uint8)).save(out_dir / f"{view}_axial_depth.png")
    return surfaces, masks


def build_sdf(ds: Dataset, surfaces, masks, resolution=(144,152,208), trunc_fraction=0.035):
    nx, ny, nz = resolution
    pad = 0.04
    xs = np.linspace(-ds.width*(0.5+pad), ds.width*(0.5+pad), nx, dtype=np.float32)
    ys = np.linspace(-ds.depth*(0.5+pad), ds.depth*(0.5+pad), ny, dtype=np.float32)
    zs = np.linspace(-ds.height*pad, ds.height*(1.0+pad), nz, dtype=np.float32)
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    pxf, pyf = world_to_pixel_front(X, Z, masks['front'], ds.height)
    pxb, pyb = world_to_pixel_front(-X, Z, masks['back'], ds.height)
    pxl, pyl = world_to_pixel_side(-Y, Z, masks['left'], ds.height)
    pxr, pyr = world_to_pixel_side(Y, Z, masks['right'], ds.height)

    af = sample_bilinear(surfaces['front'], pxf, pyf)
    ab = sample_bilinear(surfaces['back'], pxb, pyb)
    al = sample_bilinear(surfaces['left'], pxl, pyl)
    ar = sample_bilinear(surfaces['right'], pxr, pyr)

    mf = sample_bilinear(masks['front'].astype(np.float32), pxf, pyf)
    mb = sample_bilinear(masks['back'].astype(np.float32), pxb, pyb)
    ml = sample_bilinear(masks['left'].astype(np.float32), pxl, pyl)
    mr = sample_bilinear(masks['right'].astype(np.float32), pxr, pyr)

    y_front = -af
    y_back  =  ab
    x_left  = -al
    x_right =  ar

    d_front = Y - y_front
    d_back  = y_back - Y
    d_left  = X - x_left
    d_right = x_right - X

    huge = max(ds.width, ds.depth, ds.height)
    d_front = np.where(mf > 0.45, d_front, -huge)
    d_back  = np.where(mb > 0.45, d_back,  -huge)
    d_left  = np.where(ml > 0.45, d_left,  -huge)
    d_right = np.where(mr > 0.45, d_right, -huge)

    sdf = np.minimum.reduce([d_front, d_back, d_left, d_right]).astype(np.float32)
    trunc = max(ds.width, ds.depth, ds.height) * trunc_fraction
    sdf = np.clip(sdf, -trunc, trunc)
    sdf = ndimage.gaussian_filter(sdf, sigma=(1.15,1.15,1.0))
    return sdf, (xs,ys,zs)


def mesh_from_sdf(sdf, axes):
    xs,ys,zs = axes
    spacing=(float(xs[1]-xs[0]),float(ys[1]-ys[0]),float(zs[1]-zs[0]))
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=0.0, spacing=spacing)
    verts[:,0]+=xs[0]; verts[:,1]+=ys[0]; verts[:,2]+=zs[0]
    mesh=trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    comps=mesh.split(only_watertight=False)
    if len(comps)>1:
        mesh=max(comps,key=lambda m: len(m.faces))
    trimesh.smoothing.filter_taubin(mesh, lamb=0.42, nu=-0.43, iterations=8)
    mesh.remove_unreferenced_vertices()
    return mesh


def make_point_cloud(ds: Dataset, surfaces, masks, stride=5):
    pts=[]; cols=[]
    colors={'front':[240,80,80,255], 'back':[80,240,80,255], 'left':[80,80,240,255], 'right':[240,200,80,255]}
    for view in VIEWS:
        mask=masks[view]; axial=surfaces[view]
        x0,y0,x1,y1=bbox(mask); ppu=(y1-y0)/ds.height; cx=.5*(x0+x1)
        yy,xx=np.where(mask); take=np.arange(0,len(xx),stride)
        xx=xx[take].astype(np.float32); yy=yy[take].astype(np.float32); z=(y1-yy)/ppu
        if view in ('front','back'):
            x=(xx-cx)/ppu; a=axial[yy.astype(int),xx.astype(int)]
            if view=='front': y=-a
            else: x=-x; y=a
        else:
            y=(xx-cx)/ppu; a=axial[yy.astype(int),xx.astype(int)]
            if view=='left': y=-y; x=-a
            else: x=a
        pts.append(np.stack([x,y,z],axis=1)); cols.append(np.tile(np.array(colors[view],dtype=np.uint8),(len(xx),1)))
    return np.concatenate(pts,axis=0), np.concatenate(cols,axis=0)


def build(root: str|Path, output: str|Path, diagnostics: str|Path, model_id: str, resolution):
    ds=load_dataset(root)
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    diag=Path(diagnostics); diag.mkdir(parents=True,exist_ok=True)
    depths=infer_depths(ds,diag,model_id)
    surfaces,masks=make_surface_maps(ds,depths,diag)
    sdf,axes=build_sdf(ds,surfaces,masks,resolution=tuple(resolution))
    mesh=mesh_from_sdf(sdf,axes)
    mesh.export(out)
    pts,cols=make_point_cloud(ds,surfaces,masks)
    trimesh.points.PointCloud(pts,colors=cols).export(diag/'fused_learned_depth_points.ply')
    np.savez_compressed(diag/'fused_sdf.npz',sdf=sdf,x=axes[0],y=axes[1],z=axes[2])
    bb=mesh.bounds
    report={
        'pipeline':'Depth Anything V2 Small relative inverse depth -> orthographic point surfaces -> robust SDF intersection -> marching cubes -> silhouette carve',
        'model':model_id,
        'views':list(VIEWS),
        'resolution':list(resolution),
        'vertices':int(len(mesh.vertices)),
        'faces':int(len(mesh.faces)),
        'watertight':bool(mesh.is_watertight),
        'is_volume':bool(mesh.is_volume),
        'bbox':{'xlen':float(bb[1,0]-bb[0,0]),'ylen':float(bb[1,1]-bb[0,1]),'zlen':float(bb[1,2]-bb[0,2])},
    }
    (out.with_suffix('.build.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('dataset_root')
    ap.add_argument('-o','--output',required=True)
    ap.add_argument('--diagnostics',default='output/diagnostics')
    ap.add_argument('--model',default='depth-anything/Depth-Anything-V2-Small-hf')
    ap.add_argument('--resolution',nargs=3,type=int,default=[144,152,208])
    args=ap.parse_args()
    build(args.dataset_root,args.output,args.diagnostics,args.model,args.resolution)

if __name__=='__main__':
    main()
