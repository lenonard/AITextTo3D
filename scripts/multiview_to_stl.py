from __future__ import annotations
import argparse, io, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure
import trimesh

ORTHO_VIEWS = ["front", "back", "left", "right", "top", "bottom"]

@dataclass
class Dataset:
    name: str
    width: float
    height: float
    depth: float
    contours: Dict[str, List[Tuple[float, float]]]
    metrics: Dict[str, dict]
    masks: Dict[str, np.ndarray] | None = None


def _load_json_from_zip(zf: zipfile.ZipFile, suffix: str):
    for n in zf.namelist():
        if n.endswith(suffix):
            return json.loads(zf.read(n).decode('utf-8'))
    raise FileNotFoundError(suffix)


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    masks = None
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as zf:
            contours_raw = _load_json_from_zip(zf, 'metadata/contours.json')
            dims = _load_json_from_zip(zf, 'metadata/dimensions.json')
            desc = _load_json_from_zip(zf, 'metadata/description.json')
            try:
                metrics = _load_json_from_zip(zf, 'metadata/view_metrics.json')
            except FileNotFoundError:
                metrics = {}
            masks = {}
            for view in ORTHO_VIEWS:
                suffix = f'masks/{view}_mask.png'
                found = next((n for n in zf.namelist() if n.endswith(suffix)), None)
                if found:
                    masks[view] = np.array(Image.open(io.BytesIO(zf.read(found))).convert('L')) > 127
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        contours_raw = data['contours']
        dims = data['dimensions']
        desc = data.get('description', {})
        metrics = data.get('view_metrics', {})
    contours = {k: [tuple(map(float,p)) for p in contours_raw[k]['points']] for k in contours_raw if 'points' in contours_raw[k]}
    return Dataset(
        name=desc.get('object_name', path.stem),
        width=float(dims['width']),
        height=float(dims['height']),
        depth=float(dims['depth']),
        contours=contours,
        metrics=metrics,
        masks=masks,
    )


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError('empty mask')
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def polygon_to_mask(points: List[Tuple[float, float]], size: int=768) -> np.ndarray:
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    xy = [(p[0]*(size-1), p[1]*(size-1)) for p in points]
    draw.polygon(xy, fill=255)
    return np.array(img, dtype=np.uint8) > 0


def view_expected_ratio(ds: Dataset, view: str) -> float:
    if view in ('front','back'):
        return ds.width / ds.height
    if view in ('left','right'):
        return ds.depth / ds.height
    if view in ('top','bottom'):
        return ds.width / ds.depth
    return 1.0


def view_reliability(ds: Dataset, view: str, mask: np.ndarray) -> float:
    x0,y0,x1,y1 = bbox_from_mask(mask)
    obs = max(1.0, x1-x0+1) / max(1.0, y1-y0+1)
    exp = view_expected_ratio(ds, view)
    logerr = abs(math.log(max(obs,1e-9)/max(exp,1e-9)))
    sigma = 0.18
    w = math.exp(-0.5*(logerr/sigma)**2)
    return float(np.clip(w, 0.02, 1.0))


def soft_crop(mask: np.ndarray, tolerance_px: float=3.5) -> np.ndarray:
    x0,y0,x1,y1 = bbox_from_mask(mask)
    crop = mask[y0:y1+1, x0:x1+1]
    crop = ndimage.binary_closing(crop, iterations=1)
    din = ndimage.distance_transform_edt(crop)
    dout = ndimage.distance_transform_edt(~crop)
    sdf = din - dout
    score = 1.0 / (1.0 + np.exp(-sdf/max(tolerance_px, 1e-6)))
    return score.astype(np.float32)


def sample_zero(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h,w = img.shape
    x = u*(w-1); y = v*(h-1)
    valid = (x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)
    out = np.zeros_like(x, dtype=np.float32)
    if not np.any(valid):
        return out
    xv=x[valid]; yv=y[valid]
    x0=np.floor(xv).astype(np.int32); y0=np.floor(yv).astype(np.int32)
    x1=np.minimum(x0+1,w-1); y1=np.minimum(y0+1,h-1)
    wx=xv-x0; wy=yv-y0
    vals=(img[y0,x0]*(1-wx)*(1-wy)+img[y0,x1]*wx*(1-wy)+img[y1,x0]*(1-wx)*wy+img[y1,x1]*wx*wy)
    out[valid]=vals
    return out


def build_source_masks(ds: Dataset):
    masks={}
    if ds.masks:
        masks.update(ds.masks)
    for v in ORTHO_VIEWS:
        if v not in masks:
            masks[v]=polygon_to_mask(ds.contours[v], size=768)
    return masks


def make_score_volume(ds: Dataset, resolution=(120,126,170), pad=0.12):
    nx,ny,nz=resolution
    W,H,D=ds.width,ds.height,ds.depth
    xs=np.linspace(-W/2-pad*W, W/2+pad*W, nx)
    ys=np.linspace(-D/2-pad*D, D/2+pad*D, ny)
    zs=np.linspace(-pad*H, H+pad*H, nz)
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij')

    raw=build_source_masks(ds)
    soft={v:soft_crop(raw[v]) for v in ORTHO_VIEWS}
    weights={v:view_reliability(ds,v,raw[v]) for v in ORTHO_VIEWS}

    active=['front','back','left','right']
    for v in ['top','bottom']:
        if weights[v] >= 0.45:
            active.append(v)

    logs=[]; ws=[]
    for v in active:
        if v=='front':
            u=(X+W/2)/W; vv=(H-Z)/H
        elif v=='back':
            u=(W/2-X)/W; vv=(H-Z)/H
        elif v=='left':
            u=(D/2-Y)/D; vv=(H-Z)/H
        elif v=='right':
            u=(Y+D/2)/D; vv=(H-Z)/H
        elif v=='top':
            u=(X+W/2)/W; vv=(D/2-Y)/D
        elif v=='bottom':
            u=(X+W/2)/W; vv=(Y+D/2)/D
        p=sample_zero(soft[v],u,vv)
        w=weights[v]
        logs.append(w*np.log(np.clip(p,1e-5,1.0)))
        ws.append(w)
    score=np.exp(np.sum(logs,axis=0)/max(np.sum(ws),1e-9))
    score=ndimage.gaussian_filter(score,sigma=(0.75,0.75,0.75))
    return score,(xs,ys,zs),weights,active


def score_to_mesh(score,axes,level=0.48):
    xs,ys,zs=axes
    spacing=(float(xs[1]-xs[0]),float(ys[1]-ys[0]),float(zs[1]-zs[0]))
    verts,faces,normals,_=measure.marching_cubes(score,level=level,spacing=spacing)
    verts[:,0]+=xs[0]; verts[:,1]+=ys[0]; verts[:,2]+=zs[0]
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,vertex_normals=normals,process=True)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.smoothing.filter_laplacian(mesh,iterations=3,lamb=0.35)
    mesh.remove_unreferenced_vertices()
    return mesh


def build(path_in,path_out,resolution=(120,126,170),level=0.48):
    ds=load_dataset(path_in)
    score,axes,weights,active=make_score_volume(ds,resolution=resolution)
    mesh=score_to_mesh(score,axes,level=level)
    out=Path(path_out); out.parent.mkdir(parents=True,exist_ok=True); mesh.export(out)
    bb=mesh.bounds
    report={
      'pipeline':'multiview_soft_visual_hull_v2',
      'input':str(path_in),'output':str(out),'name':ds.name,
      'resolution':list(resolution),'isosurface_level':level,
      'active_views':active,'view_reliability':weights,
      'watertight':bool(mesh.is_watertight),'is_volume':bool(mesh.is_volume),
      'volume':float(mesh.volume),'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces)),
      'bbox':{'xmin':float(bb[0,0]),'xmax':float(bb[1,0]),'xlen':float(bb[1,0]-bb[0,0]),
              'ymin':float(bb[0,1]),'ymax':float(bb[1,1]),'ylen':float(bb[1,1]-bb[0,1]),
              'zmin':float(bb[0,2]),'zmax':float(bb[1,2]),'zlen':float(bb[1,2]-bb[0,2])}
    }
    out.with_suffix('.build.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description='Robust multiview silhouette reconstruction to STL')
    ap.add_argument('input')
    ap.add_argument('-o','--output',required=True)
    ap.add_argument('--resolution',nargs=3,type=int,default=[120,126,170])
    ap.add_argument('--level',type=float,default=0.48)
    args=ap.parse_args()
    build(args.input,args.output,tuple(args.resolution),args.level)

if __name__=='__main__':
    main()
