from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cadquery as cq


class JsonCadError(ValueError):
    pass


def _v3(value: Sequence[float] | None, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if value is None:
        return tuple(float(v) for v in default)
    if len(value) != 3:
        raise JsonCadError(f"Expected a 3-vector, got {value!r}")
    return tuple(float(v) for v in value)


def _shape(obj: Any) -> cq.Shape:
    if isinstance(obj, cq.Workplane):
        return obj.val()
    if isinstance(obj, cq.Shape):
        return obj
    raise JsonCadError(f"Operation did not produce a CadQuery Shape: {type(obj)!r}")


def _matrix_scale(sx: float, sy: float, sz: float) -> cq.Matrix:
    return cq.Matrix([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def apply_transform(shape: cq.Shape, spec: Mapping[str, Any] | None) -> cq.Shape:
    if not spec:
        return shape
    scale = spec.get("scale")
    if scale is not None:
        if isinstance(scale, (int, float)):
            sx = sy = sz = float(scale)
        else:
            sx, sy, sz = _v3(scale, (1.0, 1.0, 1.0))
        shape = shape.transformGeometry(_matrix_scale(sx, sy, sz))
    for rot in spec.get("rotate", []):
        axis_start = _v3(rot.get("axis_start"), (0, 0, 0))
        axis_end = _v3(rot.get("axis_end"), (0, 0, 1))
        angle = float(rot.get("angle_deg", 0.0))
        shape = shape.rotate(cq.Vector(*axis_start), cq.Vector(*axis_end), angle)
    translate = spec.get("translate")
    if translate is not None:
        x, y, z = _v3(translate)
        shape = shape.translate(cq.Vector(x, y, z))
    return shape


def make_ellipsoid(node: Mapping[str, Any]) -> cq.Shape:
    radii = node.get("radii")
    if not radii or len(radii) != 3:
        raise JsonCadError("ellipsoid requires radii: [rx, ry, rz]")
    rx, ry, rz = [float(v) for v in radii]
    if min(rx, ry, rz) <= 0:
        raise JsonCadError("ellipsoid radii must be > 0")
    return cq.Solid.makeSphere(1.0).transformGeometry(_matrix_scale(rx, ry, rz))


def make_box(node: Mapping[str, Any]) -> cq.Shape:
    size = node.get("size")
    if not size or len(size) != 3:
        raise JsonCadError("box requires size: [x, y, z]")
    x, y, z = [float(v) for v in size]
    centered = bool(node.get("centered", True))
    return cq.Workplane("XY").box(x, y, z, centered=(centered, centered, centered)).val()


def make_sphere(node: Mapping[str, Any]) -> cq.Shape:
    return cq.Solid.makeSphere(float(node["radius"]))


def make_cylinder(node: Mapping[str, Any]) -> cq.Shape:
    radius = float(node["radius"])
    height = float(node["height"])
    axis = _v3(node.get("axis"), (0, 0, 1))
    base = _v3(node.get("base"), (0, 0, 0))
    return cq.Solid.makeCylinder(radius, height, cq.Vector(*base), cq.Vector(*axis))


def make_cone(node: Mapping[str, Any]) -> cq.Shape:
    r1 = float(node.get("radius1", 0.0))
    r2 = float(node.get("radius2", 0.0))
    height = float(node["height"])
    axis = _v3(node.get("axis"), (0, 0, 1))
    base = _v3(node.get("base"), (0, 0, 0))
    return cq.Solid.makeCone(r1, r2, height, cq.Vector(*base), cq.Vector(*axis))


def _wire_for_section(section: Mapping[str, Any]) -> cq.Wire:
    z = float(section.get("z", 0.0))
    cx, cy = [float(v) for v in section.get("center", [0.0, 0.0])]
    kind = section.get("shape", "ellipse")
    wp = cq.Workplane("XY", origin=(cx, cy, z))
    if kind == "circle":
        wire = wp.circle(float(section["radius"])).val()
    elif kind == "ellipse":
        wire = wp.ellipse(float(section["rx"]), float(section["ry"])).val()
    elif kind == "closed_spline":
        pts = section.get("points")
        if not pts or len(pts) < 3:
            raise JsonCadError("closed_spline section requires at least 3 points")
        xy = [(float(p[0]), float(p[1])) for p in pts]
        wire = wp.moveTo(*xy[0]).spline(xy[1:], includeCurrent=True).close().val()
    else:
        raise JsonCadError(f"Unsupported loft section shape: {kind}")
    if not isinstance(wire, cq.Wire):
        raise JsonCadError(f"Section did not create a wire: {kind}")
    return wire


def make_loft(node: Mapping[str, Any]) -> cq.Shape:
    sections = node.get("sections", [])
    if len(sections) < 2:
        raise JsonCadError("loft requires at least 2 sections")
    wires = [_wire_for_section(s) for s in sections]
    return cq.Solid.makeLoft(wires, ruled=bool(node.get("ruled", False)))


def make_extrude_profile(node: Mapping[str, Any]) -> cq.Shape:
    profile = node.get("profile", {})
    distance = float(node["distance"])
    plane = str(node.get("plane", "XY"))
    wp = cq.Workplane(plane)
    kind = profile.get("shape", "closed_spline")
    if kind == "circle":
        wp = wp.circle(float(profile["radius"]))
    elif kind == "ellipse":
        wp = wp.ellipse(float(profile["rx"]), float(profile["ry"]))
    elif kind == "polygon":
        wp = wp.polyline([(float(x), float(y)) for x, y in profile["points"]]).close()
    elif kind == "closed_spline":
        pts = [(float(x), float(y)) for x, y in profile["points"]]
        wp = wp.moveTo(*pts[0]).spline(pts[1:], includeCurrent=True).close()
    else:
        raise JsonCadError(f"Unsupported extrude profile: {kind}")
    return wp.extrude(distance, both=bool(node.get("both", False))).val()


def make_sweep(node: Mapping[str, Any]) -> cq.Shape:
    path = node.get("path", {})
    pts = path.get("points", [])
    if len(pts) < 2:
        raise JsonCadError("sweep.path.points requires at least 2 points")
    plane = str(path.get("plane", "XZ"))
    p2 = [(float(p[0]), float(p[1])) for p in pts]
    path_wp = cq.Workplane(plane).moveTo(*p2[0]).spline(p2[1:], includeCurrent=True)
    profile = node.get("profile", {})
    kind = profile.get("shape", "circle")
    profile_plane = {"XY": "YZ", "XZ": "YZ", "YZ": "XZ"}.get(plane, "YZ")
    pwp = cq.Workplane(profile_plane)
    if kind == "circle":
        pwp = pwp.circle(float(profile["radius"]))
    elif kind == "ellipse":
        pwp = pwp.ellipse(float(profile["rx"]), float(profile["ry"]))
    else:
        raise JsonCadError("sweep profile supports circle or ellipse")
    return pwp.sweep(path_wp, isFrenet=bool(node.get("is_frenet", True))).val()


def build_node(node: Mapping[str, Any], named: Mapping[str, cq.Shape]) -> cq.Shape:
    op = str(node.get("op", "")).lower()
    if op == "ref":
        name = str(node["name"])
        if name not in named:
            raise JsonCadError(f"Unknown ref: {name}")
        result = named[name]
    elif op == "box": result = make_box(node)
    elif op == "sphere": result = make_sphere(node)
    elif op == "ellipsoid": result = make_ellipsoid(node)
    elif op == "cylinder": result = make_cylinder(node)
    elif op == "cone": result = make_cone(node)
    elif op == "loft": result = make_loft(node)
    elif op == "extrude_profile": result = make_extrude_profile(node)
    elif op == "sweep": result = make_sweep(node)
    elif op in {"union", "cut", "intersect"}:
        items = node.get("items", [])
        if len(items) < 2:
            raise JsonCadError(f"{op} requires at least 2 items")
        shapes = [build_node(i if isinstance(i, Mapping) else {"op": "ref", "name": i}, named) for i in items]
        result = shapes[0]
        for other in shapes[1:]:
            result = result.fuse(other) if op == "union" else result.cut(other) if op == "cut" else result.intersect(other)
    elif op == "compound":
        items = node.get("items", [])
        shapes = [build_node(i if isinstance(i, Mapping) else {"op": "ref", "name": i}, named) for i in items]
        result = cq.Compound.makeCompound(shapes)
    else:
        raise JsonCadError(f"Unsupported operation: {op!r}")
    return apply_transform(_shape(result), node.get("transform"))


@dataclass
class BuildResult:
    shape: cq.Shape
    parts: Dict[str, cq.Shape]
    metadata: Dict[str, Any]


def build_document(doc: Mapping[str, Any]) -> BuildResult:
    version = str(doc.get("version", "1.0"))
    if not version.startswith("1."):
        raise JsonCadError(f"Unsupported JSON CAD version: {version}")
    units = str(doc.get("units", "mm")).lower()
    if units not in {"mm", "millimeter", "millimeters"}:
        raise JsonCadError("v1 currently requires millimetres (units='mm')")
    named: Dict[str, cq.Shape] = {}
    for part in doc.get("parts", []):
        part_id = str(part.get("id", "")).strip()
        if not part_id:
            raise JsonCadError("Every part requires a non-empty id")
        if part_id in named:
            raise JsonCadError(f"Duplicate part id: {part_id}")
        named[part_id] = build_node(part, named)
    result_spec = doc.get("result")
    if result_spec is None:
        if not named:
            raise JsonCadError("Document has no parts and no result")
        result_spec = {"op": "compound", "items": list(named.keys())}
    return BuildResult(shape=build_node(result_spec, named), parts=named, metadata=dict(doc.get("metadata", {})))


def load_document(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JsonCadError(f"Invalid JSON in {path}: {exc}") from exc


def export_stl(shape: cq.Shape, output_path: str | Path, tolerance: float = 0.08, angular_tolerance: float = 0.12) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(output_path), exportType="STL", tolerance=float(tolerance), angularTolerance=float(angular_tolerance))
    return output_path


def bounding_box(shape: cq.Shape) -> Dict[str, float]:
    bb = shape.BoundingBox()
    return {"xmin": bb.xmin, "xmax": bb.xmax, "ymin": bb.ymin, "ymax": bb.ymax, "zmin": bb.zmin, "zmax": bb.zmax, "xlen": bb.xlen, "ylen": bb.ylen, "zlen": bb.zlen}
