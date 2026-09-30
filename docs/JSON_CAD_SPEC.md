# AiTextTo3D JSON CAD v1

Đây là lớp trung gian giữa **ChatGPT nhìn bộ ảnh multi-view** và **OCCT tạo geometry/STL**.

JSON không nên chứa hàng chục nghìn vertex. ChatGPT mô tả **ý nghĩa hình học**: primitive, loft, spline profile, sweep, transform và boolean. CadQuery sử dụng OCCT để dựng BRep rồi tessellate thành STL.

## Document

```json
{"version":"1.0","units":"mm","metadata":{},"parts":[],"result":{}}
```

`parts[]` được dựng theo thứ tự và phải có `id` duy nhất.

## Operations

Hỗ trợ: `box`, `sphere`, `ellipsoid`, `cylinder`, `cone`, `loft`, `extrude_profile`, `sweep`, `union`, `cut`, `intersect`, `compound`, `ref`.

### Loft

Loft là operation quan trọng nhất cho object cong từ ảnh nhiều góc nhìn.

```json
{
  "id":"body","op":"loft",
  "sections":[
    {"z":0,"shape":"ellipse","rx":20,"ry":18,"center":[0,0]},
    {"z":30,"shape":"ellipse","rx":40,"ry":34,"center":[0,0]},
    {"z":70,"shape":"ellipse","rx":35,"ry":31,"center":[0,-2]},
    {"z":100,"shape":"ellipse","rx":15,"ry":16,"center":[0,-3]}
  ]
}
```

Section hỗ trợ `circle`, `ellipse`, `closed_spline`. `closed_spline` nhận danh sách điểm 2D.

### Extrude profile

```json
{"id":"plate","op":"extrude_profile","plane":"XY","distance":8,"both":true,"profile":{"shape":"closed_spline","points":[[-20,-10],[20,-10],[24,0],[20,10],[-20,10],[-24,0]]}}
```

### Sweep

```json
{"id":"handle","op":"sweep","path":{"plane":"XZ","points":[[0,0],[10,20],[18,45],[5,70]]},"profile":{"shape":"circle","radius":4}}
```

### Boolean / compound

```json
{"op":"union","items":["body","handle"]}
{"op":"cut","items":["body","hole"]}
{"op":"compound","items":["body","eye_left","eye_right"]}
```

Compound phù hợp cho character/stylized khi các part chạm hoặc xuyên nhẹ nhau mà không cần fuse.

## Transform

```json
"transform": {
  "scale":[1.0,0.8,1.2],
  "rotate":[{"axis_start":[0,0,0],"axis_end":[0,0,1],"angle_deg":25}],
  "translate":[10,20,30]
}
```

Thứ tự: scale → rotate → translate.

## Coordinate convention

- +X: bên phải object
- +Y: phía sau object
- +Z: phía trên
- Front camera ở phía -Y nhìn vào object
- Top camera nhìn theo -Z
