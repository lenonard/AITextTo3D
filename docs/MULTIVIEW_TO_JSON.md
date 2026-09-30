# Multi-view ZIP → JSON CAD → STL

Repo này **không bắt LLM sinh trực tiếp mesh**.

```text
ZIP ảnh 8 hướng
    ↓
ChatGPT / vision reasoning
    ↓
JSON CAD v1
    ↓
GitHub Actions (Linux)
    ↓
CadQuery + OCCT
    ↓
STL artifact
```

## Bộ view chuẩn

1. front
2. back
3. left
4. right
5. top
6. bottom
7. iso_left_top
8. iso_right_bottom

Sáu view đầu nên là orthographic/near-orthographic. Hai view isometric dùng để kiểm tra hình dạng và vùng bị che.

## 3 pass khi ChatGPT phân tích ZIP

### Pass 1 — Shape understanding
Xác định coordinate system, W/H/D, symmetry, part list, landmarks, silhouette front/side/top và phần nào là geometry hay material.

### Pass 2 — Construction strategy
- vật tròn/hữu cơ đơn giản → `ellipsoid`
- thân biến dạng liên tục → `loft`
- tay cầm/ống cong → `sweep`
- profile phẳng có đường cong → `extrude_profile`
- lỗ/cắt → primitive + `cut`

### Pass 3 — JSON CAD
Sinh JSON theo `JSON_CAD_SPEC.md`, ưu tiên số đo từ silhouette và tỷ lệ giữa các view.

## Rule quan trọng cho body gần tròn

- front view cung cấp `width(z)` → `rx(z)=width(z)/2`
- side view cung cấp `depth(z)` → `ry(z)=depth(z)/2`
- tạo 6–16 sections rồi `loft`
- top/bottom/isometric dùng để sửa center drift và kiểm tra shape

Đây là cách giảm lỗi mọi thứ thành box/cylinder.

## Output

Action tạo `*.stl` và `*.build.json` (bounding box + part list), rồi upload artifact `aitextto3d-stl`.
