# AITextTo3D

Pipeline thử nghiệm **multi-view → JSON CAD → STL**.

Ý tưởng chính: ChatGPT phân tích ZIP ảnh nhiều góc nhìn và sinh **JSON CAD có ý nghĩa hình học**, còn GitHub Actions chạy trên Linux dùng **CadQuery + OCCT** để dựng BRep và tessellate thành STL. LLM không phải tự sinh hàng chục nghìn vertex.

```text
8-view image ZIP
      ↓
ChatGPT phân tích silhouette / part / curve
      ↓
JSON CAD v1
      ↓
GitHub Actions (Ubuntu)
      ↓
CadQuery / OCCT
      ↓
STL + build report
```

## Dùng nhanh

### 1. Thêm JSON CAD

Đặt file vào:

```text
inputs/my_model.json
```

Rồi commit/push. Workflow `JSON CAD to STL` tự chạy và tạo artifact `aitextto3d-stl`.

Hoặc chạy thủ công trong tab **Actions → JSON CAD to STL → Run workflow** và nhập đường dẫn JSON.

### 2. Chạy local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m aitextto3d build examples/penguin.json -o output/penguin.stl
```

## JSON CAD hỗ trợ

- `box`
- `sphere`
- `ellipsoid`
- `cylinder`
- `cone`
- `loft` với circle / ellipse / closed B-spline sections
- `extrude_profile`
- `sweep`
- `union`, `cut`, `intersect`
- `compound`
- scale / rotate / translate

Chi tiết: [`docs/JSON_CAD_SPEC.md`](docs/JSON_CAD_SPEC.md)

Quy trình chuyển bộ ảnh sang JSON: [`docs/MULTIVIEW_TO_JSON.md`](docs/MULTIVIEW_TO_JSON.md)

## Ví dụ

`examples/penguin.json` minh họa một object stylized có thân được dựng bằng **smooth loft**, cánh/chân/mắt dạng ellipsoid và mỏ dạng cone biến dạng. Đây là kiểu representation phù hợp hơn primitive-only JSON khi làm từ ảnh nhiều góc nhìn.

## Hiện trạng

Phiên bản này tập trung vào bước deterministic **JSON CAD → STL**. Việc nhìn ZIP ảnh và sinh JSON hiện do ChatGPT thực hiện bên ngoài Action. Sau khi bước này ổn định, có thể bổ sung automatic silhouette/depth/visual-hull processing ở phase tiếp theo.
