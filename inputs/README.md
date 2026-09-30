# inputs/

Đặt JSON CAD cần build vào thư mục này, ví dụ `inputs/my_object.json`.

Mỗi lần commit/push file JSON vào `inputs/`, GitHub Actions sẽ chạy trên Linux, dựng geometry bằng CadQuery/OCCT và upload STL trong artifact `aitextto3d-stl`.
