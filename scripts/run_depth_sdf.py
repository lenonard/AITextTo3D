from __future__ import annotations

import base64
import os
import sys
from pathlib import Path


def repair_contact_sheet(root: Path) -> None:
    parts = sorted(root.glob("contact_sheet.b64.part*"))
    if not parts:
        return
    payload = "".join(p.read_text(encoding="ascii").strip() for p in parts)
    # Validate before handing it to the reconstruction script.
    raw = base64.b64decode(payload, validate=True)
    if len(raw) < 1000:
        raise RuntimeError("Contact sheet decoded to an implausibly small payload")
    (root / "contact_sheet.jpg.b64").write_text(payload, encoding="ascii")
    print(f"Reassembled contact sheet from {len(parts)} chunks: {len(payload)} base64 chars / {len(raw)} bytes")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_depth_sdf.py DATASET_ROOT [depth_sdf_reconstruct.py args ...]")
    root = Path(sys.argv[1])
    repair_contact_sheet(root)
    target = Path(__file__).with_name("depth_sdf_reconstruct.py")
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    main()
