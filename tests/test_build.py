import json
from pathlib import Path

from aitextto3d.engine import bounding_box, build_document, export_stl


def test_curved_loft(tmp_path: Path):
    doc = json.loads(Path("examples/curved_vase.json").read_text())
    result = build_document(doc)
    bb = bounding_box(result.shape)
    assert bb["zlen"] > 100
    out = tmp_path / "vase.stl"
    export_stl(result.shape, out)
    assert out.exists() and out.stat().st_size > 1000


def test_penguin_compound(tmp_path: Path):
    doc = json.loads(Path("examples/penguin.json").read_text())
    result = build_document(doc)
    bb = bounding_box(result.shape)
    assert bb["zlen"] > 100
    assert len(result.parts) >= 8
    out = tmp_path / "penguin.stl"
    export_stl(result.shape, out)
    assert out.exists() and out.stat().st_size > 1000
