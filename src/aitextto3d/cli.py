from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import JsonCadError, bounding_box, build_document, export_stl, load_document


def cmd_build(args: argparse.Namespace) -> int:
    try:
        doc = load_document(args.input)
        result = build_document(doc)
        output = Path(args.output)
        export_stl(result.shape, output, args.tolerance, args.angular_tolerance)
        report = {
            "input": str(Path(args.input)),
            "output": str(output),
            "parts": list(result.parts.keys()),
            "bounding_box_mm": bounding_box(result.shape),
            "metadata": result.metadata,
        }
        report_path = output.with_suffix(".build.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (JsonCadError, KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aitextto3d", description="Build STL from AiTextTo3D JSON CAD")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="Build one JSON CAD document into STL")
    p.add_argument("input", help="Path to JSON CAD file")
    p.add_argument("-o", "--output", required=True, help="Output STL path")
    p.add_argument("--tolerance", type=float, default=0.08, help="STL linear meshing tolerance in mm")
    p.add_argument("--angular-tolerance", type=float, default=0.12, help="STL angular meshing tolerance in radians")
    p.set_defaults(func=cmd_build)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
