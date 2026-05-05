from __future__ import annotations

import argparse
import json
from pathlib import Path

from instagram.renderer.template_renderer import render_template_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Instagram JSON template with YAML/JSON bindings.")
    parser.add_argument("--template", required=True, help="Path to template JSON file.")
    parser.add_argument("--bindings", required=True, help="Path to YAML/JSON bindings file.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--palette", help="Optional palette ID override, without .json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_template_file(
        template_path=Path(args.template),
        bindings_path=Path(args.bindings),
        output_path=Path(args.output),
        palette_override=args.palette,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
