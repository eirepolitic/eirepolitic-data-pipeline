from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from instagram.renderer.template_renderer import load_json, render_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Instagram JSON template with YAML/JSON bindings.")
    parser.add_argument("--template", help="Path to template JSON. Optional if --bindings contains template.")
    parser.add_argument("--bindings", required=True, help="YAML/JSON file containing bindings and optional output path.")
    parser.add_argument("--output", help="Output PNG path. Overrides output in bindings file.")
    return parser.parse_args()


def load_bindings(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Bindings file must contain a mapping: {path}")
    return data


def main() -> None:
    args = parse_args()
    config = load_bindings(args.bindings)
    template_path = args.template or config.get("template")
    if not template_path:
        raise RuntimeError("Provide --template or a template key in the bindings file.")
    output_path = args.output or config.get("output")
    if not output_path:
        raise RuntimeError("Provide --output or an output key in the bindings file.")

    bindings = config.get("bindings", config)
    if not isinstance(bindings, dict):
        raise RuntimeError("bindings must be a mapping.")

    template = load_json(template_path)
    result = render_template(template, bindings, output_path)
    print(f"Rendered: {result.output_path}")
    print(f"Source values: {result.source_values_path}")
    print(f"Manifest: {result.manifest_path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
