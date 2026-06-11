from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.visuals.renderers.common import load_yaml, rows_from_sample


RENDERERS = {
    "horizontal_bar": "instagram.visuals.renderers.horizontal_bar",
    "vertical_bar": "instagram.visuals.renderers.vertical_bar",
    "line_chart": "instagram.visuals.renderers.line_chart",
    "stacked_bar": "instagram.visuals.renderers.stacked_bar",
    "ranking_table": "instagram.visuals.renderers.ranking_table",
    "choropleth_map": "instagram.visuals.renderers.choropleth_map",
    "point_map": "instagram.visuals.renderers.point_map",
    "table_card": "instagram.visuals.renderers.table_card",
    "small_multiples": "instagram.visuals.renderers.small_multiples",
    "area_chart": "instagram.visuals.renderers.area_chart",
    "scatter_plot": "instagram.visuals.renderers.scatter_plot",
    "dot_plot": "instagram.visuals.renderers.dot_plot",
    "lollipop_chart": "instagram.visuals.renderers.lollipop_chart",
    "slope_chart": "instagram.visuals.renderers.slope_chart",
}


def apply_filters(rows: list[dict[str, Any]], filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = list(rows)
    for rule in filters or []:
        field = str(rule.get("field", ""))
        operator = str(rule.get("operator", "equals"))
        value = str(rule.get("value", ""))
        if not field:
            continue
        if operator == "equals":
            filtered = [row for row in filtered if str(row.get(field, "")) == value]
        else:
            raise ValueError(f"Unsupported filter operator: {operator}")
    return filtered


def render_sample(sample_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    sample_path = Path(sample_path)
    if not sample_path.is_absolute():
        sample_path = REPO_ROOT / sample_path
    sample = load_yaml(sample_path)

    template_path = Path(str(sample["template"]))
    if not template_path.is_absolute():
        template_path = REPO_ROOT / template_path
    template = load_yaml(template_path)

    renderer_name = str(template.get("renderer", ""))
    module_name = RENDERERS.get(renderer_name)
    if not module_name:
        raise RuntimeError(f"Unsupported visual renderer: {renderer_name}")

    rows, input_metadata = rows_from_sample(sample)
    rows = apply_filters(rows, sample.get("filters", []))

    visual_id = str(sample.get("visual_id") or template.get("template_id"))
    output_root = Path(output_root)
    output_png = output_root / "png" / f"{visual_id}.png"
    metadata_path = output_root / "metadata" / f"{visual_id}.json"
    manifest_path = output_root / "manifests" / f"{visual_id}.render_manifest.json"

    module = importlib.import_module(module_name)
    manifest = module.render(
        template=template,
        sample=sample,
        rows=rows,
        output_png=output_png,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        input_metadata={
            **input_metadata,
            "sample_path": str(sample_path.relative_to(REPO_ROOT)),
            "template_path": str(template_path.relative_to(REPO_ROOT)),
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Instagram visual sample for review.")
    parser.add_argument("--sample", default="instagram/visuals/samples/horizontal_bar_draft_v1.sample.yml")
    parser.add_argument("--output-root", default="generated_visuals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_sample(args.sample, args.output_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
