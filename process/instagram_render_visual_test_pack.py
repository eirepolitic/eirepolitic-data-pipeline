from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.renderer.constants import FONT_CANDIDATES
from instagram.visuals.renderers.common import load_yaml, rows_from_sample, write_json


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
}


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES.get(key, []):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _apply_filters(rows: list[dict[str, Any]], filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _render_case(
    registry: dict[str, Any],
    template: dict[str, Any],
    case: dict[str, Any],
    output_root: Path,
    registry_path: Path,
) -> dict[str, Any]:
    case_id = str(case["id"])
    sample = {
        "visual_id": case_id,
        "template": registry["template"],
        "description": case.get("description", ""),
        "input": {"mode": "local_csv", "path": case["data"]},
        "geography": case.get("geography", registry.get("geography", {})),
        "bindings": registry.get("bindings", {}),
        "filters": case.get("filters", []),
        "grouping": {"batch_enabled": False, "test_case": case_id},
        "source_note": registry.get("source_note", ""),
        "attribution": registry.get("attribution", {}),
    }

    rows, input_metadata = rows_from_sample(sample)
    rows = _apply_filters(rows, sample.get("filters", []))
    renderer_name = str(template.get("renderer", ""))
    module_name = RENDERERS.get(renderer_name)
    if not module_name:
        raise RuntimeError(f"Unsupported visual renderer: {renderer_name}")

    output_png = output_root / "png" / f"{case_id}.png"
    metadata_path = output_root / "metadata" / f"{case_id}.json"
    manifest_path = output_root / "manifests" / f"{case_id}.render_manifest.json"

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
            "case_id": case_id,
            "case_description": case.get("description", ""),
            "registry_path": str(registry_path.relative_to(REPO_ROOT)),
            "template_path": registry["template"],
        },
    )
    manifest["description"] = case.get("description", "")
    write_json(manifest_path, manifest)
    return manifest


def _fit_thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#0f2f24")
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def build_contact_sheet(manifests: list[dict[str, Any]], output_root: Path, columns: int = 3) -> Path:
    tile_width = 540
    image_height = 430
    header_height = 74
    tile_height = header_height + image_height
    gap = 18
    margin = 24
    rows = math.ceil(len(manifests) / columns)

    sheet_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    sheet_height = margin * 2 + rows * tile_height + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#0f2f24")
    draw = ImageDraw.Draw(sheet)
    title_font = _font("bold", 26)
    note_font = _font("regular", 18)

    for index, manifest in enumerate(manifests):
        case_id = str(manifest["visual_id"])
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        y = margin + row * (tile_height + gap)

        draw.rounded_rectangle((x, y, x + tile_width, y + tile_height), radius=18, fill="#173d30", outline="#f4ead7", width=2)
        draw.text((x + 18, y + 12), case_id, font=title_font, fill="#f4ead7")
        warnings = manifest.get("warnings", []) or []
        warning_text = " · ".join(warnings) if warnings else "No warnings"
        draw.text((x + 18, y + 45), warning_text, font=note_font, fill="#cbbf9f")

        image_path = Path(str(manifest["output_png"]))
        image = Image.open(image_path).convert("RGB")
        thumb = _fit_thumbnail(image, tile_width - 24, image_height - 24)
        sheet.paste(thumb, (x + 12, y + header_height + 12))

    output_path = output_root / "contact_sheet.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


def run_test_pack(registry_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    registry_path = Path(registry_path)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    registry = load_yaml(registry_path)

    template_path = Path(str(registry["template"]))
    if not template_path.is_absolute():
        template_path = REPO_ROOT / template_path
    template = load_yaml(template_path)

    output_root = Path(output_root)
    manifests = [_render_case(registry, template, case, output_root, registry_path) for case in registry.get("cases", [])]
    contact_sheet = build_contact_sheet(manifests, output_root)

    pack_manifest = {
        "success": True,
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "template": str(template_path.relative_to(REPO_ROOT)),
        "case_count": len(manifests),
        "contact_sheet": str(contact_sheet),
        "cases": manifests,
    }
    write_json(output_root / "test_pack_manifest.json", pack_manifest)
    return pack_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Instagram visual variation test pack.")
    parser.add_argument("--registry", default="instagram/visuals/tests/horizontal_bar_draft_v1/cases.yml")
    parser.add_argument("--output-root", default="generated_visual_tests/horizontal_bar_draft_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_test_pack(args.registry, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
