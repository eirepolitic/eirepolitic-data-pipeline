from __future__ import annotations

import argparse
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
from instagram.visuals.renderers.common import utc_now, write_json


def _font(kind: str, size: int) -> ImageFont.ImageFont:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES.get(key, []):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#0f2f24")
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("manifest_path", str(path))
    return payload


def _find_manifest(root: Path) -> dict[str, Any] | None:
    manifest_dir = root / "manifests"
    candidates = sorted(manifest_dir.glob("*.render_manifest.json")) if manifest_dir.exists() else []
    if not candidates:
        return None
    return _load_manifest(candidates[0])


def _manifest_label(manifest: dict[str, Any], fallback: str) -> str:
    visual_id = str(manifest.get("visual_id") or fallback)
    if visual_id == "horizontal_bar_s3_debate_issues_draft_v1":
        return "Debate issue counts"
    if visual_id == "donut_chart_s3_member_parties_draft_v1":
        return "Member party counts"
    return visual_id.replace("_", " ")


def build_s3_smoke_contact_sheet(input_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_root = Path(input_root)
    output_path = Path(output_path)
    smoke_roots = [
        input_root / "debate_issues",
        input_root / "member_parties",
    ]
    manifests = [manifest for root in smoke_roots if (manifest := _find_manifest(root))]
    if not manifests:
        raise FileNotFoundError(f"No S3 smoke render manifests found under {input_root}")

    columns = min(2, len(manifests))
    tile_width = 640
    image_height = 520
    header_height = 102
    footer_height = 54
    tile_height = header_height + image_height + footer_height
    gap = 24
    margin = 28
    rows = math.ceil(len(manifests) / columns)

    title_height = 96
    sheet_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    sheet_height = margin * 2 + title_height + rows * tile_height + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#0f2f24")
    draw = ImageDraw.Draw(sheet)
    title_font = _font("bold", 34)
    subtitle_font = _font("regular", 22)
    case_font = _font("bold", 26)
    note_font = _font("regular", 18)

    draw.text((margin, margin), "S3 smoke visual preview", font=title_font, fill="#f4ead7")
    draw.text(
        (margin, margin + 42),
        "Smoke-only live-data plumbing preview · not an approved fixture contact sheet",
        font=subtitle_font,
        fill="#cbbf9f",
    )

    rendered_cases: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        y = margin + title_height + row * (tile_height + gap)

        visual_id = str(manifest.get("visual_id") or f"s3_smoke_{index + 1}")
        output_png = Path(str(manifest["output_png"]))
        warnings = manifest.get("warnings", []) or []
        warning_text = " · ".join(str(warning) for warning in warnings) if warnings else "No render warnings"
        label = _manifest_label(manifest, visual_id)

        draw.rounded_rectangle((x, y, x + tile_width, y + tile_height), radius=20, fill="#173d30", outline="#f4ead7", width=2)
        draw.text((x + 18, y + 14), label, font=case_font, fill="#f4ead7")
        draw.text((x + 18, y + 48), visual_id, font=note_font, fill="#cbbf9f")
        draw.text((x + 18, y + 74), warning_text[:92], font=note_font, fill="#cbbf9f")

        image = Image.open(output_png).convert("RGB")
        thumb = _fit_thumbnail(image, tile_width - 28, image_height - 28)
        sheet.paste(thumb, (x + 14, y + header_height + 14))

        footer_y = y + header_height + image_height + 14
        draw.text((x + 18, footer_y), "Review-only · no publishing or approval", font=note_font, fill="#cbbf9f")

        rendered_cases.append(
            {
                "visual_id": visual_id,
                "label": label,
                "output_png": str(output_png),
                "warnings": warnings,
                "manifest_path": str(manifest.get("manifest_path", "")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")

    contact_manifest = {
        "success": True,
        "created_at": utc_now(),
        "review_only": True,
        "publishes_content": False,
        "input_root": str(input_root),
        "contact_sheet": str(output_path),
        "case_count": len(rendered_cases),
        "cases": rendered_cases,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(manifest_path, contact_manifest)
    return contact_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact sheet for mapped S3 smoke visual renders.")
    parser.add_argument("--input-root", default="generated_visuals/s3_smoke")
    parser.add_argument("--output", default="generated_visuals/s3_smoke/contact_sheet.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_s3_smoke_contact_sheet(args.input_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
