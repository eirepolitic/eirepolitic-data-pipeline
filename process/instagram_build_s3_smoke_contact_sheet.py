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
from process.instagram_render_s3_smoke_samples import SMOKE_SAMPLES, run_samples

LABELS = {
    "horizontal_bar_s3_debate_issues_draft_v1": "Debate issue counts · horizontal bar",
    "vertical_bar_s3_debate_issues_draft_v1": "Debate issue counts · vertical bar",
    "donut_chart_s3_member_parties_draft_v1": "Member party counts · donut",
    "horizontal_bar_s3_member_parties_draft_v1": "Member party counts · horizontal bar",
}

CANONICAL_OUTPUT_ROOTS = {sample["output_root"] for sample in SMOKE_SAMPLES}


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


def _is_canonical_manifest(manifest: dict[str, Any]) -> bool:
    manifest_path = str(manifest.get("manifest_path") or "")
    return any(root in manifest_path for root in CANONICAL_OUTPUT_ROOTS)


def _find_manifests(input_root: Path) -> list[dict[str, Any]]:
    candidates = sorted(input_root.glob("*/manifests/*.render_manifest.json"))
    manifests = [_load_manifest(path) for path in candidates]

    canonical_visual_ids = set(LABELS)
    matching = [manifest for manifest in manifests if str(manifest.get("visual_id") or "") in canonical_visual_ids]
    if matching:
        by_visual_id: dict[str, dict[str, Any]] = {}
        for manifest in matching:
            visual_id = str(manifest.get("visual_id") or "")
            existing = by_visual_id.get(visual_id)
            if existing is None or (_is_canonical_manifest(manifest) and not _is_canonical_manifest(existing)):
                by_visual_id[visual_id] = manifest
        manifests = list(by_visual_id.values())

    return sorted(manifests, key=lambda item: str(item.get("visual_id") or item.get("manifest_path") or ""))


def _manifest_label(manifest: dict[str, Any], fallback: str) -> str:
    visual_id = str(manifest.get("visual_id") or fallback)
    return LABELS.get(visual_id, visual_id.replace("_", " "))


def build_s3_smoke_contact_sheet(input_root: str | Path, output_path: str | Path, render_samples: bool = True) -> dict[str, Any]:
    input_root = Path(input_root)
    output_path = Path(output_path)

    sample_render_manifest: dict[str, Any] | None = None
    if render_samples:
        sample_render_manifest = run_samples(SMOKE_SAMPLES)
        sample_manifest_path = input_root / "smoke_samples.manifest.json"
        write_json(sample_manifest_path, sample_render_manifest)

    manifests = _find_manifests(input_root)
    if not manifests:
        raise FileNotFoundError(f"No S3 smoke render manifests found under {input_root}")

    columns = min(2, len(manifests))
    tile_width = 640
    image_height = 520
    header_height = 106
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
    case_font = _font("bold", 25)
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
        draw.text((x + 18, y + 14), label[:52], font=case_font, fill="#f4ead7")
        draw.text((x + 18, y + 49), visual_id[:72], font=note_font, fill="#cbbf9f")
        draw.text((x + 18, y + 76), warning_text[:92], font=note_font, fill="#cbbf9f")

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
        "sample_render_manifest": sample_render_manifest,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(manifest_path, contact_manifest)
    return contact_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact sheet for mapped S3 smoke visual renders.")
    parser.add_argument("--input-root", default="generated_visuals/s3_smoke")
    parser.add_argument("--output", default="generated_visuals/s3_smoke/contact_sheet.png")
    parser.add_argument("--skip-render-samples", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_s3_smoke_contact_sheet(args.input_root, args.output, render_samples=not args.skip_render_samples)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
