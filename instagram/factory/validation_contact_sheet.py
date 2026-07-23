from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from instagram.visuals.renderers.common import write_json

PAGE_WIDTH = 2800
HEADER_HEIGHT = 220
ROW_HEIGHT = 940
MARGIN = 70
METADATA_WIDTH = 760
GAP = 45
THUMBNAIL_WIDTH = 900
THUMBNAIL_HEIGHT = 790
MAX_SINGLE_IMAGE_HEIGHT = 30000


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _wrapped_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(str(text or ""), width=max(12, width), break_long_words=False) or [""]


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
    line_height: int,
    max_lines: int,
) -> int:
    x, y = xy
    lines = _wrapped_lines(text, width)[:max_lines]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    canvas = Image.new("RGB", (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), "white")
    fitted = ImageOps.contain(image, (THUMBNAIL_WIDTH - 20, THUMBNAIL_HEIGHT - 20))
    x = (THUMBNAIL_WIDTH - fitted.width) // 2
    y = (THUMBNAIL_HEIGHT - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _metadata_text(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    status = str(manifest.get("status") or "unknown").upper()
    output: list[tuple[str, str]] = [("Status", status)]
    if manifest.get("source_item_label"):
        output.append(("Source", str(manifest["source_item_label"])))
    if manifest.get("selection_reason"):
        output.append(("Why selected", str(manifest["selection_reason"])))
    if manifest.get("waiver_reason"):
        output.append(("Why waived", str(manifest["waiver_reason"])))
    metrics = manifest.get("scenario_metrics")
    if isinstance(metrics, dict):
        concise = []
        for key in (
            "displayed_item_count",
            "longest_label_length",
            "minimum_value",
            "maximum_value",
            "relative_spread",
            "positive_max_to_min_ratio",
            "top_to_second_ratio",
            "has_ties",
            "all_equal",
            "has_zero",
        ):
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                value = round(value, 3)
            concise.append(f"{key}={value}")
        if concise:
            output.append(("Metrics", ", ".join(concise)))
    return output


def _draw_scenario_row(
    canvas: Image.Image,
    *,
    y: int,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    label_font = _font(24, bold=True)
    body_font = _font(24)
    status_font = _font(30, bold=True)

    row_box = (MARGIN, y, PAGE_WIDTH - MARGIN, y + ROW_HEIGHT - 25)
    draw.rounded_rectangle(row_box, radius=28, fill="#f7f7f4", outline="#c8c8c2", width=3)

    scenario = str(manifest.get("scenario") or "unknown")
    draw.text((MARGIN + 35, y + 30), scenario, font=title_font, fill="#173d30")

    status = str(manifest.get("status") or "unknown").upper()
    draw.text((MARGIN + 35, y + 95), status, font=status_font, fill="#725416")

    text_y = y + 155
    for label, value in _metadata_text(manifest):
        draw.text((MARGIN + 35, text_y), f"{label}:", font=label_font, fill="#202020")
        text_y = _draw_wrapped(
            draw,
            (MARGIN + 35, text_y + 34),
            value,
            font=body_font,
            fill="#333333",
            width=47,
            line_height=31,
            max_lines=5 if label in {"Why selected", "Why waived"} else 4,
        ) + 15

    if status == "WAIVED":
        panel_x = MARGIN + METADATA_WIDTH + GAP
        panel_y = y + 80
        panel_w = PAGE_WIDTH - MARGIN - panel_x - 35
        panel_h = ROW_HEIGHT - 160
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=24,
            fill="#eee8d8",
            outline="#b89b55",
            width=4,
        )
        draw.text((panel_x + panel_w // 2, panel_y + 185), "NO REAL QUALIFYING CASE", font=_font(46, bold=True), fill="#725416", anchor="mm")
        _draw_wrapped(
            draw,
            (panel_x + 90, panel_y + 275),
            str(manifest.get("waiver_reason") or "No reason recorded."),
            font=_font(32),
            fill="#3b3423",
            width=80,
            line_height=44,
            max_lines=8,
        )
        return

    slides = manifest.get("slides") or []
    start_x = MARGIN + METADATA_WIDTH + GAP
    for index, slide in enumerate(slides[:2]):
        slide_path = root / str(slide["path"])
        thumb = _thumbnail(slide_path)
        x = start_x + index * (THUMBNAIL_WIDTH + GAP)
        canvas.paste(thumb, (x, y + 95))
        draw.rounded_rectangle(
            (x, y + 95, x + THUMBNAIL_WIDTH, y + 95 + THUMBNAIL_HEIGHT),
            radius=18,
            outline="#888888",
            width=3,
        )
        draw.text(
            (x + THUMBNAIL_WIDTH // 2, y + ROW_HEIGHT - 42),
            str(slide.get("slide_id") or f"slide_{index + 1}"),
            font=label_font,
            fill="#333333",
            anchor="mm",
        )


def build_validation_contact_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    ordered = [scenario_manifests[name] for name in scenario_order if name in scenario_manifests]
    rows_per_page = max(1, (MAX_SINGLE_IMAGE_HEIGHT - HEADER_HEIGHT) // ROW_HEIGHT)
    pages: list[str] = []

    for page_index, start in enumerate(range(0, len(ordered), rows_per_page), start=1):
        page_rows = ordered[start : start + rows_per_page]
        height = HEADER_HEIGHT + len(page_rows) * ROW_HEIGHT + MARGIN
        canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 55), f"{project_id} validation review", font=_font(58, bold=True), fill="#173d30")
        draw.text(
            (MARGIN, 132),
            "Large-tile review sheet · real data unless explicitly marked otherwise · not for publication",
            font=_font(28),
            fill="#444444",
        )
        for row_index, manifest in enumerate(page_rows):
            _draw_scenario_row(
                canvas,
                y=HEADER_HEIGHT + row_index * ROW_HEIGHT,
                root=root,
                manifest=manifest,
            )

        filename = "validation_contact_sheet.png" if len(ordered) <= rows_per_page else f"validation_contact_sheet_{page_index:02d}.png"
        output_path = root / filename
        canvas.save(output_path, format="PNG", optimize=True)
        pages.append(filename)

    manifest = {
        "project_id": project_id,
        "layout": "large_readable_tiles",
        "scenario_count": len(ordered),
        "rows_per_page": rows_per_page,
        "pages": pages,
        "scenario_order": [manifest.get("scenario") for manifest in ordered],
    }
    write_json(root / "validation_contact_sheet_manifest.json", manifest)
    return manifest
