from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from instagram.visuals.renderers.common import write_json

PAGE_WIDTH = 2800
MARGIN = 70
GAP = 40
MAX_SINGLE_IMAGE_HEIGHT = 30000
SUMMARY_HEADER_HEIGHT = 210
SUMMARY_ROW_HEIGHT = 820
SUMMARY_METADATA_WIDTH = 650
SUMMARY_THUMBNAIL_WIDTH = 1900
SUMMARY_THUMBNAIL_HEIGHT = 700
AUDIT_HEADER_HEIGHT = 220
AUDIT_ROW_HEIGHT = 940
AUDIT_METADATA_WIDTH = 760
AUDIT_THUMBNAIL_WIDTH = 900
AUDIT_THUMBNAIL_HEIGHT = 790
LEGACY_ALIAS_SCENARIOS = {"minimum", "maximum"}


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
    for line in _wrapped_lines(text, width)[:max_lines]:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _thumbnail(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    canvas = Image.new("RGB", (width, height), "white")
    fitted = ImageOps.contain(image, (width - 20, height - 20))
    x = (width - fitted.width) // 2
    y = (height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_slide(manifest: dict[str, Any], slide_id: str) -> dict[str, Any] | None:
    for slide in manifest.get("slides") or []:
        if str(slide.get("slide_id")) == slide_id:
            return slide
    return None


def _metric_summary(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    labels = {
        "displayed_item_count": "bars",
        "longest_label_length": "longest label",
        "minimum_value": "min",
        "maximum_value": "max",
        "relative_spread": "spread",
        "positive_max_to_min_ratio": "max/min",
        "top_to_second_ratio": "top/second",
        "has_ties": "ties",
        "all_equal": "all equal",
        "has_zero": "zero",
    }
    parts: list[str] = []
    for key in labels:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            value = round(value, 2)
        parts.append(f"{labels[key]}: {value}")
    return " · ".join(parts)


def _summary_groups(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    rendered = [manifests[name] for name in scenario_order if name in manifests and manifests[name].get("status") == "rendered"]
    waived = [manifests[name] for name in scenario_order if name in manifests and manifests[name].get("status") == "waived"]

    cover_manifest = next((item for item in rendered if item.get("scenario") == "real_example" and _find_slide(item, "cover")), None)
    if cover_manifest is None:
        cover_manifest = next((item for item in rendered if _find_slide(item, "cover")), None)

    grouped: dict[str, dict[str, Any]] = {}
    for manifest in rendered:
        scenario = str(manifest.get("scenario") or "unknown")
        if scenario in LEGACY_ALIAS_SCENARIOS:
            continue
        slide = _find_slide(manifest, "issue_profile")
        if slide is None:
            visual_slides = [item for item in manifest.get("slides") or [] if str(item.get("slide_id")) != "cover"]
            slide = visual_slides[0] if visual_slides else None
        if slide is None:
            continue
        path = root / str(slide["path"])
        digest = _sha256(path)
        group = grouped.setdefault(
            digest,
            {
                "sha256": digest,
                "slide": slide,
                "scenarios": [],
                "sources": [],
                "selection_reasons": [],
                "metrics": manifest.get("scenario_metrics"),
            },
        )
        group["scenarios"].append(scenario)
        source = str(manifest.get("source_item_label") or "").strip()
        if source and source not in group["sources"]:
            group["sources"].append(source)
        reason = str(manifest.get("selection_reason") or "").strip()
        if reason and reason not in group["selection_reasons"]:
            group["selection_reasons"].append(reason)

    return cover_manifest, list(grouped.values()), waived


def _draw_badges(draw: ImageDraw.ImageDraw, x: int, y: int, badges: list[str], *, max_width: int) -> int:
    font = _font(24, bold=True)
    cursor_x = x
    cursor_y = y
    for badge in badges:
        label = badge.replace("_", " ").upper()
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0] + 28
        if cursor_x + width > x + max_width:
            cursor_x = x
            cursor_y += 50
        draw.rounded_rectangle((cursor_x, cursor_y, cursor_x + width, cursor_y + 38), radius=16, fill="#d8b45f")
        draw.text((cursor_x + 14, cursor_y + 7), label, font=font, fill="#173d30")
        cursor_x += width + 12
    return cursor_y + 50


def _draw_summary_cover(canvas: Image.Image, root: Path, manifest: dict[str, Any], y: int) -> None:
    draw = ImageDraw.Draw(canvas)
    row = (MARGIN, y, PAGE_WIDTH - MARGIN, y + SUMMARY_ROW_HEIGHT - 20)
    draw.rounded_rectangle(row, radius=28, fill="#f7f7f4", outline="#c8c8c2", width=3)
    draw.text((MARGIN + 30, y + 28), "COVER LAYOUT", font=_font(40, bold=True), fill="#173d30")
    source = str(manifest.get("source_item_label") or "Representative real example")
    draw.text((MARGIN + 30, y + 92), source, font=_font(30), fill="#333333")
    draw.text((MARGIN + 30, y + 146), "Shown once because chart stress scenarios do not change the cover layout.", font=_font(24), fill="#555555")
    slide = _find_slide(manifest, "cover")
    if slide:
        thumb = _thumbnail(root / str(slide["path"]), SUMMARY_THUMBNAIL_WIDTH, SUMMARY_THUMBNAIL_HEIGHT)
        x = MARGIN + SUMMARY_METADATA_WIDTH + GAP
        canvas.paste(thumb, (x, y + 70))
        draw.rounded_rectangle((x, y + 70, x + SUMMARY_THUMBNAIL_WIDTH, y + 70 + SUMMARY_THUMBNAIL_HEIGHT), radius=18, outline="#888888", width=3)


def _draw_summary_visual(canvas: Image.Image, root: Path, group: dict[str, Any], y: int) -> None:
    draw = ImageDraw.Draw(canvas)
    row = (MARGIN, y, PAGE_WIDTH - MARGIN, y + SUMMARY_ROW_HEIGHT - 20)
    draw.rounded_rectangle(row, radius=28, fill="#f7f7f4", outline="#c8c8c2", width=3)
    text_x = MARGIN + 30
    text_y = _draw_badges(draw, text_x, y + 28, list(group["scenarios"]), max_width=SUMMARY_METADATA_WIDTH - 70)
    sources = ", ".join(group.get("sources") or []) or "Unknown source"
    draw.text((text_x, text_y), "Source", font=_font(24, bold=True), fill="#202020")
    text_y = _draw_wrapped(draw, (text_x, text_y + 34), sources, font=_font(26), fill="#333333", width=35, line_height=34, max_lines=3) + 18
    metric_text = _metric_summary(group.get("metrics"))
    if metric_text:
        draw.text((text_x, text_y), "Key metrics", font=_font(24, bold=True), fill="#202020")
        text_y = _draw_wrapped(draw, (text_x, text_y + 34), metric_text, font=_font(24), fill="#333333", width=38, line_height=32, max_lines=5) + 18
    reasons = group.get("selection_reasons") or []
    if reasons:
        draw.text((text_x, text_y), "Selection", font=_font(24, bold=True), fill="#202020")
        _draw_wrapped(draw, (text_x, text_y + 34), reasons[0], font=_font(22), fill="#555555", width=42, line_height=30, max_lines=4)
    x = MARGIN + SUMMARY_METADATA_WIDTH + GAP
    slide = group["slide"]
    thumb = _thumbnail(root / str(slide["path"]), SUMMARY_THUMBNAIL_WIDTH, SUMMARY_THUMBNAIL_HEIGHT)
    canvas.paste(thumb, (x, y + 70))
    draw.rounded_rectangle((x, y + 70, x + SUMMARY_THUMBNAIL_WIDTH, y + 70 + SUMMARY_THUMBNAIL_HEIGHT), radius=18, outline="#888888", width=3)


def _draw_waiver_block(canvas: Image.Image, waived: list[dict[str, Any]], y: int) -> int:
    if not waived:
        return y
    height = 120 + len(waived) * 95
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((MARGIN, y, PAGE_WIDTH - MARGIN, y + height), radius=28, fill="#eee8d8", outline="#b89b55", width=3)
    draw.text((MARGIN + 30, y + 28), "WAIVED SCENARIOS", font=_font(38, bold=True), fill="#725416")
    line_y = y + 88
    for manifest in waived:
        scenario = str(manifest.get("scenario") or "unknown").replace("_", " ").upper()
        reason = str(manifest.get("waiver_reason") or "No reason recorded")
        draw.text((MARGIN + 35, line_y), scenario, font=_font(24, bold=True), fill="#725416")
        _draw_wrapped(draw, (MARGIN + 360, line_y), reason, font=_font(23), fill="#3b3423", width=120, line_height=30, max_lines=2)
        line_y += 95
    return y + height


def _build_summary_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    cover, groups, waived = _summary_groups(root, scenario_manifests, scenario_order)
    row_count = len(groups) + (1 if cover else 0)
    waiver_height = 120 + len(waived) * 95 if waived else 0
    height = SUMMARY_HEADER_HEIGHT + row_count * SUMMARY_ROW_HEIGHT + waiver_height + MARGIN
    canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 45), f"{project_id} validation summary", font=_font(58, bold=True), fill="#173d30")
    draw.text((MARGIN, 124), "Unique renders only · cover shown once · compact waivers · not for publication", font=_font(28), fill="#444444")
    y = SUMMARY_HEADER_HEIGHT
    if cover:
        _draw_summary_cover(canvas, root, cover, y)
        y += SUMMARY_ROW_HEIGHT
    for group in groups:
        _draw_summary_visual(canvas, root, group, y)
        y += SUMMARY_ROW_HEIGHT
    _draw_waiver_block(canvas, waived, y)
    filename = "validation_contact_sheet.png"
    canvas.save(root / filename, format="PNG", optimize=True)
    return {
        "pages": [filename],
        "unique_visual_count": len(groups),
        "cover_shown_once": bool(cover),
        "waived_scenario_count": len(waived),
        "render_groups": [
            {
                "sha256": group["sha256"],
                "scenarios": group["scenarios"],
                "sources": group["sources"],
                "slide_path": group["slide"]["path"],
            }
            for group in groups
        ],
        "waived_scenarios": [item.get("scenario") for item in waived],
    }


def _audit_metadata(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    output = [("Status", str(manifest.get("status") or "unknown").upper())]
    if manifest.get("source_item_label"):
        output.append(("Source", str(manifest["source_item_label"])))
    if manifest.get("selection_reason"):
        output.append(("Why selected", str(manifest["selection_reason"])))
    if manifest.get("waiver_reason"):
        output.append(("Why waived", str(manifest["waiver_reason"])))
    metrics = _metric_summary(manifest.get("scenario_metrics"))
    if metrics:
        output.append(("Metrics", metrics))
    return output


def _draw_audit_row(canvas: Image.Image, *, y: int, root: Path, manifest: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(canvas)
    row_box = (MARGIN, y, PAGE_WIDTH - MARGIN, y + AUDIT_ROW_HEIGHT - 25)
    draw.rounded_rectangle(row_box, radius=28, fill="#f7f7f4", outline="#c8c8c2", width=3)
    scenario = str(manifest.get("scenario") or "unknown")
    draw.text((MARGIN + 35, y + 30), scenario, font=_font(42, bold=True), fill="#173d30")
    text_y = y + 100
    for label, value in _audit_metadata(manifest):
        draw.text((MARGIN + 35, text_y), f"{label}:", font=_font(24, bold=True), fill="#202020")
        text_y = _draw_wrapped(draw, (MARGIN + 35, text_y + 34), value, font=_font(24), fill="#333333", width=47, line_height=31, max_lines=5) + 15
    if manifest.get("status") == "waived":
        panel_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
        panel_y = y + 80
        panel_w = PAGE_WIDTH - MARGIN - panel_x - 35
        panel_h = AUDIT_ROW_HEIGHT - 160
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=24, fill="#eee8d8", outline="#b89b55", width=4)
        draw.text((panel_x + panel_w // 2, panel_y + 185), "NO REAL QUALIFYING CASE", font=_font(46, bold=True), fill="#725416", anchor="mm")
        _draw_wrapped(draw, (panel_x + 90, panel_y + 275), str(manifest.get("waiver_reason") or "No reason recorded."), font=_font(32), fill="#3b3423", width=80, line_height=44, max_lines=8)
        return
    start_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
    for index, slide in enumerate((manifest.get("slides") or [])[:2]):
        thumb = _thumbnail(root / str(slide["path"]), AUDIT_THUMBNAIL_WIDTH, AUDIT_THUMBNAIL_HEIGHT)
        x = start_x + index * (AUDIT_THUMBNAIL_WIDTH + GAP)
        canvas.paste(thumb, (x, y + 95))
        draw.rounded_rectangle((x, y + 95, x + AUDIT_THUMBNAIL_WIDTH, y + 95 + AUDIT_THUMBNAIL_HEIGHT), radius=18, outline="#888888", width=3)


def _build_audit_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    ordered = [scenario_manifests[name] for name in scenario_order if name in scenario_manifests]
    rows_per_page = max(1, (MAX_SINGLE_IMAGE_HEIGHT - AUDIT_HEADER_HEIGHT) // AUDIT_ROW_HEIGHT)
    pages: list[str] = []
    for page_index, start in enumerate(range(0, len(ordered), rows_per_page), start=1):
        rows = ordered[start : start + rows_per_page]
        height = AUDIT_HEADER_HEIGHT + len(rows) * AUDIT_ROW_HEIGHT + MARGIN
        canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 55), f"{project_id} validation audit", font=_font(58, bold=True), fill="#173d30")
        draw.text((MARGIN, 132), "Complete scenario-by-scenario evidence · not for publication", font=_font(28), fill="#444444")
        for row_index, manifest in enumerate(rows):
            _draw_audit_row(canvas, y=AUDIT_HEADER_HEIGHT + row_index * AUDIT_ROW_HEIGHT, root=root, manifest=manifest)
        filename = "validation_audit_contact_sheet.png" if len(ordered) <= rows_per_page else f"validation_audit_contact_sheet_{page_index:02d}.png"
        canvas.save(root / filename, format="PNG", optimize=True)
        pages.append(filename)
    return {"pages": pages, "scenario_count": len(ordered), "rows_per_page": rows_per_page}


def build_validation_contact_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    summary = _build_summary_sheet(
        root=root,
        project_id=project_id,
        scenario_manifests=scenario_manifests,
        scenario_order=scenario_order,
    )
    audit = _build_audit_sheet(
        root=root,
        project_id=project_id,
        scenario_manifests=scenario_manifests,
        scenario_order=scenario_order,
    )
    manifest = {
        "project_id": project_id,
        "layout": "deduplicated_summary_plus_complete_audit",
        "scenario_count": len([name for name in scenario_order if name in scenario_manifests]),
        "summary": summary,
        "audit": audit,
        "pages": summary["pages"],
        "scenario_order": [name for name in scenario_order if name in scenario_manifests],
    }
    write_json(root / "validation_contact_sheet_manifest.json", manifest)
    return manifest
