from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from instagram.visuals.renderers.common import write_json

PAGE_WIDTH = 2400
MARGIN = 50
GAP = 30
COLS = 2
CARD_WIDTH = (PAGE_WIDTH - (2 * MARGIN) - GAP) // COLS
CARD_HEIGHT = 1180
THUMB_WIDTH = 760
THUMB_HEIGHT = 950
HEADER_HEIGHT = 180
MAX_SINGLE_IMAGE_HEIGHT = 30000
AUDIT_ROW_HEIGHT = 900
AUDIT_METADATA_WIDTH = 610
AUDIT_THUMBNAIL_WIDTH = 760
AUDIT_THUMBNAIL_HEIGHT = 700
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
    lines = _wrapped_lines(text, width)[:max_lines]
    for index, line in enumerate(lines):
        if index == max_lines - 1 and len(_wrapped_lines(text, width)) > max_lines:
            line = line.rstrip(" .") + "…"
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _thumbnail(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    canvas = Image.new("RGB", (width, height), "white")
    fitted = ImageOps.contain(image, (width - 16, height - 16))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
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


def _visual_slide(manifest: dict[str, Any]) -> dict[str, Any] | None:
    preferred = _find_slide(manifest, "issue_profile")
    if preferred is not None:
        return preferred
    return next(
        (slide for slide in manifest.get("slides") or [] if str(slide.get("slide_id")) != "cover"),
        None,
    )


def _metric_line(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    parts: list[str] = []
    count = metrics.get("displayed_item_count")
    if count is not None:
        parts.append(f"{int(count)} bars")
    longest = metrics.get("longest_label_length")
    if longest is not None:
        parts.append(f"longest label {int(longest)} chars")
    minimum = metrics.get("minimum_value")
    maximum = metrics.get("maximum_value")
    if minimum is not None and maximum is not None:
        parts.append(f"range {minimum:g}–{maximum:g}")
    if metrics.get("has_ties"):
        parts.append("ties")
    if metrics.get("all_equal"):
        parts.append("all equal")
    if metrics.get("has_zero"):
        parts.append("includes zero")
    return " · ".join(parts)


def _badge_label(scenarios: list[str]) -> str:
    return " + ".join(str(value).replace("_", " ").upper() for value in scenarios)


def _draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, max_width: int) -> int:
    font = _font(27, bold=True)
    lines = _wrapped_lines(text, 34)[:2]
    label = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=4)
    width = min(max_width, bbox[2] - bbox[0] + 32)
    height = bbox[3] - bbox[1] + 20
    draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill="#d8b45f")
    draw.multiline_text((x + 16, y + 8), label, font=font, fill="#173d30", spacing=4)
    return height


def _draw_review_card(
    canvas: Image.Image,
    *,
    root: Path,
    x: int,
    y: int,
    entry: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (x, y, x + CARD_WIDTH, y + CARD_HEIGHT),
        radius=26,
        fill="#f7f7f4",
        outline="#b9b9b3",
        width=3,
    )

    badge_height = _draw_badge(
        draw,
        x + 24,
        y + 20,
        _badge_label(list(entry["scenarios"])),
        CARD_WIDTH - 48,
    )
    text_y = y + 20 + badge_height + 10
    source = str(entry.get("source") or "Representative real example")
    draw.text((x + 24, text_y), textwrap.shorten(source, width=62, placeholder="…"), font=_font(25, bold=True), fill="#28342f")
    text_y += 38

    metric_line = _metric_line(entry.get("metrics"))
    if metric_line:
        draw.text((x + 24, text_y), textwrap.shorten(metric_line, width=82, placeholder="…"), font=_font(21), fill="#555555")
        text_y += 33

    reason = str(entry.get("selection_reason") or "")
    if reason:
        _draw_wrapped(
            draw,
            (x + 24, text_y),
            reason,
            font=_font(20),
            fill="#666666",
            width=88,
            line_height=28,
            max_lines=2,
        )

    preview_x = x + (CARD_WIDTH - THUMB_WIDTH) // 2
    preview_y = y + CARD_HEIGHT - THUMB_HEIGHT - 24
    thumb = _thumbnail(root / str(entry["slide"]["path"]), THUMB_WIDTH, THUMB_HEIGHT)
    canvas.paste(thumb, (preview_x, preview_y))
    draw.rounded_rectangle(
        (preview_x, preview_y, preview_x + THUMB_WIDTH, preview_y + THUMB_HEIGHT),
        radius=16,
        outline="#777777",
        width=2,
    )


def _waiver_height(waived: list[dict[str, Any]]) -> int:
    return 0 if not waived else 100 + 70 * len(waived)


def _draw_waivers(canvas: Image.Image, waived: list[dict[str, Any]], y: int) -> None:
    if not waived:
        return
    draw = ImageDraw.Draw(canvas)
    height = _waiver_height(waived)
    draw.rounded_rectangle(
        (MARGIN, y, PAGE_WIDTH - MARGIN, y + height - 20),
        radius=22,
        fill="#eee8d8",
        outline="#b89b55",
        width=3,
    )
    draw.text((MARGIN + 24, y + 20), "WAIVED SCENARIOS", font=_font(31, bold=True), fill="#725416")
    line_y = y + 68
    for manifest in waived:
        scenario = str(manifest.get("scenario") or "unknown").replace("_", " ").upper()
        reason = str(manifest.get("waiver_reason") or "No qualifying real case")
        draw.text((MARGIN + 28, line_y), scenario, font=_font(21, bold=True), fill="#725416")
        draw.text(
            (MARGIN + 330, line_y),
            textwrap.shorten(reason, width=175, placeholder="…"),
            font=_font(19),
            fill="#3b3423",
        )
        line_y += 70


def _build_grid_sheet(
    *,
    root: Path,
    project_id: str,
    entries: list[dict[str, Any]],
    waived: list[dict[str, Any]],
    filename: str,
    title: str,
    subtitle: str,
) -> dict[str, Any]:
    rows = max(1, (len(entries) + COLS - 1) // COLS)
    height = HEADER_HEIGHT + rows * (CARD_HEIGHT + GAP) + _waiver_height(waived) + MARGIN
    canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 38), title, font=_font(50, bold=True), fill="#173d30")
    draw.text((MARGIN, 105), subtitle, font=_font(25), fill="#444444")

    for index, entry in enumerate(entries):
        row = index // COLS
        col = index % COLS
        x = MARGIN + col * (CARD_WIDTH + GAP)
        y = HEADER_HEIGHT + row * (CARD_HEIGHT + GAP)
        _draw_review_card(canvas, root=root, x=x, y=y, entry=entry)

    waiver_y = HEADER_HEIGHT + rows * (CARD_HEIGHT + GAP)
    _draw_waivers(canvas, waived, waiver_y)
    canvas.save(root / filename, format="PNG", optimize=True)
    return {
        "pages": [filename],
        "columns": COLS,
        "card_width": CARD_WIDTH,
        "card_height": CARD_HEIGHT,
        "preview_width": THUMB_WIDTH,
        "preview_height": THUMB_HEIGHT,
        "visual_row_count": len([entry for entry in entries if entry.get("kind") != "cover"]),
        "cover_shown_once": any(entry.get("kind") == "cover" for entry in entries),
        "scenario_rows": [
            scenario
            for entry in entries
            if entry.get("kind") != "cover"
            for scenario in entry["scenarios"]
        ],
        "waived_scenarios": [item.get("scenario") for item in waived],
    }


def _full_entries(
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rendered = [
        scenario_manifests[name]
        for name in scenario_order
        if name in scenario_manifests and scenario_manifests[name].get("status") == "rendered"
    ]
    waived = [
        scenario_manifests[name]
        for name in scenario_order
        if name in scenario_manifests
        and name not in LEGACY_ALIAS_SCENARIOS
        and scenario_manifests[name].get("status") == "waived"
    ]

    entries: list[dict[str, Any]] = []
    cover_manifest = next(
        (item for item in rendered if item.get("scenario") == "real_example" and _find_slide(item, "cover")),
        None,
    )
    cover_manifest = cover_manifest or next((item for item in rendered if _find_slide(item, "cover")), None)
    if cover_manifest:
        entries.append({
            "kind": "cover",
            "scenarios": ["cover layout"],
            "source": cover_manifest.get("source_item_label"),
            "selection_reason": "Representative cover shown once; chart scenarios do not change the cover layout.",
            "metrics": None,
            "slide": _find_slide(cover_manifest, "cover"),
        })

    for name in scenario_order:
        if name in LEGACY_ALIAS_SCENARIOS:
            continue
        manifest = scenario_manifests.get(name)
        if not manifest or manifest.get("status") != "rendered":
            continue
        slide = _visual_slide(manifest)
        if slide is None:
            continue
        entries.append({
            "kind": "visual",
            "scenarios": [name],
            "source": manifest.get("source_item_label"),
            "selection_reason": manifest.get("selection_reason"),
            "metrics": manifest.get("scenario_metrics"),
            "slide": slide,
        })
    return entries, waived


def _summary_entries(root: Path, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") == "cover":
            unique["cover"] = dict(entry)
            continue
        digest = _sha256(root / str(entry["slide"]["path"]))
        if digest not in unique:
            unique[digest] = dict(entry)
            unique[digest]["sha256"] = digest
        else:
            unique[digest]["scenarios"].extend(entry["scenarios"])
    summary = list(unique.values())
    for entry in summary:
        if entry.get("kind") == "cover":
            continue
        groups.append({
            "sha256": entry.get("sha256"),
            "scenarios": entry["scenarios"],
            "sources": [entry.get("source")],
            "slide_path": entry["slide"]["path"],
        })
    return summary, groups


def _audit_metadata(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    output = [("Status", str(manifest.get("status") or "unknown").upper())]
    if manifest.get("source_item_label"):
        output.append(("Source", str(manifest["source_item_label"])))
    if manifest.get("selection_reason"):
        output.append(("Why selected", str(manifest["selection_reason"])))
    if manifest.get("waiver_reason"):
        output.append(("Why waived", str(manifest["waiver_reason"])))
    metrics = _metric_line(manifest.get("scenario_metrics"))
    if metrics:
        output.append(("Metrics", metrics))
    return output


def _draw_audit_row(canvas: Image.Image, *, y: int, root: Path, manifest: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (MARGIN, y, PAGE_WIDTH - MARGIN, y + AUDIT_ROW_HEIGHT - 22),
        radius=24,
        fill="#f7f7f4",
        outline="#c8c8c2",
        width=3,
    )
    scenario = str(manifest.get("scenario") or "unknown").replace("_", " ").upper()
    draw.text((MARGIN + 28, y + 26), scenario, font=_font(36, bold=True), fill="#173d30")
    text_y = y + 84
    for label, value in _audit_metadata(manifest):
        draw.text((MARGIN + 28, text_y), f"{label}:", font=_font(21, bold=True), fill="#202020")
        text_y = _draw_wrapped(
            draw,
            (MARGIN + 28, text_y + 28),
            value,
            font=_font(20),
            fill="#333333",
            width=46,
            line_height=27,
            max_lines=4,
        ) + 10

    if manifest.get("status") == "waived":
        panel_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
        panel_w = PAGE_WIDTH - MARGIN - panel_x - 25
        draw.rounded_rectangle(
            (panel_x, y + 70, panel_x + panel_w, y + AUDIT_ROW_HEIGHT - 70),
            radius=22,
            fill="#eee8d8",
            outline="#b89b55",
            width=3,
        )
        draw.text(
            (panel_x + panel_w // 2, y + 240),
            "NO REAL QUALIFYING CASE",
            font=_font(42, bold=True),
            fill="#725416",
            anchor="mm",
        )
        return

    start_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
    for index, slide in enumerate((manifest.get("slides") or [])[:2]):
        thumb = _thumbnail(root / str(slide["path"]), AUDIT_THUMBNAIL_WIDTH, AUDIT_THUMBNAIL_HEIGHT)
        x = start_x + index * (AUDIT_THUMBNAIL_WIDTH + GAP)
        canvas.paste(thumb, (x, y + 100))
        draw.rounded_rectangle(
            (x, y + 100, x + AUDIT_THUMBNAIL_WIDTH, y + 100 + AUDIT_THUMBNAIL_HEIGHT),
            radius=16,
            outline="#888888",
            width=2,
        )


def _build_audit_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    ordered = [scenario_manifests[name] for name in scenario_order if name in scenario_manifests]
    rows_per_page = max(1, (MAX_SINGLE_IMAGE_HEIGHT - HEADER_HEIGHT) // AUDIT_ROW_HEIGHT)
    pages: list[str] = []
    for page_index, start in enumerate(range(0, len(ordered), rows_per_page), start=1):
        rows = ordered[start : start + rows_per_page]
        height = HEADER_HEIGHT + len(rows) * AUDIT_ROW_HEIGHT + MARGIN
        canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 38), f"{project_id} validation audit", font=_font(50, bold=True), fill="#173d30")
        draw.text((MARGIN, 105), "Complete scenario evidence · not for publication", font=_font(25), fill="#444444")
        for row_index, manifest in enumerate(rows):
            _draw_audit_row(canvas, y=HEADER_HEIGHT + row_index * AUDIT_ROW_HEIGHT, root=root, manifest=manifest)
        filename = (
            "validation_audit_contact_sheet.png"
            if len(ordered) <= rows_per_page
            else f"validation_audit_contact_sheet_{page_index:02d}.png"
        )
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
    entries, waived = _full_entries(scenario_manifests, scenario_order)
    full = _build_grid_sheet(
        root=root,
        project_id=project_id,
        entries=entries,
        waived=waived,
        filename="validation_contact_sheet.png",
        title=f"{project_id} validation contact sheet",
        subtitle="Two-column review grid · larger previews · concise metadata · every defined scenario shown once",
    )

    summary_entries, render_groups = _summary_entries(root, entries)
    summary = _build_grid_sheet(
        root=root,
        project_id=project_id,
        entries=summary_entries,
        waived=waived,
        filename="validation_summary_contact_sheet.png",
        title=f"{project_id} deduplicated validation summary",
        subtitle="Two-column review grid · unique renders only · concise metadata",
    )
    summary["unique_visual_count"] = len([entry for entry in summary_entries if entry.get("kind") != "cover"])
    summary["render_groups"] = render_groups

    audit = _build_audit_sheet(
        root=root,
        project_id=project_id,
        scenario_manifests=scenario_manifests,
        scenario_order=scenario_order,
    )

    manifest = {
        "project_id": project_id,
        "layout": "two_column_full_review_plus_deduplicated_summary_plus_complete_audit",
        "scenario_count": len([name for name in scenario_order if name in scenario_manifests]),
        "full": full,
        "summary": summary,
        "audit": audit,
        "pages": full["pages"],
        "scenario_order": [name for name in scenario_order if name in scenario_manifests],
    }
    write_json(root / "validation_contact_sheet_manifest.json", manifest)
    return manifest
