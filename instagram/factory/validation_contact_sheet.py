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
WAIVER_CARD_HEIGHT = 360
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
    wrapped = _wrapped_lines(text, width)
    lines = wrapped[:max_lines]
    for index, line in enumerate(lines):
        if index == max_lines - 1 and len(wrapped) > max_lines:
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
    return next((slide for slide in manifest.get("slides") or [] if str(slide.get("slide_id")) != "cover"), None)


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _primary_metric(scenario: str, metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    count = int(metrics.get("displayed_item_count", 0) or 0)
    longest = int(metrics.get("longest_label_length", 0) or 0)
    minimum = metrics.get("minimum_value")
    maximum = metrics.get("maximum_value")
    spread = metrics.get("relative_spread")
    max_min = metrics.get("positive_max_to_min_ratio")
    top_second = metrics.get("top_to_second_ratio")
    if scenario in {"item_count_min", "item_count_max"}:
        return f"{count} displayed bars"
    if scenario in {"labels_short", "labels_long"}:
        return f"Longest label: {longest} chars"
    if scenario in {"values_small", "values_large"}:
        return f"Maximum value: {_number(maximum)}"
    if scenario == "values_tight" and spread is not None:
        return f"Relative spread: {float(spread) * 100:.1f}%"
    if scenario == "values_wide" and max_min is not None:
        return f"Max / min: {float(max_min):.1f}×"
    if scenario == "single_outlier" and top_second is not None:
        return f"Top / second: {float(top_second):.1f}×"
    if scenario == "all_equal":
        return f"All equal · {count} bars"
    if scenario == "ties":
        return f"Ties present · {count} bars"
    if scenario == "zeros":
        return f"Includes zero · {count} bars"
    if scenario == "real_example":
        return f"Representative · {count} bars"
    if minimum is not None and maximum is not None:
        return f"{count} bars · {_number(minimum)}–{_number(maximum)}"
    return f"{count} bars" if count else ""


def _secondary_metric(scenario: str, metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    count = int(metrics.get("displayed_item_count", 0) or 0)
    if scenario not in {"item_count_min", "item_count_max", "all_equal", "ties", "zeros", "real_example"} and count:
        return f"{count} bars"
    return ""


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
        parts.append(f"range {_number(minimum)}–{_number(maximum)}")
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


def _draw_review_card(canvas: Image.Image, *, root: Path, x: int, y: int, entry: dict[str, Any], card_width: int = CARD_WIDTH) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + card_width, y + CARD_HEIGHT), radius=26, fill="#f7f7f4", outline="#b9b9b3", width=3)
    badge_height = _draw_badge(draw, x + 24, y + 20, _badge_label(list(entry["scenarios"])), card_width - 48)
    text_y = y + 20 + badge_height + 12
    source = str(entry.get("source") or "Representative real example")
    draw.text((x + 24, text_y), textwrap.shorten(source, width=58 if card_width == CARD_WIDTH else 120, placeholder="…"), font=_font(24, bold=True), fill="#28342f")
    text_y += 38
    scenario = str((entry.get("scenarios") or [""])[0])
    primary = _primary_metric(scenario, entry.get("metrics"))
    if primary:
        draw.text((x + 24, text_y), primary, font=_font(31, bold=True), fill="#173d30")
        text_y += 43
    secondary = _secondary_metric(scenario, entry.get("metrics"))
    if secondary:
        draw.text((x + 24, text_y), secondary, font=_font(21), fill="#666666")
    preview_x = x + (card_width - THUMB_WIDTH) // 2
    preview_y = y + CARD_HEIGHT - THUMB_HEIGHT - 24
    thumb = _thumbnail(root / str(entry["slide"]["path"]), THUMB_WIDTH, THUMB_HEIGHT)
    canvas.paste(thumb, (preview_x, preview_y))
    draw.rounded_rectangle((preview_x, preview_y, preview_x + THUMB_WIDTH, preview_y + THUMB_HEIGHT), radius=16, outline="#777777", width=2)


def _draw_waiver_card(canvas: Image.Image, *, x: int, y: int, entry: dict[str, Any], card_width: int = CARD_WIDTH) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + card_width, y + WAIVER_CARD_HEIGHT), radius=26, fill="#eee8d8", outline="#b89b55", width=3)
    badge_height = _draw_badge(draw, x + 24, y + 24, _badge_label(list(entry["scenarios"])), card_width - 48)
    draw.text((x + 24, y + 24 + badge_height + 18), "NO REAL QUALIFYING CASE", font=_font(31, bold=True), fill="#725416")
    _draw_wrapped(
        draw,
        (x + 24, y + 24 + badge_height + 70),
        str(entry.get("waiver_reason") or "No qualifying real case was found."),
        font=_font(24),
        fill="#3b3423",
        width=62 if card_width == CARD_WIDTH else 128,
        line_height=34,
        max_lines=4,
    )


def _layout_grid(entries: list[dict[str, Any]]) -> list[tuple[int, int, int, dict[str, Any]]]:
    placements: list[tuple[int, int, int, dict[str, Any]]] = []
    row = 0
    col = 0
    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        span = 2 if is_last and col == 0 and entry.get("kind") == "waiver" else 1
        placements.append((row, col, span, entry))
        if span == 2 or col == 1:
            row += 1
            col = 0
        else:
            col = 1
    return placements


def _row_heights(placements: list[tuple[int, int, int, dict[str, Any]]]) -> dict[int, int]:
    heights: dict[int, int] = {}
    for row, _, _, entry in placements:
        entry_height = WAIVER_CARD_HEIGHT if entry.get("kind") == "waiver" else CARD_HEIGHT
        heights[row] = max(heights.get(row, 0), entry_height)
    return heights


def _row_offsets(row_heights: dict[int, int]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    cursor = HEADER_HEIGHT
    for row in sorted(row_heights):
        offsets[row] = cursor
        cursor += row_heights[row] + GAP
    return offsets


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
    waiver_entries = [{"kind": "waiver", "scenarios": [str(item.get("scenario") or "unknown")], "waiver_reason": item.get("waiver_reason")} for item in waived]
    grid_entries = [*entries, *waiver_entries]
    placements = _layout_grid(grid_entries)
    row_heights = _row_heights(placements)
    row_offsets = _row_offsets(row_heights)
    content_height = sum(row_heights.values()) + max(0, len(row_heights) - 1) * GAP
    height = HEADER_HEIGHT + content_height + MARGIN
    canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 38), title, font=_font(50, bold=True), fill="#173d30")
    draw.text((MARGIN, 105), subtitle, font=_font(25), fill="#444444")
    for row, col, span, entry in placements:
        x = MARGIN + col * (CARD_WIDTH + GAP)
        y = row_offsets[row]
        width = CARD_WIDTH if span == 1 else (2 * CARD_WIDTH + GAP)
        if entry.get("kind") == "waiver":
            _draw_waiver_card(canvas, x=x, y=y, entry=entry, card_width=width)
        else:
            _draw_review_card(canvas, root=root, x=x, y=y, entry=entry, card_width=width)
    canvas.save(root / filename, format="PNG", optimize=True)
    return {
        "pages": [filename],
        "columns": COLS,
        "card_width": CARD_WIDTH,
        "card_height": CARD_HEIGHT,
        "waiver_card_height": WAIVER_CARD_HEIGHT,
        "preview_width": THUMB_WIDTH,
        "preview_height": THUMB_HEIGHT,
        "metric_first": True,
        "waivers_inline": True,
        "waivers_compact": True,
        "cover_metadata_compact": True,
        "waiver_card_count": len(waived),
        "visual_row_count": len([entry for entry in entries if entry.get("kind") != "cover"]),
        "cover_shown_once": any(entry.get("kind") == "cover" for entry in entries),
        "scenario_rows": [scenario for entry in entries if entry.get("kind") != "cover" for scenario in entry["scenarios"]],
        "waived_scenarios": [item.get("scenario") for item in waived],
    }


def _full_entries(scenario_manifests: dict[str, dict[str, Any]], scenario_order: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rendered = [scenario_manifests[name] for name in scenario_order if name in scenario_manifests and scenario_manifests[name].get("status") == "rendered"]
    waived = [scenario_manifests[name] for name in scenario_order if name in scenario_manifests and name not in LEGACY_ALIAS_SCENARIOS and scenario_manifests[name].get("status") == "waived"]
    entries: list[dict[str, Any]] = []
    cover_manifest = next((item for item in rendered if item.get("scenario") == "real_example" and _find_slide(item, "cover")), None)
    cover_manifest = cover_manifest or next((item for item in rendered if _find_slide(item, "cover")), None)
    if cover_manifest:
        entries.append({"kind": "cover", "scenarios": ["cover layout"], "source": cover_manifest.get("source_item_label"), "metrics": None, "slide": _find_slide(cover_manifest, "cover")})
    for name in scenario_order:
        if name in LEGACY_ALIAS_SCENARIOS:
            continue
        manifest = scenario_manifests.get(name)
        if not manifest or manifest.get("status") != "rendered":
            continue
        slide = _visual_slide(manifest)
        if slide is None:
            continue
        entries.append({"kind": "visual", "scenarios": [name], "source": manifest.get("source_item_label"), "metrics": manifest.get("scenario_metrics"), "slide": slide})
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
        groups.append({"sha256": entry.get("sha256"), "scenarios": entry["scenarios"], "sources": [entry.get("source")], "slide_path": entry["slide"]["path"]})
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
    draw.rounded_rectangle((MARGIN, y, PAGE_WIDTH - MARGIN, y + AUDIT_ROW_HEIGHT - 22), radius=24, fill="#f7f7f4", outline="#c8c8c2", width=3)
    scenario = str(manifest.get("scenario") or "unknown").replace("_", " ").upper()
    draw.text((MARGIN + 28, y + 26), scenario, font=_font(36, bold=True), fill="#173d30")
    text_y = y + 84
    for label, value in _audit_metadata(manifest):
        draw.text((MARGIN + 28, text_y), f"{label}:", font=_font(21, bold=True), fill="#202020")
        text_y = _draw_wrapped(draw, (MARGIN + 28, text_y + 28), value, font=_font(20), fill="#333333", width=46, line_height=27, max_lines=4) + 10
    if manifest.get("status") == "waived":
        panel_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
        panel_w = PAGE_WIDTH - MARGIN - panel_x - 25
        draw.rounded_rectangle((panel_x, y + 70, panel_x + panel_w, y + AUDIT_ROW_HEIGHT - 70), radius=22, fill="#eee8d8", outline="#b89b55", width=3)
        draw.text((panel_x + panel_w // 2, y + 240), "NO REAL QUALIFYING CASE", font=_font(42, bold=True), fill="#725416", anchor="mm")
        return
    start_x = MARGIN + AUDIT_METADATA_WIDTH + GAP
    for index, slide in enumerate((manifest.get("slides") or [])[:2]):
        thumb = _thumbnail(root / str(slide["path"]), AUDIT_THUMBNAIL_WIDTH, AUDIT_THUMBNAIL_HEIGHT)
        x = start_x + index * (AUDIT_THUMBNAIL_WIDTH + GAP)
        canvas.paste(thumb, (x, y + 100))
        draw.rounded_rectangle((x, y + 100, x + AUDIT_THUMBNAIL_WIDTH, y + 100 + AUDIT_THUMBNAIL_HEIGHT), radius=16, outline="#888888", width=2)


def _build_audit_sheet(*, root: Path, project_id: str, scenario_manifests: dict[str, dict[str, Any]], scenario_order: list[str]) -> dict[str, Any]:
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
        filename = "validation_audit_contact_sheet.png" if len(ordered) <= rows_per_page else f"validation_audit_contact_sheet_{page_index:02d}.png"
        canvas.save(root / filename, format="PNG", optimize=True)
        pages.append(filename)
    return {"pages": pages, "scenario_count": len(ordered), "rows_per_page": rows_per_page}


def build_validation_contact_sheet(*, root: Path, project_id: str, scenario_manifests: dict[str, dict[str, Any]], scenario_order: list[str]) -> dict[str, Any]:
    entries, waived = _full_entries(scenario_manifests, scenario_order)
    full = _build_grid_sheet(
        root=root,
        project_id=project_id,
        entries=entries,
        waived=waived,
        filename="validation_contact_sheet.png",
        title=f"{project_id} validation contact sheet",
        subtitle="Metric-first two-column review · large previews · compact inline waivers · every defined scenario shown once",
    )
    summary_entries, render_groups = _summary_entries(root, entries)
    summary = _build_grid_sheet(
        root=root,
        project_id=project_id,
        entries=summary_entries,
        waived=waived,
        filename="validation_summary_contact_sheet.png",
        title=f"{project_id} deduplicated validation summary",
        subtitle="Metric-first two-column review · unique renders only · compact inline waivers",
    )
    summary["unique_visual_count"] = len([entry for entry in summary_entries if entry.get("kind") != "cover"])
    summary["render_groups"] = render_groups
    audit = _build_audit_sheet(root=root, project_id=project_id, scenario_manifests=scenario_manifests, scenario_order=scenario_order)
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
