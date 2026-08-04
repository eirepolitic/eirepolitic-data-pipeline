from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from instagram.visuals.renderers.common import write_json

PAGE_WIDTH = 2400
MARGIN = 60
GAP = 30
COLS = 3
CARD_WIDTH = 740
CARD_HEIGHT = 900
THUMB_WIDTH = 680
THUMB_HEIGHT = 720
HEADER_HEIGHT = 170
LEGACY_ALIASES = {"minimum", "maximum"}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _slide(manifest: dict[str, Any], slide_id: str) -> dict[str, Any] | None:
    for slide in manifest.get("slides") or []:
        if str(slide.get("slide_id")) == slide_id:
            return slide
    return None


def _visual_slide(manifest: dict[str, Any]) -> dict[str, Any] | None:
    preferred = _slide(manifest, "issue_profile")
    if preferred is not None:
        return preferred
    return next((slide for slide in manifest.get("slides") or [] if slide.get("slide_id") != "cover"), None)


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


def _draw_card(
    canvas: Image.Image,
    *,
    root: Path,
    x: int,
    y: int,
    label: str,
    source: str,
    slide: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (x, y, x + CARD_WIDTH, y + CARD_HEIGHT),
        radius=24,
        fill="#f7f7f4",
        outline="#b9b9b3",
        width=3,
    )
    draw.text((x + 24, y + 20), label.replace("_", " ").upper(), font=_font(28, bold=True), fill="#173d30")
    if source:
        draw.text((x + 24, y + 58), source[:46], font=_font(20), fill="#555555")
    thumb = _thumbnail(root / str(slide["path"]), THUMB_WIDTH, THUMB_HEIGHT)
    canvas.paste(thumb, (x + 30, y + 130))
    draw.rounded_rectangle(
        (x + 30, y + 130, x + 30 + THUMB_WIDTH, y + 130 + THUMB_HEIGHT),
        radius=16,
        outline="#777777",
        width=2,
    )


def _build_grid(
    *,
    root: Path,
    project_id: str,
    entries: list[dict[str, Any]],
    waived: list[dict[str, Any]],
    filename: str,
    title: str,
) -> dict[str, Any]:
    rows = max(1, (len(entries) + COLS - 1) // COLS)
    waiver_height = 110 + 72 * len(waived) if waived else 0
    height = HEADER_HEIGHT + rows * (CARD_HEIGHT + GAP) + waiver_height + MARGIN
    canvas = Image.new("RGB", (PAGE_WIDTH, height), "#e9ebe6")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 40), title, font=_font(48, bold=True), fill="#173d30")
    draw.text((MARGIN, 105), "Every defined scenario shown individually · not for publication", font=_font(24), fill="#444444")

    for index, entry in enumerate(entries):
        row = index // COLS
        col = index % COLS
        x = MARGIN + col * (CARD_WIDTH + GAP)
        y = HEADER_HEIGHT + row * (CARD_HEIGHT + GAP)
        _draw_card(
            canvas,
            root=root,
            x=x,
            y=y,
            label=str(entry["scenario"]),
            source=str(entry.get("source") or ""),
            slide=entry["slide"],
        )

    if waived:
        y = HEADER_HEIGHT + rows * (CARD_HEIGHT + GAP)
        draw.rounded_rectangle((MARGIN, y, PAGE_WIDTH - MARGIN, y + waiver_height - 20), radius=20, fill="#eee8d8", outline="#b89b55", width=3)
        draw.text((MARGIN + 24, y + 22), "WAIVED SCENARIOS", font=_font(30, bold=True), fill="#725416")
        line_y = y + 70
        for item in waived:
            scenario = str(item.get("scenario") or "unknown").replace("_", " ").upper()
            reason = str(item.get("waiver_reason") or "No qualifying real case")
            draw.text((MARGIN + 28, line_y), scenario, font=_font(22, bold=True), fill="#725416")
            draw.text((MARGIN + 360, line_y), reason[:140], font=_font(19), fill="#3b3423")
            line_y += 72

    canvas.save(root / filename, format="PNG")
    return {
        "pages": [filename],
        "visual_row_count": len(entries),
        "cover_shown_once": any(entry.get("scenario") == "cover" for entry in entries),
        "scenario_rows": [entry["scenario"] for entry in entries if entry.get("scenario") != "cover"],
        "waived_scenarios": [item.get("scenario") for item in waived],
    }


def build_validation_contact_sheet(
    *,
    root: Path,
    project_id: str,
    scenario_manifests: dict[str, dict[str, Any]],
    scenario_order: list[str],
) -> dict[str, Any]:
    rendered = [
        scenario_manifests[name]
        for name in scenario_order
        if name in scenario_manifests and scenario_manifests[name].get("status") == "rendered"
    ]
    waived = [
        scenario_manifests[name]
        for name in scenario_order
        if name in scenario_manifests
        and name not in LEGACY_ALIASES
        and scenario_manifests[name].get("status") == "waived"
    ]

    entries: list[dict[str, Any]] = []
    cover_manifest = next((item for item in rendered if item.get("scenario") == "real_example" and _slide(item, "cover")), None)
    cover_manifest = cover_manifest or next((item for item in rendered if _slide(item, "cover")), None)
    if cover_manifest:
        entries.append({
            "scenario": "cover",
            "source": cover_manifest.get("source_item_label"),
            "slide": _slide(cover_manifest, "cover"),
        })

    for name in scenario_order:
        if name in LEGACY_ALIASES:
            continue
        manifest = scenario_manifests.get(name)
        if not manifest or manifest.get("status") != "rendered":
            continue
        slide = _visual_slide(manifest)
        if slide is None:
            continue
        entries.append({
            "scenario": name,
            "source": manifest.get("source_item_label"),
            "slide": slide,
        })

    full = _build_grid(
        root=root,
        project_id=project_id,
        entries=entries,
        waived=waived,
        filename="validation_contact_sheet.png",
        title=f"{project_id} full validation contact sheet",
    )

    unique: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["scenario"] == "cover":
            unique["cover"] = entry
            continue
        digest = _sha256(root / str(entry["slide"]["path"]))
        unique.setdefault(digest, entry)
    summary_entries = list(unique.values())
    summary = _build_grid(
        root=root,
        project_id=project_id,
        entries=summary_entries,
        waived=waived,
        filename="validation_summary_contact_sheet.png",
        title=f"{project_id} deduplicated validation summary",
    )
    summary["unique_visual_count"] = len([entry for entry in summary_entries if entry["scenario"] != "cover"])
    summary["render_groups"] = []

    audit = _build_grid(
        root=root,
        project_id=project_id,
        entries=entries,
        waived=waived,
        filename="validation_audit_contact_sheet.png",
        title=f"{project_id} validation audit",
    )

    manifest = {
        "project_id": project_id,
        "layout": "compact_full_grid_plus_deduplicated_summary_plus_audit",
        "scenario_count": len([name for name in scenario_order if name in scenario_manifests]),
        "full": full,
        "summary": summary,
        "audit": audit,
        "pages": full["pages"],
        "scenario_order": [name for name in scenario_order if name in scenario_manifests],
    }
    write_json(root / "validation_contact_sheet_manifest.json", manifest)
    return manifest
