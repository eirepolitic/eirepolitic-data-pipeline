from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from instagram.factory.render_primitives import (
    ACCENT,
    BG,
    MUTED,
    TEXT,
    draw_glossary,
    draw_title,
    font,
    wrap_text_px,
)

MEDIA_W = 1032
MEDIA_H = 1126
LEFT = 86
RIGHT = MEDIA_W - 86
CONTENT_W = RIGHT - LEFT


def _wrapped_lines(draw: ImageDraw.ImageDraw, text: str, *, start_size: int, min_size: int, max_width: int, max_lines: int, bold: bool = False):
    for size in range(start_size, min_size - 1, -1):
        text_font = font(size, bold)
        lines = wrap_text_px(draw, str(text or ""), text_font, max_width)
        if len(lines) <= max_lines:
            return text_font, lines
    raise RuntimeError(f"Bill Tracker copy exceeds {max_lines} lines at minimum size: {text!r}")


def _draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], *, x: int, y: int, text_font, fill: str, line_gap: int) -> int:
    current = y
    for line in lines:
        draw.text((x, current), line, font=text_font, fill=fill, anchor="la")
        bbox = draw.textbbox((x, current), line, font=text_font, anchor="la")
        current = int(bbox[3] + line_gap)
    return current


def render_cover_media(edition: dict[str, Any], output_png: str | Path) -> dict[str, Any]:
    image = Image.new("RGB", (MEDIA_W, MEDIA_H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((LEFT, 60), "EIREPOLITIC BILL TRACKER", font=font(25, True), fill=ACCENT, anchor="la")
    headline_font, headline_lines = _wrapped_lines(
        draw,
        str(edition["cover_headline"]),
        start_size=58,
        min_size=42,
        max_width=CONTENT_W,
        max_lines=3,
        bold=True,
    )
    y = _draw_lines(draw, headline_lines, x=LEFT, y=155, text_font=headline_font, fill=TEXT, line_gap=12)
    y += 34
    draw.rectangle((LEFT, y, RIGHT, y + 4), fill=ACCENT)
    y += 55

    subtitle_font, subtitle_lines = _wrapped_lines(
        draw,
        str(edition["cover_subtitle"]),
        start_size=31,
        min_size=26,
        max_width=CONTENT_W,
        max_lines=4,
    )
    y = _draw_lines(draw, subtitle_lines, x=LEFT, y=y, text_font=subtitle_font, fill=TEXT, line_gap=11)
    y += 70

    draw.text((LEFT, y), "IN THIS EDITION", font=font(22, True), fill=ACCENT, anchor="la")
    y += 55
    topics_font, topics_lines = _wrapped_lines(
        draw,
        str(edition["cover_topics"]),
        start_size=29,
        min_size=25,
        max_width=CONTENT_W,
        max_lines=2,
        bold=True,
    )
    y = _draw_lines(draw, topics_lines, x=LEFT, y=y, text_font=topics_font, fill=TEXT, line_gap=10)

    draw.text((LEFT, MEDIA_H - 110), "Swipe for six one-page Bill explainers →", font=font(25, True), fill=TEXT, anchor="la")
    draw.text((LEFT, MEDIA_H - 58), f"Source: {edition['source_footer']}", font=font(16), fill=MUTED, anchor="la")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    return {"success": True, "renderer": "bill_tracker_cover_media_v1", "warnings": [], "headline_lines": len(headline_lines)}


def render_bill_media(bill: dict[str, Any], output_png: str | Path) -> dict[str, Any]:
    image = Image.new("RGB", (MEDIA_W, MEDIA_H), BG)
    draw = ImageDraw.Draw(image)

    formal_font, formal_lines = _wrapped_lines(
        draw,
        str(bill["formal_title"]),
        start_size=25,
        min_size=20,
        max_width=CONTENT_W,
        max_lines=3,
        bold=True,
    )
    y = _draw_lines(draw, formal_lines, x=LEFT, y=36, text_font=formal_font, fill=MUTED, line_gap=6)
    y += 18
    draw.text((LEFT, y), str(bill["status_line"]), font=font(20, True), fill=ACCENT, anchor="la")
    y += 37

    sponsor_font, sponsor_lines = _wrapped_lines(
        draw,
        f"Introduced by: {bill['introduced_by']}",
        start_size=20,
        min_size=18,
        max_width=CONTENT_W,
        max_lines=2,
    )
    y = _draw_lines(draw, sponsor_lines, x=LEFT, y=y, text_font=sponsor_font, fill=MUTED, line_gap=5)
    y += 22
    draw.rectangle((LEFT, y, RIGHT, y + 3), fill=ACCENT)
    y += 38

    draw.text((LEFT, y), "WHAT IT DOES", font=font(21, True), fill=ACCENT, anchor="la")
    y += 42
    what_font, what_lines = _wrapped_lines(
        draw,
        str(bill["what_it_does"]),
        start_size=27,
        min_size=23,
        max_width=CONTENT_W,
        max_lines=5,
    )
    y = _draw_lines(draw, what_lines, x=LEFT, y=y, text_font=what_font, fill=TEXT, line_gap=9)
    y += 30

    draw.text((LEFT, y), "MAIN DEBATE", font=font(21, True), fill=ACCENT, anchor="la")
    y += 43
    draw.text((LEFT, y), str(bill["case_label"]), font=font(18, True), fill=TEXT, anchor="la")
    y += 34
    case_font, case_lines = _wrapped_lines(
        draw,
        str(bill["case_for"]),
        start_size=22,
        min_size=19,
        max_width=CONTENT_W,
        max_lines=4,
    )
    y = _draw_lines(draw, case_lines, x=LEFT, y=y, text_font=case_font, fill=MUTED, line_gap=7)
    y += 28

    draw.text((LEFT, y), str(bill["concern_label"]), font=font(18, True), fill=TEXT, anchor="la")
    y += 34
    concern_font, concern_lines = _wrapped_lines(
        draw,
        str(bill["concerns"]),
        start_size=22,
        min_size=19,
        max_width=CONTENT_W,
        max_lines=4,
    )
    y = _draw_lines(draw, concern_lines, x=LEFT, y=y, text_font=concern_font, fill=MUTED, line_gap=7)

    footer_top = MEDIA_H - 116
    if y > footer_top - 20:
        raise RuntimeError(f"Bill Tracker body collides with footer for {bill['display_title']}: final_y={y}")
    draw.rectangle((LEFT, footer_top - 22, RIGHT, footer_top - 19), fill=ACCENT)
    record_font, record_lines = _wrapped_lines(
        draw,
        str(bill["record"]),
        start_size=17,
        min_size=15,
        max_width=CONTENT_W,
        max_lines=2,
        bold=True,
    )
    _draw_lines(draw, record_lines, x=LEFT, y=footer_top, text_font=record_font, fill=TEXT, line_gap=4)
    draw.text((LEFT, MEDIA_H - 46), f"Source: {bill['source']}", font=font(14), fill=MUTED, anchor="la")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    return {
        "success": True,
        "renderer": "bill_tracker_bill_media_v1",
        "warnings": [],
        "display_title": str(bill["display_title"]),
        "what_lines": len(what_lines),
        "case_lines": len(case_lines),
        "concern_lines": len(concern_lines),
        "final_body_y": y,
    }


def render_methodology(entries: list[tuple[str, str]], output_png: str | Path, *, title: str) -> dict[str, Any]:
    output_png = Path(output_png)
    metrics = draw_glossary(entries, output_png)
    image = Image.open(output_png).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((112, 34, 968, 169), fill=BG)
    draw_title(image, [title])
    image.save(output_png)
    return {
        "success": True,
        "renderer": "approved_glossary_methodology_v1",
        "output_path": str(output_png),
        "warnings": [],
        "glossary_metrics": metrics,
    }
