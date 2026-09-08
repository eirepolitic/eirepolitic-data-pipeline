from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from instagram.factory.render_primitives import ACCENT, BG, MUTED, TEXT, font, wrap_text_px

MEDIA_W = 1032
MEDIA_H = 1126
LEFT = 86
RIGHT = MEDIA_W - 86
CONTENT_W = RIGHT - LEFT
FOR = ACCENT
AGAINST = TEXT
NO_VOTE = "#65756d"
ABSTAIN = "#9a8c72"


def _fit(draw, text, start, minimum, width, max_lines, bold=False):
    for size in range(start, minimum - 1, -1):
        f = font(size, bold)
        lines = wrap_text_px(draw, str(text), f, width)
        if len(lines) <= max_lines:
            return f, lines
    raise RuntimeError(f"copy overflow: {text}")


def _lines(draw, lines, x, y, f, fill, gap=6):
    cur = y
    for line in lines:
        draw.text((x, cur), line, font=f, fill=fill, anchor="la")
        cur = int(draw.textbbox((x, cur), line, font=f, anchor="la")[3] + gap)
    return cur


def _header(draw):
    f, lines = _fit(draw, "Development (Strategic Gas Reserve) Bill 2026", 24, 20, CONTENT_W, 2, True)
    y = _lines(draw, lines, LEFT, 34, f, MUTED, 5)
    y += 12
    draw.text((LEFT, y), "Dáil passage vote · 30 June 2026", font=font(19, True), fill=ACCENT, anchor="la")
    y += 40
    draw.text((LEFT, y), "Tá 90 · Níl 57 · 174 eligible TDs", font=font(20, True), fill=TEXT, anchor="la")
    return y + 42


def _draw_stack(draw, *, x, y, w, h, segments):
    total = sum(v for _, v, _ in segments)
    cursor = x
    for idx, (label, value, color) in enumerate(segments):
        seg_w = w - (cursor - x) if idx == len(segments) - 1 else round(w * value / total)
        draw.rectangle((cursor, y, cursor + seg_w, y + h), fill=color)
        cursor += seg_w
    draw.rectangle((x, y, x + w, y + h), outline=MUTED, width=2)


def _legend(draw, y):
    items = [("Tá", 90, FOR), ("Níl", 57, AGAINST), ("Abstain", 0, ABSTAIN), ("No recorded vote", 27, NO_VOTE)]
    x = LEFT
    for label, value, color in items:
        draw.rectangle((x, y + 5, x + 18, y + 23), fill=color, outline=MUTED)
        draw.text((x + 28, y), f"{label} {value}", font=font(17, True), fill=TEXT, anchor="la")
        x += 195 if label != "No recorded vote" else 245


def _draw_context(draw, y):
    draw.text((LEFT, y), "WHAT THE VOTE DECIDED", font=font(19, True), fill=ACCENT, anchor="la")
    y += 36
    copy = (
        "The Chair put the remaining sections, Title, Fourth Stage and passage of the Bill together. "
        "The motion carried and the Bill was sent to the Seanad."
    )
    f, lines = _fit(draw, copy, 22, 19, CONTENT_W, 4)
    y = _lines(draw, lines, LEFT, y, f, TEXT, 7)
    y += 24
    draw.text((LEFT, y), "Important: “no recorded vote” does not necessarily mean absent.", font=font(16, True), fill=MUTED, anchor="la")
    return y + 34


def render_option_a(output_png: str | Path) -> dict[str, Any]:
    image = Image.new("RGB", (MEDIA_W, MEDIA_H), BG)
    draw = ImageDraw.Draw(image)
    y = _header(draw)
    y = _draw_context(draw, y)

    draw.text((LEFT, y), "WHOLE DÁIL", font=font(20, True), fill=ACCENT, anchor="la")
    y += 42
    _draw_stack(draw, x=LEFT, y=y, w=CONTENT_W, h=72, segments=[
        ("Tá", 90, FOR), ("Níl", 57, AGAINST), ("No recorded vote", 27, NO_VOTE)
    ])
    y += 92
    _legend(draw, y)

    pct_y = y + 72
    draw.text((LEFT, pct_y), "51.7% Tá", font=font(26, True), fill=FOR, anchor="la")
    draw.text((LEFT + 285, pct_y), "32.8% Níl", font=font(26, True), fill=TEXT, anchor="la")
    draw.text((LEFT + 570, pct_y), "15.5% no recorded vote", font=font(22, True), fill=MUTED, anchor="la")

    draw.rectangle((LEFT, MEDIA_H - 102, RIGHT, MEDIA_H - 99), fill=ACCENT)
    draw.text((LEFT, MEDIA_H - 76), "Source: Houses of the Oireachtas · division vote_162", font=font(14), fill=MUTED, anchor="la")
    draw.text((LEFT, MEDIA_H - 48), "Vote denominator reconstructed from Dáil 34 membership on 30 June 2026.", font=font(14), fill=MUTED, anchor="la")

    output_png = Path(output_png); output_png.parent.mkdir(parents=True, exist_ok=True); image.save(output_png)
    return {"success": True, "renderer": "bill_vote_option_a_v1", "overall": {"for":90,"against":57,"abstain":0,"no_recorded_vote":27,"eligible":174}}


def render_option_b(output_png: str | Path) -> dict[str, Any]:
    image = Image.new("RGB", (MEDIA_W, MEDIA_H), BG)
    draw = ImageDraw.Draw(image)
    y = _header(draw)

    _draw_stack(draw, x=LEFT, y=y, w=CONTENT_W, h=48, segments=[
        ("Tá", 90, FOR), ("Níl", 57, AGAINST), ("No recorded vote", 27, NO_VOTE)
    ])
    y += 63
    _legend(draw, y)
    y += 62

    draw.text((LEFT, y), "PARTY BREAKDOWN", font=font(20, True), fill=ACCENT, anchor="la")
    y += 38
    draw.text((LEFT, y), "Each row uses that party/group’s eligible TDs as the denominator.", font=font(16), fill=MUTED, anchor="la")
    y += 35

    rows = [
        ("Fianna Fáil", 48, 42, 0, 6),
        ("Sinn Féin", 39, 0, 31, 8),
        ("Fine Gael", 38, 35, 0, 3),
        ("Independent", 15, 10, 2, 3),
        ("Social Democrats", 12, 0, 12, 0),
        ("Labour", 11, 0, 7, 4),
    ]
    bar_x = LEFT + 270
    bar_w = 500
    for party, eligible, yes, no, nr in rows:
        draw.text((LEFT, y + 4), party, font=font(18, True), fill=TEXT, anchor="la")
        _draw_stack(draw, x=bar_x, y=y, w=bar_w, h=27, segments=[
            ("Tá", yes, FOR), ("Níl", no, AGAINST), ("No recorded", nr, NO_VOTE)
        ])
        draw.text((RIGHT, y + 3), f"{yes}–{no}–{nr}", font=font(16, True), fill=MUTED, anchor="ra")
        y += 52

    y += 8
    draw.text((LEFT, y), "Smaller groups", font=font(17, True), fill=ACCENT, anchor="la")
    y += 30
    small = "Independent Ireland 3–0–1 · PBP–S 0–3–0 · Green 0–1–0 · 100% RDR 0–1–0 · Aontú 0–0–2"
    f, lines = _fit(draw, small, 16, 14, CONTENT_W, 2)
    y = _lines(draw, lines, LEFT, y, f, MUTED, 4)

    draw.rectangle((LEFT, MEDIA_H - 102, RIGHT, MEDIA_H - 99), fill=ACCENT)
    draw.text((LEFT, MEDIA_H - 76), "Row key: Tá – Níl – no recorded vote. Abstentions: 0.", font=font(14, True), fill=MUTED, anchor="la")
    draw.text((LEFT, MEDIA_H - 48), "Party affiliation reconstructed from date-correct party history.", font=font(14), fill=MUTED, anchor="la")

    output_png = Path(output_png); output_png.parent.mkdir(parents=True, exist_ok=True); image.save(output_png)
    return {"success": True, "renderer": "bill_vote_option_b_v1", "party_rows": len(rows), "abstain": 0}
