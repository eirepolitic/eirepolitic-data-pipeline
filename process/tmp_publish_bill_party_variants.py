#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("artifacts/bill-party-variants")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
BG = "#0f2f24"
TEXT = "#f4ead7"
MUTED = "#c8bda8"
ACCENT = "#d8b45f"
FOR = ACCENT
AGAINST = TEXT
NO_VOTE = "#65756d"

overall = {"for": 90, "against": 57, "no": 27, "eligible": 174}
party_rows = [
    ("Fianna Fáil", 48, 42, 0, 6),
    ("Sinn Féin", 39, 0, 31, 8),
    ("Fine Gael", 38, 35, 0, 3),
    ("Independent", 15, 10, 2, 3),
    ("Social Democrats", 12, 0, 12, 0),
    ("Labour", 11, 0, 7, 4),
    ("Independent Ireland", 4, 3, 0, 1),
    ("PBP–S", 3, 0, 3, 0),
    ("Aontú", 2, 0, 0, 2),
    ("Green", 1, 0, 1, 0),
    ("100% Redress", 1, 0, 1, 0),
]

def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()

def width(draw, text, fnt):
    return draw.textbbox((0, 0), str(text), font=fnt)[2]

def centered(draw, y, text, fnt, fill):
    draw.text(((W - width(draw, text, fnt)) // 2, y), str(text), font=fnt, fill=fill)

def bar(draw, x, y, w, h, yes, no, nr):
    total = max(1, yes + no + nr)
    segs = [(yes, FOR), (no, AGAINST), (nr, NO_VOTE)]
    segs = [(v, c) for v, c in segs if v > 0]
    cursor = x
    for i, (value, color) in enumerate(segs):
        seg_w = (x + w - cursor) if i == len(segs) - 1 else round(w * value / total)
        draw.rectangle([cursor, y, cursor + seg_w, y + h], fill=color)
        cursor += seg_w
    draw.rectangle([x, y, x + w, y + h], outline=MUTED, width=2)

def build():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    centered(draw, 66, "Strategic Gas Reserve · Party Split", font(44, True), TEXT)
    draw.rectangle([115, 132, 965, 136], fill=ACCENT)

    centered(draw, 164, "Dáil passage vote · 30 June 2026", font(24, True), ACCENT)
    bar(draw, 70, 210, 940, 52, overall["for"], overall["against"], overall["no"])

    legend = [("Tá", FOR), ("Níl", AGAINST), ("No recorded vote", NO_VOTE)]
    lf = font(19, True)
    widths = [18 + 10 + width(draw, label, lf) for label, _ in legend]
    gap = 42
    cursor = (W - (sum(widths) + gap * (len(legend) - 1))) // 2
    for (label, color), item_w in zip(legend, widths):
        draw.rectangle([cursor, 287, cursor + 18, 305], fill=color, outline=MUTED)
        draw.text((cursor + 28, 282), label, font=lf, fill=TEXT)
        cursor += item_w + gap

    centered(draw, 325, "174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote", font(20, True), TEXT)
    centered(draw, 380, "PARTY BREAKDOWN", font(30, True), ACCENT)

    name_x = 72
    bar_x = 352
    bar_w = 448
    nums_x = 848
    row_top = 438

    draw.text((nums_x, row_top - 28), "Tá", font=font(16, True), fill=ACCENT)
    draw.text((nums_x + 44, row_top - 28), "Níl", font=font(16, True), fill=TEXT)
    draw.text((nums_x + 96, row_top - 28), "No vote", font=font(16, True), fill=MUTED)

    y = row_top
    for name, eligible, yes, no, nr in party_rows[:8]:
        draw.text((name_x, y + 3), name, font=font(23, True), fill=TEXT)
        bar(draw, bar_x, y, bar_w, 32, yes, no, nr)
        draw.text((nums_x, y + 4), f"{yes}", font=font(19, True), fill=ACCENT)
        draw.text((nums_x + 44, y + 4), f"{no}", font=font(19, True), fill=TEXT)
        draw.text((nums_x + 116, y + 4), f"{nr}", font=font(19, True), fill=MUTED)
        y += 70

    centered(draw, 1030, "Smaller groups", font(26, True), ACCENT)
    centered(draw, 1070, "Independent Ireland 3 / 0 / 1 · PBP–S 0 / 3 / 0 · Aontú 0 / 0 / 2", font(21), MUTED)
    centered(draw, 1098, "Green 0 / 1 / 0 · 100% Redress 0 / 1 / 0", font(21), MUTED)

    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    centered(draw, 1248, "Row format: Tá · Níl · No vote", font(16, True), MUTED)
    centered(draw, 1274, "No recorded vote is shown separately and does not automatically mean absent.", font(15), MUTED)

    image.save(OUT / "variant-b3-revised-v2.png")

    sheet = Image.new("RGB", (1200, 1500), "white")
    sd = ImageDraw.Draw(sheet)
    sd.text((40, 30), "Bill Tracker · Revised B3 v2", font=font(28, True), fill="black")
    preview = Image.open(OUT / "variant-b3-revised-v2.png").convert("RGB")
    preview.thumbnail((1080, 1350))
    sheet.paste(preview, (60, 90))
    sheet.save(OUT / "contact-sheet-v2.png")

if __name__ == "__main__":
    build()
    print(OUT / "variant-b3-revised-v2.png")
    print(OUT / "contact-sheet-v2.png")
