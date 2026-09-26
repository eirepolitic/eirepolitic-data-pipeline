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

def text_size(draw, txt, fnt):
    bbox = draw.textbbox((0, 0), str(txt), font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def centered(draw, y, txt, fnt, fill):
    tw, _ = text_size(draw, txt, fnt)
    draw.text(((W - tw) // 2, y), str(txt), font=fnt, fill=fill)

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

    centered(draw, 62, "Strategic Gas Reserve · Party Split", font(45, True), TEXT)
    draw.rectangle([110, 130, 970, 135], fill=ACCENT)
    centered(draw, 160, "Dáil passage vote · 30 June 2026", font(25, True), ACCENT)
    bar(draw, 70, 208, 940, 56, overall["for"], overall["against"], overall["no"])

    legend = [("Tá", FOR), ("Níl", AGAINST), ("No recorded vote", NO_VOTE)]
    legend_font = font(19, True)
    widths = [18 + 12 + text_size(draw, label, legend_font)[0] for label, _ in legend]
    gap = 42
    cursor = (W - (sum(widths) + gap * (len(widths) - 1))) // 2
    for (label, color), item_w in zip(legend, widths):
        draw.rectangle([cursor, 290, cursor + 18, 308], fill=color, outline=MUTED)
        draw.text((cursor + 30, 286), label, font=legend_font, fill=TEXT)
        cursor += item_w + gap

    centered(draw, 328, "174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote", font(20, True), TEXT)
    centered(draw, 382, "PARTY BREAKDOWN", font(31, True), ACCENT)

    name_x = 64
    bar_x = 348
    bar_w = 470
    row_top = 438
    row_h = 73
    bar_h = 36
    name_font = font(24, True)
    num_font = font(20, True)
    head_font = font(16, True)

    count_centers = [866, 922, 998]
    headers = [("Tá", ACCENT), ("Níl", TEXT), ("No vote", MUTED)]
    for (label, color), cx in zip(headers, count_centers):
        tw, _ = text_size(draw, label, head_font)
        draw.text((cx - tw / 2, row_top - 32), label, font=head_font, fill=color)

    for i, (name, eligible, yes, no, nr) in enumerate(party_rows[:8]):
        y = row_top + i * row_h
        draw.text((name_x, y + 2), name, font=name_font, fill=TEXT)
        bar(draw, bar_x, y, bar_w, bar_h, yes, no, nr)
        for (num, color), cx in zip([(str(yes), ACCENT), (str(no), TEXT), (str(nr), MUTED)], count_centers):
            tw, _ = text_size(draw, num, num_font)
            draw.text((cx - tw / 2, y + 5), num, font=num_font, fill=color)

    centered(draw, 1064, "Smaller groups", font(27, True), ACCENT)
    centered(draw, 1102, "Independent Ireland 3 / 0 / 1 · PBP–S 0 / 3 / 0 · Aontú 0 / 0 / 2", font(22), MUTED)
    centered(draw, 1134, "Green 0 / 1 / 0 · 100% Redress 0 / 1 / 0", font(22), MUTED)

    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    centered(draw, 1248, "Row format: Tá · Níl · No vote", font(16, True), MUTED)
    centered(draw, 1274, "No recorded vote is shown separately and does not automatically mean absent.", font(15), MUTED)

    image.save(OUT / "variant-b3-revised-v3.png")

    sheet = Image.new("RGB", (1200, 1500), "white")
    sd = ImageDraw.Draw(sheet)
    sd.text((40, 30), "Bill Tracker · Revised B3 v3", font=font(28, True), fill="black")
    preview = Image.open(OUT / "variant-b3-revised-v3.png").convert("RGB")
    preview.thumbnail((1080, 1350))
    sheet.paste(preview, (60, 90))
    sheet.save(OUT / "contact-sheet-v3.png")

if __name__ == "__main__":
    build()
    print(OUT / "variant-b3-revised-v3.png")
    print(OUT / "contact-sheet-v3.png")
