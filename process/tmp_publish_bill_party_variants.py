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

overall = {"for": 90, "against": 57, "abstain": 0, "no": 27, "eligible": 174}
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

def wrap(draw, text, fnt, max_width):
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if width(draw, trial, fnt) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def centered_lines(draw, y, text, fnt, fill, max_width, gap=4):
    for line in wrap(draw, text, fnt, max_width):
        centered(draw, y, line, fnt, fill)
        y = draw.textbbox((0, y), line, font=fnt)[3] + gap
    return y

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

def revised_b3():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 55), "EIREPOLITIC BILL TRACKER", font=font(22, True), fill=ACCENT)
    draw.text((70, 108), "Strategic Gas Reserve · Party Split", font=font(42, True), fill=TEXT)
    draw.rectangle([70, 170, 1010, 174], fill=ACCENT)
    draw.text((70, 198), "Revised B3 · cleaner layout with wider bars and clearer labels", font=font(20), fill=MUTED)

    centered(draw, 238, "Dáil passage vote · 30 June 2026", font(23, True), ACCENT)
    bar(draw, 70, 282, 940, 46, overall["for"], overall["against"], overall["no"])

    legend = [("Tá", FOR), ("Níl", AGAINST), ("No recorded vote", NO_VOTE)]
    lf = font(18, True)
    item_widths = [18 + 10 + width(draw, label, lf) for label, _ in legend]
    gap = 36
    cursor = (W - (sum(item_widths) + gap * (len(legend) - 1))) // 2
    for (label, color), item_w in zip(legend, item_widths):
        draw.rectangle([cursor, 349, cursor + 18, 367], fill=color, outline=MUTED)
        draw.text((cursor + 28, 345), label, font=lf, fill=TEXT)
        cursor += item_w + gap

    centered(draw, 388, "174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote", font(19, True), TEXT)

    centered(draw, 445, "PARTY BREAKDOWN", font(28, True), ACCENT)
    centered_lines(draw, 482, "Each row uses that party/group’s eligible TDs as the denominator.", font(18), MUTED, 760)

    name_x = 70
    bar_x = 360
    bar_w = 430
    numbers_x = 845
    row_top = 560

    draw.text((numbers_x, row_top - 26), "Tá", font=font(15, True), fill=ACCENT)
    draw.text((numbers_x + 42, row_top - 26), "Níl", font=font(15, True), fill=TEXT)
    draw.text((numbers_x + 92, row_top - 26), "No vote", font=font(15, True), fill=MUTED)

    y = row_top
    for name, eligible, yes, no, nr in party_rows[:8]:
        draw.text((name_x, y + 2), name, font=font(22, True), fill=TEXT)
        bar(draw, bar_x, y, bar_w, 30, yes, no, nr)
        draw.text((numbers_x, y + 2), f"{yes}", font=font(18, True), fill=ACCENT)
        draw.text((numbers_x + 42, y + 2), f"{no}", font=font(18, True), fill=TEXT)
        draw.text((numbers_x + 112, y + 2), f"{nr}", font=font(18, True), fill=MUTED)
        y += 66

    centered(draw, 1092, "Smaller groups", font(24, True), ACCENT)
    small = "Independent Ireland 3 / 0 / 1 · PBP–S 0 / 3 / 0 · Aontú 0 / 0 / 2 · Green 0 / 1 / 0 · 100% Redress 0 / 1 / 0"
    centered_lines(draw, 1127, small, font(19), MUTED, 900, gap=5)

    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    centered(draw, 1248, "Row format: Tá · Níl · No vote", font(16, True), MUTED)
    centered(draw, 1274, "No recorded vote is shown separately and does not automatically mean absent.", font(15), MUTED)
    image.save(OUT / "variant-b3-revised.png")

def contact_sheet():
    sheet = Image.new("RGB", (1200, 1500), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 30), "Bill Tracker · Revised B3", font=font(28, True), fill="black")
    preview = Image.open(OUT / "variant-b3-revised.png").convert("RGB")
    preview.thumbnail((1080, 1350))
    sheet.paste(preview, (60, 90))
    sheet.save(OUT / "contact-sheet-revised.png")

if __name__ == "__main__":
    revised_b3()
    contact_sheet()
    print(OUT / "variant-b3-revised.png")
    print(OUT / "contact-sheet-revised.png")
