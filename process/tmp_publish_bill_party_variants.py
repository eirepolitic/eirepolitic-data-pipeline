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
ABSTAIN = "#9a8c72"

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
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_lines(draw, x, y, text, fnt, fill, width, gap=6):
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y = draw.textbbox((x, y), line, font=fnt)[3] + gap
    return y


def bar(draw, x, y, w, h, yes, no, abstain, nr):
    total = max(1, yes + no + abstain + nr)
    positive = [(yes, FOR), (no, AGAINST), (abstain, ABSTAIN), (nr, NO_VOTE)]
    positive = [(value, color) for value, color in positive if value > 0]
    cursor = x
    for index, (value, color) in enumerate(positive):
        seg_w = (x + w - cursor) if index == len(positive) - 1 else round(w * value / total)
        draw.rectangle([cursor, y, cursor + seg_w, y + h], fill=color)
        cursor += seg_w
    draw.rectangle([x, y, x + w, y + h], outline=MUTED, width=2)


def legend(draw, x, y):
    items = [("Tá", FOR), ("Níl", AGAINST), ("No recorded vote", NO_VOTE)]
    cursor = x
    for label, color in items:
        draw.rectangle([cursor, y + 4, cursor + 18, y + 22], fill=color, outline=MUTED)
        draw.text((cursor + 28, y), label, font=font(18, True), fill=TEXT)
        cursor += 230 if label != "No recorded vote" else 290


def header_base(subtitle):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 55), "EIREPOLITIC BILL TRACKER", font=font(22, True), fill=ACCENT)
    draw.text((70, 108), "Strategic Gas Reserve · Party Split", font=font(42, True), fill=TEXT)
    draw.rectangle([70, 170, 1010, 174], fill=ACCENT)
    draw.text((70, 198), subtitle, font=font(21), fill=MUTED)
    draw.text((70, 238), "Dáil passage vote · 30 June 2026", font=font(23, True), fill=ACCENT)
    bar(draw, 70, 282, 940, 46, overall["for"], overall["against"], overall["abstain"], overall["no"])
    legend(draw, 70, 345)
    draw.text((70, 388), "174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote", font=font(19, True), fill=TEXT)
    return image, draw


def variant_b1():
    image, draw = header_base("Variant B1 · larger text, top parties only, smaller groups combined")
    draw.text((70, 438), "PARTY BREAKDOWN", font=font(26, True), fill=ACCENT)
    draw.text((70, 475), "Using each party/group's eligible TDs as the denominator.", font=font(18), fill=MUTED)
    top = party_rows[:6]
    others = party_rows[6:]
    rows = top + [("Other small groups", sum(r[1] for r in others), sum(r[2] for r in others), sum(r[3] for r in others), sum(r[4] for r in others))]
    y = 530
    for name, eligible, yes, no, nr in rows:
        draw.text((70, y), name, font=font(24, True), fill=TEXT)
        draw.text((980, y), f"{eligible} TDs", font=font(18), fill=MUTED, anchor="ra")
        bar(draw, 70, y + 34, 770, 32, yes, no, 0, nr)
        draw.text((860, y + 32), f"Tá {yes}", font=font(18, True), fill=FOR)
        draw.text((930, y + 32), f"Níl {no}", font=font(18, True), fill=TEXT)
        draw.text((860, y + 58), f"No vote {nr}", font=font(16), fill=MUTED)
        y += 100
    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    draw.text((70, 1250), "Smaller groups combined here for readability.", font=font(16), fill=MUTED)
    image.save(OUT / "variant-b1.png")


def variant_b2():
    image, draw = header_base("Variant B2 · two-column grid, more parties shown with less tiny text")
    draw.text((70, 438), "PARTY BREAKDOWN", font=font(26, True), fill=ACCENT)
    draw.text((70, 475), "Two-column version to use space better while keeping party-level rows readable.", font=font(18), fill=MUTED)
    col_x = [70, 545]
    col_w = 435
    for index, row in enumerate(party_rows[:10]):
        col = index % 2
        grid_row = index // 2
        x = col_x[col]
        y = 530 + grid_row * 125
        name, eligible, yes, no, nr = row
        draw.text((x, y), name, font=font(21, True), fill=TEXT)
        draw.text((x + col_w, y), f"{eligible} TDs", font=font(16), fill=MUTED, anchor="ra")
        bar(draw, x, y + 33, col_w, 26, yes, no, 0, nr)
        chip_y = y + 68
        chips = [(f"Tá {yes}", FOR), (f"Níl {no}", AGAINST), (f"No vote {nr}", NO_VOTE)]
        cursor = x
        for text, color in chips:
            width = draw.textbbox((0, 0), text, font=font(15, True))[2] + 28
            draw.rounded_rectangle([cursor, chip_y, cursor + width, chip_y + 28], radius=8, outline=MUTED, fill=BG)
            draw.rectangle([cursor + 8, chip_y + 7, cursor + 20, chip_y + 19], fill=color, outline=MUTED)
            draw.text((cursor + 28, chip_y + 4), text, font=font(15, True), fill=TEXT)
            cursor += width + 8
    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    draw.text((70, 1250), "Shows more parties while using the width more efficiently.", font=font(16), fill=MUTED)
    image.save(OUT / "variant-b2.png")


def variant_b3():
    image, draw = header_base("Variant B3 · simplified party rows, least text")
    draw.text((70, 438), "PARTY BREAKDOWN", font=font(26, True), fill=ACCENT)
    draw.text((70, 475), "Simplified rows designed for faster mobile reading.", font=font(18), fill=MUTED)
    y = 525
    for name, eligible, yes, no, nr in party_rows[:8]:
        draw.text((70, y + 5), name, font=font(22, True), fill=TEXT)
        direction = "Mostly Tá" if yes > no else ("Mostly Níl" if no > yes else "Mixed")
        draw.text((375, y + 5), direction, font=font(18, True), fill=ACCENT if yes >= no else TEXT)
        bar(draw, 540, y, 340, 28, yes, no, 0, nr)
        draw.text((900, y + 2), f"{yes} / {no} / {nr}", font=font(17, True), fill=MUTED)
        draw.text((900, y + 26), "Tá · Níl · No vote", font=font(13), fill=MUTED)
        y += 72
    draw.text((70, 1120), "Smaller groups", font=font(20, True), fill=ACCENT)
    small = "Independent Ireland 3/0/1 · PBP–S 0/3/0 · Aontú 0/0/2 · Green 0/1/0 · 100% Redress 0/1/0"
    draw_lines(draw, 70, 1152, small, font(17), MUTED, 940, gap=4)
    draw.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    draw.text((70, 1250), "Most compressed; easiest to scan, but less precise than B1/B2.", font=font(16), fill=MUTED)
    image.save(OUT / "variant-b3.png")


def contact_sheet():
    sheet = Image.new("RGB", (1600, 1900), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((40, 25), "Bill Tracker · Variants of Option B", font=font(28, True), fill="black")
    items = [
        (OUT / "variant-b1.png", "B1", (40, 90)),
        (OUT / "variant-b2.png", "B2", (820, 90)),
        (OUT / "variant-b3.png", "B3", (430, 990)),
    ]
    for path, label, position in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 900))
        sheet.paste(image, position)
        draw.text((position[0], position[1] - 28), label, font=font(24, True), fill="black")
    sheet.save(OUT / "contact-sheet.png")


if __name__ == "__main__":
    variant_b1()
    variant_b2()
    variant_b3()
    contact_sheet()
    for path in sorted(OUT.iterdir()):
        print(path)
