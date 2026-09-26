#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('artifacts/bill-gas-reserve-pair')
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
BG = '#0f2f24'
TEXT = '#f4ead7'
MUTED = '#c8bda8'
ACCENT = '#d8b45f'
FOR = ACCENT
AGAINST = TEXT
NO_VOTE = '#65756d'
PANEL = '#173f31'

party_rows = [
    ('Fianna Fáil', 48, 42, 0, 6),
    ('Sinn Féin', 39, 0, 31, 8),
    ('Fine Gael', 38, 35, 0, 3),
    ('Independent', 15, 10, 2, 3),
    ('Social Democrats', 12, 0, 12, 0),
    ('Labour', 11, 0, 7, 4),
    ('Independent Ireland', 4, 3, 0, 1),
    ('PBP–S', 3, 0, 3, 0),
    ('Aontú', 2, 0, 0, 2),
    ('Green', 1, 0, 1, 0),
    ('100% Redress', 1, 0, 1, 0),
]


def font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def measure(draw, text, fnt):
    b = draw.textbbox((0, 0), str(text), font=fnt)
    return b[2] - b[0], b[3] - b[1]


def centered(draw, y, text, fnt, fill):
    tw, _ = measure(draw, text, fnt)
    draw.text(((W - tw) / 2, y), text, font=fnt, fill=fill)


def wrap(draw, text, fnt, width):
    words = str(text).split()
    lines, current = [], ''
    for word in words:
        trial = word if not current else current + ' ' + word
        if measure(draw, trial, fnt)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, x, y, text, fnt, fill, width, gap=7, max_lines=None):
    lines = wrap(draw, text, fnt, width)
    if max_lines is not None and len(lines) > max_lines:
        raise RuntimeError(f'copy overflow: {text}')
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        _, h = measure(draw, line, fnt)
        y += h + gap
    return y


def panel(draw, x, y, w, h, heading, body, heading_size=21, body_size=19):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=20, fill=PANEL, outline='#31594a', width=2)
    draw.text((x+24, y+20), heading, font=font(heading_size, True), fill=ACCENT)
    draw_wrapped(draw, x+24, y+56, body, font(body_size), TEXT, w-48, gap=7, max_lines=5)


def bar(draw, x, y, w, h, yes, no, nr):
    total = max(1, yes + no + nr)
    segs = [(yes, FOR), (no, AGAINST), (nr, NO_VOTE)]
    segs = [(v, c) for v, c in segs if v > 0]
    cursor = x
    for i, (value, color) in enumerate(segs):
        sw = (x+w-cursor) if i == len(segs)-1 else round(w*value/total)
        draw.rectangle([cursor, y, cursor+sw, y+h], fill=color)
        cursor += sw
    draw.rectangle([x, y, x+w, y+h], outline=MUTED, width=2)


def build_explainer():
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, 54, 'Development (Strategic Gas Reserve) Bill 2026', font(36, True), TEXT)
    d.rectangle([110, 118, 970, 123], fill=ACCENT)
    centered(d, 146, 'WHAT IT DOES & WHAT THE DÁIL VOTE MEANT', font(22, True), ACCENT)
    centered(d, 184, 'Government Bill · Minister for Climate, Energy and the Environment', font(18), MUTED)

    panel(
        d, 60, 236, 460, 245,
        'WHAT THE BILL DOES',
        'Creates a bespoke approval route for a strategic gas reserve intended for emergency energy security. For this project, the normal Planning and Development Acts are disapplied, while environmental assessment requirements remain.',
        body_size=18,
    )
    panel(
        d, 560, 236, 460, 245,
        'PRACTICAL EFFECT',
        'Allows the Minister to decide the development application directly under accelerated timelines. The Government said the reserve would be State-owned and State-controlled and used as emergency back-up rather than to increase normal gas demand.',
        body_size=18,
    )
    panel(
        d, 60, 512, 460, 260,
        'CASE MADE FOR IT',
        'Supporters argued Ireland remains highly dependent on imported gas and needs a back-up supply if imports are disrupted. The Government presented the reserve as a temporary energy-security measure during the transition to renewables.',
        body_size=18,
    )
    panel(
        d, 560, 512, 460, 260,
        'CONCERNS RAISED',
        'Opponents argued the reserve risks locking Ireland into fossil-fuel infrastructure, weakening climate objectives and bypassing normal planning safeguards. They also criticised the accelerated timetable and limited scrutiny of amendments.',
        body_size=18,
    )

    d.rounded_rectangle([60, 805, 1020, 1128], radius=22, fill='#102b22', outline=ACCENT, width=3)
    centered(d, 833, 'WHAT THE 30 JUNE DÁIL VOTE MEANT', font(24, True), ACCENT)
    draw_wrapped(
        d, 92, 882,
        'The Chair put one combined question: agree the remaining sections and Title, complete Fourth Stage, and pass the Bill. A Tá therefore meant passing the Bill through the Dáil in the form then before the House; a Níl meant rejecting that passage motion.',
        font(20), TEXT, 896, gap=8, max_lines=5,
    )
    d.text((92, 1025), 'Result:', font=font(20, True), fill=MUTED)
    d.text((180, 1025), '90 Tá · 57 Níl — carried', font=font(22, True), fill=TEXT)
    draw_wrapped(
        d, 92, 1068,
        'Effect: the Bill completed its Dáil stages and was sent to the Seanad. It was subsequently enacted on 23 July 2026.',
        font(18, True), MUTED, 896, gap=6, max_lines=3,
    )

    d.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    centered(d, 1250, 'Sources: Houses of the Oireachtas bill text, explanatory memorandum and Dáil debate record', font(15), MUTED)
    im.save(OUT/'01-gas-reserve-explainer.png')


def build_vote():
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, 62, 'Strategic Gas Reserve · Party Split', font(45, True), TEXT)
    d.rectangle([110, 130, 970, 135], fill=ACCENT)
    centered(d, 160, 'Dáil passage vote · 30 June 2026', font(25, True), ACCENT)
    bar(d, 70, 208, 940, 56, 90, 57, 27)

    legend = [('Tá', FOR), ('Níl', AGAINST), ('No recorded vote', NO_VOTE)]
    lf = font(19, True)
    widths = [18+12+measure(d, label, lf)[0] for label, _ in legend]
    gap = 42
    cursor = (W - (sum(widths)+gap*2))/2
    for (label, color), iw in zip(legend, widths):
        d.rectangle([cursor, 290, cursor+18, 308], fill=color, outline=MUTED)
        d.text((cursor+30, 286), label, font=lf, fill=TEXT)
        cursor += iw + gap

    centered(d, 328, '174 eligible TDs · 90 Tá · 57 Níl · 0 abstentions · 27 no recorded vote', font(20, True), TEXT)
    centered(d, 382, 'PARTY BREAKDOWN', font(31, True), ACCENT)

    name_x, bar_x, bar_w, row_top, row_h, bar_h = 64, 348, 470, 438, 73, 36
    centers = [866, 922, 998]
    for (label, color), cx in zip([('Tá', ACCENT), ('Níl', TEXT), ('No vote', MUTED)], centers):
        tw, _ = measure(d, label, font(16, True))
        d.text((cx-tw/2, row_top-32), label, font=font(16, True), fill=color)

    for i, (name, eligible, yes, no, nr) in enumerate(party_rows[:8]):
        y = row_top + i*row_h
        d.text((name_x, y+2), name, font=font(24, True), fill=TEXT)
        bar(d, bar_x, y, bar_w, bar_h, yes, no, nr)
        for (number, color), cx in zip([(str(yes), ACCENT), (str(no), TEXT), (str(nr), MUTED)], centers):
            tw, _ = measure(d, number, font(20, True))
            d.text((cx-tw/2, y+5), number, font=font(20, True), fill=color)

    centered(d, 1064, 'Smaller groups', font(27, True), ACCENT)
    centered(d, 1102, 'Aontú 0 / 0 / 2 · Green 0 / 1 / 0 · 100% Redress 0 / 1 / 0', font(22), MUTED)
    d.rectangle([70, 1235, 1010, 1238], fill=ACCENT)
    centered(d, 1248, 'Row format: Tá · Níl · No vote', font(16, True), MUTED)
    centered(d, 1274, 'No recorded vote is shown separately and does not automatically mean absent.', font(15), MUTED)
    im.save(OUT/'02-gas-reserve-vote.png')


def build_contact_sheet():
    slides = [Image.open(OUT/'01-gas-reserve-explainer.png').convert('RGB'), Image.open(OUT/'02-gas-reserve-vote.png').convert('RGB')]
    sheet = Image.new('RGB', (1600, 1120), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((40, 24), 'Bill Tracker · Strategic Gas Reserve · two-slide pattern', font=font(28, True), fill='black')
    x_positions = [40, 820]
    for slide, x in zip(slides, x_positions):
        slide.thumbnail((720, 900))
        sheet.paste(slide, (x, 90))
    sheet.save(OUT/'contact-sheet.png')


if __name__ == '__main__':
    build_explainer()
    build_vote()
    build_contact_sheet()
    print(OUT)
