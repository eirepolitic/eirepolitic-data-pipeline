"""Stage-specific renderers for the approved First Stage Bill Tracker variant."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ImageDraw

from instagram.factory.render_primitives import ACCENT, BG, MUTED, TEXT, base_slide, font
from instagram.projects.bill_tracker_factory_v1.renderers import (
    W,
    _draw_centered_wrapped,
    _draw_wrapped,
    _fit_wrapped,
    _footer,
    _measure_lines,
    _panel,
    _rule,
    _shared_body_font,
    _title,
)

PANEL_W = 456


def render_first_stage_cover(post: dict[str, Any], series: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    draw.text((W // 2, 126), series["title_top"], font=font(28, True), fill=ACCENT, anchor="ma")
    draw.text((W // 2, 174), series["title_main"], font=font(44, True), fill=TEXT, anchor="ma")
    _rule(draw, 240, left=138, right=942, width=5)
    draw.text((W // 2, 282), f"{series['status']} · {post['edition_label']}", font=font(26, True), fill=ACCENT, anchor="ma")
    intro_f, _ = _fit_wrapped(draw, post["cover_intro"], width=800, start=22, minimum=19, max_lines=4)
    _draw_centered_wrapped(draw, post["cover_intro"], cx=W // 2, y=342, f=intro_f, width=800, gap=4)
    draw.text((W // 2, 515), "IN THIS POST", font=font(23, True), fill=ACCENT, anchor="ma")
    y = 565
    bills = post["bills"]
    if len(bills) != 3:
        raise RuntimeError(f"First Stage cover expects three Bills; got {len(bills)}")
    for idx, bill in enumerate(bills, start=1):
        box = (76, y, 1004, y + 176)
        _panel(draw, box, radius=20)
        cx, cy = 135, y + 88
        draw.ellipse((106, cy - 30, 164, cy + 28), fill=ACCENT)
        draw.text((cx, cy), str(idx), font=font(23, True), fill=BG, anchor="mm")
        f, lines = _fit_wrapped(draw, bill["formal_title"], width=765, start=24, minimum=18, max_lines=3, bold=True)
        total = _measure_lines(draw, lines, f, 4)
        ty = y + (176 - total) // 2
        for line in lines:
            draw.text((570, ty), line, font=f, fill=TEXT, anchor="ma")
            b = draw.textbbox((570, ty), line, font=f, anchor="ma")
            ty = int(b[3] + 4)
        y += 198
    _footer(draw, f"EirePolitic · First Stage · {post['edition_label'].title()}")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_first_stage_cover_v1", "warnings": [], "bill_count": 3, "source_footer": True}


def render_first_stage_bill(bill: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    title_bottom = _title(draw, bill["formal_title"], top=38, max_width=900, start=34, minimum=24, max_lines=2)
    rule_y = max(116, title_bottom + 20); _rule(draw, rule_y)
    stage_y = rule_y + 34
    draw.text((W // 2, stage_y), f"FIRST STAGE · BILL {bill['bill_number']}", font=font(23, True), fill=ACCENT, anchor="ma")
    meta_y = stage_y + 36
    draw.text((W // 2, meta_y), f"{bill['origin_house']} · {bill['introduction_date']} · {bill['bill_type']}", font=font(16, True), fill=MUTED, anchor="ma")
    sponsor_y = meta_y + 31
    sponsor_font, _ = _fit_wrapped(draw, f"Sponsors · {bill['sponsors']}", width=880, start=16, minimum=14, max_lines=2)
    sponsor_end = _draw_centered_wrapped(draw, f"Sponsors · {bill['sponsors']}", cx=W // 2, y=sponsor_y, f=sponsor_font, width=880, fill=MUTED, gap=3)

    box_top = sponsor_end + 23; box_h = 285; gap = 24
    boxes = [
        (60, box_top, 516, box_top + box_h), (564, box_top, 1020, box_top + box_h),
        (60, box_top + box_h + gap, 516, box_top + 2 * box_h + gap),
        (564, box_top + box_h + gap, 1020, box_top + 2 * box_h + gap),
    ]
    labels = ["WHAT THE BILL PROPOSES", "PRACTICAL EFFECT", "WHY IT WAS INTRODUCED", "WHERE IT IS NOW"]
    texts = [bill["what"], bill["effect"], bill["why"], bill["where"]]
    body_font = _shared_body_font(draw, texts, width=PANEL_W - 48, height=202, start=24, minimum=17, max_lines=9, gap=4)
    for box, label, text in zip(boxes, labels, texts):
        _panel(draw, box)
        label_font, _ = _fit_wrapped(draw, label, width=PANEL_W - 48, start=18, minimum=16, max_lines=2, bold=True)
        label_end = _draw_wrapped(draw, label, xy=(box[0] + 24, box[1] + 19), f=label_font, width=PANEL_W - 48, fill=ACCENT, gap=2)
        body_y = max(box[1] + 72, label_end + 12)
        body_end = _draw_wrapped(draw, text, xy=(box[0] + 24, body_y), f=body_font, width=PANEL_W - 48, fill=TEXT, gap=4)
        if body_end > box[3] - 16:
            raise RuntimeError(f"First Stage body overflow for {bill['formal_title']} / {label}: {body_end} > {box[3] - 16}")

    note_top = boxes[2][3] + 27; note_bottom = 1211
    _panel(draw, (60, note_top, 1020, note_bottom), radius=18, outline=ACCENT, width=3, fill=BG)
    draw.text((W // 2, note_top + 27), "WHAT FIRST STAGE MEANS", font=font(20, True), fill=ACCENT, anchor="ma")
    note = (
        "This is the formal initiation step, not a finding that the House supports the Bill’s general principles. "
        "A recorded member-by-member vote is not normally required here; substantive principle debate usually comes at Second Stage."
    )
    # Human-approved change: fill the explanatory box more assertively while retaining deterministic fit.
    note_font = _shared_body_font(draw, [note], width=850, height=note_bottom - (note_top + 76) - 22, start=34, minimum=20, max_lines=8, gap=6)
    note_y = note_top + 82
    note_end = _draw_centered_wrapped(draw, note, cx=W // 2, y=note_y, f=note_font, width=850, fill=TEXT, gap=6)
    if note_end > note_bottom - 18:
        raise RuntimeError(f"First Stage note overflow: {note_end} > {note_bottom - 18}")

    _footer(draw, f"EirePolitic · Source: Houses of the Oireachtas · Bill {bill['bill_number']}")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_first_stage_bill_v1", "warnings": [], "shared_body_font": body_font.size, "first_stage_note_font": note_font.size, "source_footer": True}


def render_first_stage_explainer(glossary: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    draw.text((W // 2, 82), "GLOSSARY", font=font(40, True), fill=TEXT, anchor="ma")
    draw.text((W // 2, 142), "HOW TO READ FIRST STAGE BILLS", font=font(23, True), fill=ACCENT, anchor="ma")
    _rule(draw, 186, left=112, right=968, width=4)
    terms = glossary["first_stage_terms"]
    heights = [170, 150, 165, 190, 195]
    y = 230
    body_width = 820
    body_height = min(h - 72 for h in heights) - 12
    shared = _shared_body_font(draw, [t["body"] for t in terms], width=body_width, height=body_height, start=23, minimum=17, max_lines=5, gap=4)
    for term, bh in zip(terms, heights):
        box = (82, y, 998, y + bh)
        _panel(draw, box, radius=16)
        draw.text((106, y + 23), term["term"], font=font(20, True), fill=ACCENT, anchor="la")
        body_end = _draw_wrapped(draw, term["body"], xy=(106, y + 66), f=shared, width=body_width, fill=TEXT, gap=4)
        if body_end > box[3] - 12:
            raise RuntimeError(f"First Stage glossary overflow for {term['term']}: {body_end} > {box[3] - 12}")
        y += bh + 18
    _footer(draw, "EirePolitic · Glossary · First Stage")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_first_stage_glossary_v1", "warnings": [], "shared_body_font": shared.size, "source_footer": True}
