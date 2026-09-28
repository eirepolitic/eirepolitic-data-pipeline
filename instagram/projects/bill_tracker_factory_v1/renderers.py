from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from instagram.factory.render_primitives import ACCENT, BG, MUTED, TEXT, base_slide, font, wrap_text_px

W, H = 1080, 1350
PANEL = "#174638"
PANEL_OUTLINE = "#346857"
NO_VOTE = "#65756d"
LEFT, RIGHT = 60, 1020
CONTENT_W = RIGHT - LEFT


def _measure_lines(draw: ImageDraw.ImageDraw, lines: Iterable[str], f: ImageFont.ImageFont, gap: int) -> int:
    lines = list(lines)
    if not lines:
        return 0
    heights = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=f)
        heights.append(b[3] - b[1])
    return sum(heights) + gap * (len(lines) - 1)


def _wrapped_advance(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.ImageFont, width: int, gap: int) -> tuple[int, list[str]]:
    lines = wrap_text_px(draw, text, f, width)
    cur = 0
    for line in lines:
        b = draw.textbbox((0, cur), line, font=f, anchor="la")
        cur = int(b[3] + gap)
    return cur, lines


def _fit_wrapped(draw: ImageDraw.ImageDraw, text: str, *, width: int, start: int, minimum: int, max_lines: int, bold: bool = False) -> tuple[ImageFont.ImageFont, list[str]]:
    for size in range(start, minimum - 1, -1):
        f = font(size, bold)
        lines = wrap_text_px(draw, text, f, width)
        if len(lines) <= max_lines:
            return f, lines
    raise RuntimeError(f"Text does not fit safely at minimum size: {text!r}")


def _shared_body_font(draw: ImageDraw.ImageDraw, texts: list[str], *, width: int, height: int, start: int, minimum: int, max_lines: int, gap: int = 5) -> ImageFont.ImageFont:
    for size in range(start, minimum - 1, -1):
        f = font(size)
        good = True
        for text in texts:
            advance, lines = _wrapped_advance(draw, text, f, width, gap)
            if len(lines) > max_lines or advance > height:
                good = False
                break
        if good:
            return f
    raise RuntimeError("Comparable text boxes cannot share a safe body font size")


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, *, xy: tuple[int, int], f: ImageFont.ImageFont, width: int, fill: str = TEXT, gap: int = 5, anchor: str = "la") -> int:
    x, y = xy
    lines = wrap_text_px(draw, text, f, width)
    cur = y
    for line in lines:
        draw.text((x, cur), line, font=f, fill=fill, anchor=anchor)
        b = draw.textbbox((x, cur), line, font=f, anchor=anchor)
        cur = int(b[3] + gap)
    return cur


def _draw_centered_wrapped(draw: ImageDraw.ImageDraw, text: str, *, cx: int, y: int, f: ImageFont.ImageFont, width: int, fill: str = TEXT, gap: int = 5) -> int:
    lines = wrap_text_px(draw, text, f, width)
    cur = y
    for line in lines:
        draw.text((cx, cur), line, font=f, fill=fill, anchor="ma")
        b = draw.textbbox((cx, cur), line, font=f, anchor="ma")
        cur = int(b[3] + gap)
    return cur


def _title(draw: ImageDraw.ImageDraw, text: str, *, top: int = 48, max_width: int = 900, start: int = 35, minimum: int = 27, max_lines: int = 2) -> int:
    f, lines = _fit_wrapped(draw, text, width=max_width, start=start, minimum=minimum, max_lines=max_lines, bold=True)
    line_gap = 4
    total = _measure_lines(draw, lines, f, line_gap)
    y = top
    for line in lines:
        draw.text((W // 2, y), line, font=f, fill=TEXT, anchor="ma")
        b = draw.textbbox((W // 2, y), line, font=f, anchor="ma")
        y = int(b[3] + line_gap)
    return top + total


def _rule(draw: ImageDraw.ImageDraw, y: int, *, left: int = 82, right: int = 998, width: int = 5) -> None:
    draw.rectangle((left, y, right, y + width), fill=ACCENT)


def _footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    _rule(draw, 1260, left=58, right=1022, width=4)
    draw.text((W // 2, 1286), text, font=font(14, True), fill=MUTED, anchor="ma")


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, radius: int = 18, outline: str = PANEL_OUTLINE, width: int = 2, fill: str = PANEL) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def render_cover(post: dict[str, Any], series: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide()
    draw = ImageDraw.Draw(im)
    draw.text((W // 2, 126), series["title_top"], font=font(28, True), fill=ACCENT, anchor="ma")
    draw.text((W // 2, 174), series["title_main"], font=font(44, True), fill=TEXT, anchor="ma")
    _rule(draw, 240, left=138, right=942, width=5)
    draw.text((W // 2, 282), f"{series['status']} · {post['edition_label']}", font=font(26, True), fill=ACCENT, anchor="ma")
    _draw_centered_wrapped(draw, post["cover_intro"], cx=W // 2, y=345, f=font(22), width=780, gap=4)
    draw.text((W // 2, 518), "IN THIS POST", font=font(23, True), fill=ACCENT, anchor="ma")
    y = 570
    bills = post["bills"]
    if len(bills) != 3:
        raise RuntimeError(f"Cover expects three bills; got {len(bills)}")
    for idx, bill in enumerate(bills, start=1):
        box = (76, y, 1004, y + 176)
        _panel(draw, box, radius=20)
        cx, cy = 135, y + 88
        draw.ellipse((106, cy - 30, 164, cy + 28), fill=ACCENT)
        draw.text((cx, cy), str(idx), font=font(23, True), fill=BG, anchor="mm")
        f, lines = _fit_wrapped(draw, bill["formal_title"], width=765, start=24, minimum=19, max_lines=3, bold=True)
        total = _measure_lines(draw, lines, f, 4)
        ty = y + (176 - total) // 2
        for line in lines:
            draw.text((570, ty), line, font=f, fill=TEXT, anchor="ma")
            b = draw.textbbox((570, ty), line, font=f, anchor="ma")
            ty = int(b[3] + 4)
        y += 198
    _footer(draw, f"EirePolitic · Enacted · {post['edition_label'].title()}")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_cover_v3", "warnings": [], "bill_count": 3}


def render_explainer(bill: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    title_bottom = _title(draw, bill["formal_title"], top=38, max_width=900, start=34, minimum=25, max_lines=2)
    rule_y = max(116, title_bottom + 22)
    _rule(draw, rule_y)
    subtitle_y = rule_y + 34
    draw.text((W // 2, subtitle_y), "WHAT IT DOES & WHY IT WAS DEBATED", font=font(23, True), fill=ACCENT, anchor="ma")
    sponsor_y = subtitle_y + 38
    draw.text((W // 2, sponsor_y), f"Introduced by the {bill['introduced_by']}", font=font(17), fill=MUTED, anchor="ma")

    box_top = sponsor_y + 38
    box_h = 266
    gap = 26
    box_w = 456
    boxes = [
        (60, box_top, 60 + box_w, box_top + box_h),
        (564, box_top, 564 + box_w, box_top + box_h),
        (60, box_top + box_h + gap, 60 + box_w, box_top + 2 * box_h + gap),
        (564, box_top + box_h + gap, 564 + box_w, box_top + 2 * box_h + gap),
    ]
    labels = ["WHAT THE BILL DOES", "PRACTICAL EFFECT", "WHY SOME TDs BACKED IT", bill["concerns_label"]]
    texts = [bill["what"], bill["effect"], bill["backed"], bill["concerns"]]
    body_font = _shared_body_font(draw, texts, width=402, height=185, start=25, minimum=20, max_lines=7, gap=4)
    for box, label, text in zip(boxes, labels, texts):
        _panel(draw, box)
        draw.text((box[0] + 24, box[1] + 20), label, font=font(18, True), fill=ACCENT, anchor="la")
        body_end = _draw_wrapped(draw, text, xy=(box[0] + 24, box[1] + 63), f=body_font, width=box_w - 48, fill=TEXT, gap=4)
        if body_end > box[3] - 18:
            raise RuntimeError(f"Explainer body overflow for {bill['formal_title']} / {label}: {body_end} > {box[3] - 18}")

    result_top = boxes[2][3] + 38
    result_bottom = 1206
    result_box = (60, result_top, 1020, result_bottom)
    _panel(draw, result_box, radius=18, outline=ACCENT, width=3, fill=BG)
    draw.text((W // 2, result_top + 29), bill["vote_box_title"], font=font(22, True), fill=ACCENT, anchor="ma")
    body_y = result_top + 78
    body_font2, _ = _fit_wrapped(draw, bill["vote_box_body"], width=840, start=20, minimum=17, max_lines=4)
    body_end = _draw_centered_wrapped(draw, bill["vote_box_body"], cx=W // 2, y=body_y, f=body_font2, width=840, fill=TEXT, gap=5)
    result_label_y = body_end + 28
    draw.text((W // 2, result_label_y), "RESULT", font=font(17, True), fill=MUTED, anchor="ma")
    result_y = result_label_y + 39
    draw.text((W // 2, result_y), bill["result"], font=font(25, True), fill=TEXT, anchor="ma")
    carried_y = result_y + 46
    result_line_f, _ = _fit_wrapped(draw, bill["result_line"], width=840, start=19, minimum=16, max_lines=2, bold=True)
    final_y = _draw_centered_wrapped(draw, bill["result_line"], cx=W // 2, y=carried_y, f=result_line_f, width=840, fill=MUTED, gap=4)
    if final_y > result_bottom - 24:
        raise RuntimeError(f"Explainer result box overflow for {bill['formal_title']}: {final_y}")
    _footer(draw, "EirePolitic · Draft review copy")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_explainer_v3", "warnings": [], "shared_body_font": body_font.size, "result_bottom": final_y}


def _stack(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, *, ta: int, nil: int, no_vote: int, total: int) -> None:
    cursor = x
    segments = [(ta, ACCENT), (nil, TEXT), (no_vote, NO_VOTE)]
    for i, (value, colour) in enumerate(segments):
        sw = x + w - cursor if i == len(segments) - 1 else round(w * value / total)
        if sw > 0:
            draw.rectangle((cursor, y, cursor + sw, y + h), fill=colour)
        cursor += sw
    draw.rectangle((x, y, x + w, y + h), outline=MUTED, width=2)


def render_party_vote(vote: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    _title(draw, vote["title"], top=56, max_width=940, start=38, minimum=29, max_lines=1)
    _rule(draw, 132)
    draw.text((W // 2, 165), vote["subtitle"], font=font(24, True), fill=ACCENT, anchor="ma")
    _stack(draw, 52, 208, 976, 57, ta=vote["ta"], nil=vote["nil"], no_vote=vote["no_recorded_vote"], total=vote["eligible"])

    legend_y = 292
    items = [("Tá", ACCENT), ("Níl", TEXT), ("No recorded vote", NO_VOTE)]
    starts = [330, 430, 545]
    for (label, colour), x in zip(items, starts):
        draw.rectangle((x, legend_y, x + 18, legend_y + 18), fill=colour, outline=MUTED)
        draw.text((x + 28, legend_y - 2), label, font=font(16, True), fill=TEXT, anchor="la")
    draw.text((W // 2, 332), f"{vote['eligible']} eligible TDs · {vote['ta']} Tá · {vote['nil']} Níl · {vote['abstain']} abstentions · {vote['no_recorded_vote']} no recorded vote", font=font(18, True), fill=TEXT, anchor="ma")
    draw.text((W // 2, 380), "PARTY BREAKDOWN", font=font(26, True), fill=ACCENT, anchor="ma")
    draw.text((858, 410), "Tá", font=font(14, True), fill=ACCENT, anchor="ma")
    draw.text((916, 410), "Níl", font=font(14, True), fill=TEXT, anchor="ma")
    draw.text((990, 410), "No vote", font=font(14, True), fill=MUTED, anchor="ma")

    bar_x, bar_w = 350, 470
    y = 442
    for row in vote["rows"]:
        party, eligible, ta, nil, nr = row
        if ta + nil + nr != eligible:
            raise RuntimeError(f"Party row does not reconcile for {party}: {row}")
        draw.text((50, y + 16), party, font=font(20, True), fill=TEXT, anchor="lm")
        _stack(draw, bar_x, y, bar_w, 36, ta=ta, nil=nil, no_vote=nr, total=eligible)
        draw.text((858, y + 18), str(ta), font=font(16, True), fill=ACCENT, anchor="mm")
        draw.text((916, y + 18), str(nil), font=font(16, True), fill=TEXT, anchor="mm")
        draw.text((990, y + 18), str(nr), font=font(16, True), fill=MUTED, anchor="mm")
        y += 72

    main_ta = sum(r[2] for r in vote["rows"]); main_nil = sum(r[3] for r in vote["rows"]); main_nr = sum(r[4] for r in vote["rows"])
    draw.text((W // 2, 1068), "Smaller groups", font=font(24, True), fill=ACCENT, anchor="ma")
    draw.text((W // 2, 1108), vote["smaller_groups"], font=font(16), fill=MUTED, anchor="ma")
    context_f, _ = _fit_wrapped(draw, vote["context"], width=850, start=15, minimum=13, max_lines=2)
    _draw_centered_wrapped(draw, vote["context"], cx=W // 2, y=1145, f=context_f, width=850, fill=MUTED, gap=3)
    _rule(draw, 1234, left=52, right=1028, width=3)
    draw.text((W // 2, 1260), "Row format: Tá · Níl · No vote", font=font(15, True), fill=MUTED, anchor="ma")
    draw.text((W // 2, 1288), "No recorded vote does not automatically mean absent.", font=font(13), fill=MUTED, anchor="ma")

    total_small_ta = vote["ta"] - main_ta; total_small_nil = vote["nil"] - main_nil; total_small_nr = vote["no_recorded_vote"] - main_nr
    if min(total_small_ta, total_small_nil, total_small_nr) < 0:
        raise RuntimeError("Displayed party rows exceed overall result")
    if vote["ta"] + vote["nil"] + vote["abstain"] + vote["no_recorded_vote"] != vote["eligible"]:
        raise RuntimeError("Overall vote does not reconcile to eligible membership")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_party_vote_v3", "warnings": [], "reconciled": True, "smaller_group_totals": [total_small_ta, total_small_nil, total_small_nr]}


def render_procedure(proc: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    _title(draw, proc["title"], top=70, max_width=960, start=38, minimum=29, max_lines=1)
    _rule(draw, 142)
    draw.text((W // 2, 184), proc["subtitle"], font=font(24, True), fill=ACCENT, anchor="ma")
    y = 255
    for step in proc["steps"]:
        box = (72, y, 1008, y + 205)
        _panel(draw, box, radius=20)
        draw.text((100, y + 30), step["title"], font=font(21, True), fill=ACCENT, anchor="la")
        body_f, _ = _fit_wrapped(draw, step["body"], width=850, start=21, minimum=18, max_lines=4)
        _draw_wrapped(draw, step["body"], xy=(100, y + 78), f=body_f, width=850, fill=TEXT, gap=5)
        y += 230
    box = (72, y + 2, 1008, y + 228)
    _panel(draw, box, radius=20, outline=ACCENT, width=3, fill=BG)
    draw.text((W // 2, y + 34), proc["result_title"], font=font(22, True), fill=ACCENT, anchor="ma")
    result_f, _ = _fit_wrapped(draw, proc["result_body"], width=850, start=20, minimum=17, max_lines=4)
    _draw_centered_wrapped(draw, proc["result_body"], cx=W // 2, y=y + 82, f=result_f, width=850, fill=TEXT, gap=5)
    _footer(draw, "EirePolitic · Parliamentary procedure · No recorded division")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_procedure_v3", "warnings": []}


def render_process_glossary(glossary: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    draw.text((W // 2, 75), "GLOSSARY", font=font(40, True), fill=TEXT, anchor="ma")
    draw.text((W // 2, 136), glossary["process_subtitle"], font=font(23, True), fill=ACCENT, anchor="ma")
    _rule(draw, 176, left=112, right=968, width=4)
    steps = glossary["process_steps"]
    start_x, y, bw, bh, gap = 35, 218, 105, 104, 15
    for idx, step in enumerate(steps):
        x = start_x + idx * (bw + gap)
        active = step["label"] == glossary["process_highlight"]
        fill = ACCENT if active else PANEL
        outline = ACCENT if active else PANEL_OUTLINE
        _panel(draw, (x, y, x + bw, y + bh), radius=12, fill=fill, outline=outline, width=2)
        number_fill = BG if active else ACCENT
        label_fill = BG if active else ACCENT
        draw.text((x + bw // 2, y + 24), str(step["number"]), font=font(14, True), fill=number_fill, anchor="mm")
        label_f, lines = _fit_wrapped(draw, step["label"], width=bw - 12, start=12, minimum=10, max_lines=2, bold=True)
        total = _measure_lines(draw, lines, label_f, 1); ly = y + 62 - total // 2
        for line in lines:
            draw.text((x + bw // 2, ly), line, font=label_f, fill=label_fill, anchor="ma")
            b = draw.textbbox((x + bw // 2, ly), line, font=label_f, anchor="ma"); ly = int(b[3] + 1)
        if idx < len(steps) - 1:
            ax = x + bw + 4
            draw.polygon([(ax, y + 52), (ax + 9, y + 45), (ax + 9, y + 59)], fill=ACCENT)
    draw.text((W // 2, 352), f"THIS POST: {glossary['process_highlight']}", font=font(18, True), fill=ACCENT, anchor="ma")
    _draw_centered_wrapped(draw, glossary["process_copy"], cx=W // 2, y=390, f=font(17), width=860, fill=TEXT, gap=4)
    draw.text((W // 2, 486), "COMMON TERMS", font=font(21, True), fill=ACCENT, anchor="ma")

    terms = glossary["terms"]
    boxes = [(60, 520, 515, 685), (565, 520, 1020, 685), (60, 710, 515, 875), (565, 710, 1020, 875), (312, 900, 768, 1065)]
    body_width = min(box[2] - box[0] - 46 for box in boxes)
    body_height = min(box[3] - (box[1] + 76) - 12 for box in boxes)
    body_font = _shared_body_font(draw, [t["body"] for t in terms], width=body_width, height=body_height, start=24, minimum=19, max_lines=4, gap=4)
    for term, box in zip(terms, boxes):
        _panel(draw, box, radius=16)
        draw.text(((box[0] + box[2]) // 2, box[1] + 34), term["term"], font=font(18, True), fill=ACCENT, anchor="ma")
        body_end = _draw_centered_wrapped(draw, term["body"], cx=(box[0] + box[2]) // 2, y=box[1] + 76, f=body_font, width=box[2] - box[0] - 46, fill=TEXT, gap=4)
        if body_end > box[3] - 12:
            raise RuntimeError(f"Process glossary definition overflow for {term['term']}: {body_end} > {box[3] - 12}")
    _footer(draw, "EirePolitic · Glossary · Parliamentary process & terms")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_process_glossary_v3", "warnings": [], "shared_body_font": body_font.size, "highlight": glossary["process_highlight"]}


def render_vote_glossary(glossary: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide(); draw = ImageDraw.Draw(im)
    draw.text((W // 2, 92), "GLOSSARY", font=font(42, True), fill=TEXT, anchor="ma")
    draw.text((W // 2, 150), "HOW TO READ THE VOTE SLIDES", font=font(23, True), fill=ACCENT, anchor="ma")
    _rule(draw, 196, left=112, right=968, width=4)
    terms = glossary["vote_terms"]
    y = 260
    heights = [175, 150, 200, 185]
    body_font = _shared_body_font(draw, [t["body"] for t in terms], width=820, height=105, start=24, minimum=19, max_lines=4, gap=4)
    for term, bh in zip(terms, heights):
        box = (82, y, 998, y + bh)
        _panel(draw, box, radius=16)
        draw.text((106, y + 24), term["term"], font=font(21, True), fill=ACCENT, anchor="la")
        body_end = _draw_wrapped(draw, term["body"], xy=(106, y + 66), f=body_font, width=820, fill=TEXT, gap=4)
        if body_end > box[3] - 12:
            raise RuntimeError(f"Vote glossary definition overflow for {term['term']}: {body_end} > {box[3] - 12}")
        y += bh + 20
    rule_box = (82, y, 998, min(y + 150, 1198))
    _panel(draw, rule_box, radius=16, outline=ACCENT, width=3, fill=BG)
    draw.text((W // 2, rule_box[1] + 28), "OUR VOTE-LABEL RULE", font=font(19, True), fill=ACCENT, anchor="ma")
    rule_f, _ = _fit_wrapped(draw, glossary["vote_rule"], width=820, start=17, minimum=15, max_lines=4)
    _draw_centered_wrapped(draw, glossary["vote_rule"], cx=W // 2, y=rule_box[1] + 72, f=rule_f, width=820, fill=TEXT, gap=4)
    _footer(draw, "EirePolitic · Glossary · Vote terms & safeguards")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); im.save(output)
    return {"renderer": "bill_tracker_vote_glossary_v3", "warnings": [], "shared_body_font": body_font.size}
