"""Review-only First Stage prototype renderer.

This module deliberately reuses the canonical Bill Tracker primitives/helpers.
Once the human approves the direction, migrate the approved component into
renderers.py before scaling the full First Stage periods.
"""
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
    _panel,
    _rule,
    _shared_body_font,
    _title,
)

PANEL_W = 456


def render_first_stage_bill(bill: dict[str, Any], output: str | Path) -> dict[str, Any]:
    im = base_slide()
    draw = ImageDraw.Draw(im)

    title_bottom = _title(
        draw,
        bill["formal_title"],
        top=38,
        max_width=900,
        start=34,
        minimum=24,
        max_lines=2,
    )
    rule_y = max(116, title_bottom + 20)
    _rule(draw, rule_y)

    stage_y = rule_y + 34
    draw.text(
        (W // 2, stage_y),
        f"FIRST STAGE · BILL {bill['bill_number']}",
        font=font(23, True),
        fill=ACCENT,
        anchor="ma",
    )
    meta_y = stage_y + 36
    draw.text(
        (W // 2, meta_y),
        f"{bill['origin_house']} · {bill['introduction_date']}",
        font=font(17, True),
        fill=MUTED,
        anchor="ma",
    )
    sponsor_y = meta_y + 31
    sponsor_font, _ = _fit_wrapped(
        draw,
        f"Sponsors · {bill['sponsors']}",
        width=880,
        start=16,
        minimum=14,
        max_lines=2,
    )
    sponsor_end = _draw_centered_wrapped(
        draw,
        f"Sponsors · {bill['sponsors']}",
        cx=W // 2,
        y=sponsor_y,
        f=sponsor_font,
        width=880,
        fill=MUTED,
        gap=3,
    )

    box_top = sponsor_end + 23
    box_h = 285
    gap = 24
    boxes = [
        (60, box_top, 516, box_top + box_h),
        (564, box_top, 1020, box_top + box_h),
        (60, box_top + box_h + gap, 516, box_top + 2 * box_h + gap),
        (564, box_top + box_h + gap, 1020, box_top + 2 * box_h + gap),
    ]
    labels = [
        "WHAT THE BILL PROPOSES",
        "PRACTICAL EFFECT",
        "WHY IT WAS INTRODUCED",
        "WHERE IT IS NOW",
    ]
    texts = [bill["what"], bill["effect"], bill["why"], bill["where"]]

    body_font = _shared_body_font(
        draw,
        texts,
        width=PANEL_W - 48,
        height=202,
        start=24,
        minimum=18,
        max_lines=8,
        gap=4,
    )

    for box, label, text in zip(boxes, labels, texts):
        _panel(draw, box)
        label_font, _ = _fit_wrapped(
            draw, label, width=PANEL_W - 48, start=18, minimum=16, max_lines=2, bold=True
        )
        label_end = _draw_wrapped(
            draw,
            label,
            xy=(box[0] + 24, box[1] + 19),
            f=label_font,
            width=PANEL_W - 48,
            fill=ACCENT,
            gap=2,
        )
        body_y = max(box[1] + 72, label_end + 12)
        body_end = _draw_wrapped(
            draw,
            text,
            xy=(box[0] + 24, body_y),
            f=body_font,
            width=PANEL_W - 48,
            fill=TEXT,
            gap=4,
        )
        if body_end > box[3] - 16:
            raise RuntimeError(f"First Stage body overflow for {label}: {body_end} > {box[3] - 16}")

    note_top = boxes[2][3] + 27
    note_bottom = 1211
    _panel(draw, (60, note_top, 1020, note_bottom), radius=18, outline=ACCENT, width=3, fill=BG)
    draw.text((W // 2, note_top + 27), "WHAT FIRST STAGE MEANS", font=font(20, True), fill=ACCENT, anchor="ma")
    note = (
        "This is the formal initiation step, not a finding that the House supports the Bill’s general principles. "
        "A recorded member-by-member vote is not normally required here; substantive principle debate usually comes at Second Stage."
    )
    note_font, _ = _fit_wrapped(draw, note, width=850, start=18, minimum=16, max_lines=4)
    note_end = _draw_centered_wrapped(
        draw, note, cx=W // 2, y=note_top + 68, f=note_font, width=850, fill=TEXT, gap=4
    )
    if note_end > note_bottom - 18:
        raise RuntimeError(f"First Stage note overflow: {note_end} > {note_bottom - 18}")

    _footer(draw, f"EirePolitic · Source: Houses of the Oireachtas · Bill {bill['bill_number']}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    im.save(output)
    return {
        "renderer": "bill_tracker_first_stage_prototype_v1_review_only",
        "warnings": [],
        "shared_body_font": body_font.size,
        "source_footer": True,
    }
