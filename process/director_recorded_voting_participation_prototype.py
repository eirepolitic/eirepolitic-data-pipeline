#!/usr/bin/env python3
"""Render one representative Director-review slide for recorded voting participation.

This is intentionally a single-slide prototype for the visual-direction gate.
The composition is rendered as one coordinated canvas so text, bars, ornaments,
rule and footer can be collision-checked together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.constants import FONT_CANDIDATES

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"

W, H = 1080, 1350
COLORS = {
    "background": "#0f2f24",
    "panel": "#173d30",
    "panel_alt": "#214a3b",
    "text": "#f4ead7",
    "muted": "#cbbf9f",
    "accent": "#d8b45f",
    "accent_2": "#9ec5a2",
    "grid": "#6c8978",
}


def font_path(kind: str) -> str | None:
    key = "bold" if kind == "bold" else "regular"
    for candidate in FONT_CANDIDATES[key]:
        if Path(candidate).exists():
            return candidate
    return None


def font(kind: str, size: int) -> ImageFont.ImageFont:
    path = font_path(kind)
    return ImageFont.truetype(path, size=size) if path else ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, ft: ImageFont.ImageFont, *, anchor: str = "la", spacing: int = 4) -> tuple[int, int, int, int]:
    box = draw.multiline_textbbox(xy, text, font=ft, anchor=anchor, spacing=spacing)
    return tuple(int(v) for v in box)


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def within(box: tuple[int, int, int, int], outer: tuple[int, int, int, int], pad: int = 0) -> bool:
    return box[0] >= outer[0] + pad and box[1] >= outer[1] + pad and box[2] <= outer[2] - pad and box[3] <= outer[3] - pad


def fit_single_line(draw: ImageDraw.ImageDraw, text: str, *, kind: str, max_size: int, min_size: int, max_width: int) -> ImageFont.ImageFont:
    for size in range(max_size, min_size - 1, -1):
        ft = font(kind, size)
        if draw.textbbox((0, 0), text, font=ft)[2] <= max_width:
            return ft
    return font(kind, min_size)


def fit_wrapped(draw: ImageDraw.ImageDraw, text: str, *, kind: str, max_size: int, min_size: int, max_width: int, max_lines: int) -> tuple[ImageFont.ImageFont, str]:
    words = str(text).split()
    for size in range(max_size, min_size - 1, -1):
        ft = font(kind, size)
        lines: list[str] = []
        current = ""
        for word in words:
            probe = word if not current else f"{current} {word}"
            if draw.textbbox((0, 0), probe, font=ft)[2] <= max_width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) <= max_lines:
            return ft, "\n".join(lines)
    ft = font(kind, min_size)
    return ft, text


def register(elements: list[dict[str, Any]], *, element_id: str, kind: str, bbox: tuple[int, int, int, int], row: int | None = None) -> None:
    elements.append({"id": element_id, "kind": kind, "bbox": list(bbox), "row": row})


def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, y: int, ft: ImageFont.ImageFont, fill: str, *, spacing: int = 4) -> tuple[int, int, int, int]:
    box = text_bbox(draw, (W // 2, y), text, ft, anchor="ma", spacing=spacing)
    draw.multiline_text((W // 2, y), text, font=ft, fill=fill, anchor="ma", align="center", spacing=spacing)
    return box


def render_slide(parties: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    image = Image.new("RGB", (W, H), COLORS["background"])
    draw = ImageDraw.Draw(image)
    elements: list[dict[str, Any]] = []

    ornament_ft = font("regular", 72)
    ornament = "❦"
    ornament_specs = [
        ("ornament_tl", (22, 18), "la"),
        ("ornament_tr", (W - 22, 18), "ra"),
        ("ornament_bl", (22, H - 22), "ls"),
        ("ornament_br", (W - 22, H - 22), "rs"),
    ]
    for element_id, xy, anchor in ornament_specs:
        box = text_bbox(draw, xy, ornament, ornament_ft, anchor=anchor)
        draw.text(xy, ornament, font=ornament_ft, fill=COLORS["accent"], anchor=anchor)
        register(elements, element_id=element_id, kind="ornament", bbox=box)

    title_text = "Recorded voting participation — parties"
    title_ft = fit_single_line(draw, title_text, kind="bold", max_size=44, min_size=38, max_width=900)
    title_box = draw_centered_text(draw, title_text, 78, title_ft, COLORS["text"])
    register(elements, element_id="title", kind="text", bbox=title_box)

    subtitle_ft = font("regular", 22)
    subtitle = "Share of eligible Dáil division opportunities with a recorded Tá, Níl or formal abstention"
    subtitle_box = draw_centered_text(draw, subtitle, 145, subtitle_ft, COLORS["muted"])
    register(elements, element_id="subtitle", kind="text", bbox=subtitle_box)

    qualifier_ft = font("regular", 18)
    qualifier = "Alphabetical order · recorded participation is not an attendance measure"
    qualifier_box = draw_centered_text(draw, qualifier, 181, qualifier_ft, COLORS["muted"])
    register(elements, element_id="qualifier", kind="text", bbox=qualifier_box)

    rule = (70, 222, 1010, 230)
    draw.rectangle(rule, fill=COLORS["accent"])
    register(elements, element_id="top_rule", kind="rule", bbox=rule)

    panel = (58, 258, 1022, 1192)
    draw.rounded_rectangle(panel, radius=28, fill=COLORS["panel"], outline=COLORS["panel_alt"], width=2)
    register(elements, element_id="panel", kind="container", bbox=panel)

    header_ft = font("bold", 15)
    draw.text((86, 282), "PARTY / GROUP", font=header_ft, fill=COLORS["muted"])
    draw.text((982, 282), "RATE  ·  RECORDED / ELIGIBLE", font=header_ft, fill=COLORS["muted"], anchor="ra")
    register(elements, element_id="header_left", kind="text", bbox=text_bbox(draw, (86, 282), "PARTY / GROUP", header_ft))
    register(elements, element_id="header_right", kind="text", bbox=text_bbox(draw, (982, 282), "RATE  ·  RECORDED / ELIGIBLE", header_ft, anchor="ra"))

    chart_top = 320
    chart_bottom = 1144
    row_count = len(parties)
    row_h = (chart_bottom - chart_top) / row_count
    label_x = 86
    label_w = 320
    bar_x = 420
    bar_w = 370
    value_x = 982
    max_rate = 100.0

    grid_top = chart_top + 4
    grid_bottom = chart_bottom - 5
    for tick in (25, 50, 75, 100):
        gx = int(bar_x + bar_w * tick / max_rate)
        draw.line((gx, grid_top, gx, grid_bottom), fill=COLORS["grid"], width=1)
        tick_ft = font("regular", 12)
        draw.text((gx, chart_bottom + 8), f"{tick}%", font=tick_ft, fill=COLORS["muted"], anchor="ma")

    for idx, row in enumerate(parties.itertuples(index=False)):
        y0 = int(chart_top + idx * row_h)
        y1 = int(chart_top + (idx + 1) * row_h)
        cy = (y0 + y1) // 2
        if idx:
            draw.line((80, y0, 994, y0), fill=COLORS["panel_alt"], width=1)

        name = str(row.party_name)
        name_ft, name_text = fit_wrapped(draw, name, kind="bold", max_size=20, min_size=16, max_width=label_w, max_lines=2)
        name_box = text_bbox(draw, (label_x, cy), name_text, name_ft, anchor="lm", spacing=2)
        draw.multiline_text((label_x, cy), name_text, font=name_ft, fill=COLORS["text"], anchor="lm", spacing=2)
        register(elements, element_id=f"party_{idx}_name", kind="text", bbox=name_box, row=idx)

        pct = float(row.recorded_participation_pct)
        bar_height = 34
        bx1 = int(bar_x + bar_w * max(0.0, min(max_rate, pct)) / max_rate)
        bar_box = (bar_x, cy - bar_height // 2, bx1, cy + bar_height // 2)
        draw.rounded_rectangle((bar_x, cy - bar_height // 2, bar_x + bar_w, cy + bar_height // 2), radius=8, fill=COLORS["panel_alt"])
        draw.rounded_rectangle(bar_box, radius=8, fill=COLORS["accent"])
        register(elements, element_id=f"party_{idx}_bar", kind="bar", bbox=bar_box, row=idx)

        pct_ft = font("bold", 22)
        pct_text = f"{pct:.1f}%"
        pct_box = text_bbox(draw, (value_x, cy - 5), pct_text, pct_ft, anchor="rs")
        draw.text((value_x, cy - 5), pct_text, font=pct_ft, fill=COLORS["text"], anchor="rs")
        register(elements, element_id=f"party_{idx}_pct", kind="text", bbox=pct_box, row=idx)

        denom_ft = font("regular", 14)
        numerator = int(row.recorded_participation_opportunities)
        denominator = int(row.eligible_division_opportunities)
        denom_text = f"{numerator:,} / {denominator:,}"
        denom_box = text_bbox(draw, (value_x, cy + 11), denom_text, denom_ft, anchor="ra")
        draw.text((value_x, cy + 11), denom_text, font=denom_ft, fill=COLORS["muted"], anchor="ra")
        register(elements, element_id=f"party_{idx}_denom", kind="text", bbox=denom_box, row=idx)

    footer_ft = font("regular", 16)
    footer_y = 1278
    left_footer = "@eirepolitic"
    right_footer = "28 Feb–28 Aug 2026 · 136 Dáil divisions"
    draw.text((66, footer_y), left_footer, font=footer_ft, fill=COLORS["muted"], anchor="la")
    draw.text((1014, footer_y), right_footer, font=footer_ft, fill=COLORS["muted"], anchor="ra")
    left_footer_box = text_bbox(draw, (66, footer_y), left_footer, footer_ft)
    right_footer_box = text_bbox(draw, (1014, footer_y), right_footer, footer_ft, anchor="ra")
    register(elements, element_id="footer_left", kind="text", bbox=left_footer_box)
    register(elements, element_id="footer_right", kind="text", bbox=right_footer_box)

    source_ft = font("regular", 12)
    source_text = "Source: Houses of the Oireachtas · EirePolitic production data"
    source_box = text_bbox(draw, (W // 2, 1310), source_text, source_ft, anchor="ma")
    draw.text((W // 2, 1310), source_text, font=source_ft, fill=COLORS["muted"], anchor="ma")
    register(elements, element_id="source", kind="text", bbox=source_box)

    collisions: list[dict[str, Any]] = []
    text_elements = [e for e in elements if e["kind"] == "text"]
    for i, a in enumerate(text_elements):
        for b in text_elements[i + 1:]:
            if a.get("row") is not None and a.get("row") == b.get("row") and {a["id"].split("_")[-1], b["id"].split("_")[-1]} == {"pct", "denom"}:
                continue
            if intersects(tuple(a["bbox"]), tuple(b["bbox"]), pad=2):
                collisions.append({"a": a["id"], "b": b["id"], "a_bbox": a["bbox"], "b_bbox": b["bbox"]})

    for e in text_elements:
        if e.get("row") is None:
            continue
        bar = next(x for x in elements if x["id"] == f"party_{e['row']}_bar")
        if intersects(tuple(e["bbox"]), tuple(bar["bbox"]), pad=5):
            collisions.append({"a": e["id"], "b": bar["id"], "a_bbox": e["bbox"], "b_bbox": bar["bbox"]})

    out_of_bounds = [e for e in elements if e["kind"] in {"text", "ornament"} and not within(tuple(e["bbox"]), (0, 0, W, H), pad=4)]
    panel_text_ids = [e for e in text_elements if e["id"].startswith("party_") or e["id"].startswith("header_")]
    panel_out = [e for e in panel_text_ids if not within(tuple(e["bbox"]), panel, pad=16)]

    row_spacing_issues: list[dict[str, Any]] = []
    for idx in range(row_count - 1):
        row_a = [e for e in text_elements if e.get("row") == idx]
        row_b = [e for e in text_elements if e.get("row") == idx + 1]
        max_bottom = max(e["bbox"][3] for e in row_a)
        min_top = min(e["bbox"][1] for e in row_b)
        gap = min_top - max_bottom
        if gap < 7:
            row_spacing_issues.append({"rows": [idx, idx + 1], "gap_px": gap})

    qa = {
        "canvas": [W, H],
        "text_collision_count": len(collisions),
        "text_collisions": collisions,
        "out_of_bounds_count": len(out_of_bounds),
        "out_of_bounds": out_of_bounds,
        "panel_text_out_of_bounds_count": len(panel_out),
        "panel_text_out_of_bounds": panel_out,
        "row_spacing_issue_count": len(row_spacing_issues),
        "row_spacing_issues": row_spacing_issues,
        "ornament_count": len([e for e in elements if e["kind"] == "ornament"]),
        "top_rule_present": any(e["id"] == "top_rule" for e in elements),
        "panel_present": any(e["id"] == "panel" for e in elements),
        "footer_present": any(e["id"] == "footer_left" for e in elements) and any(e["id"] == "footer_right" for e in elements),
        "elements": elements,
    }
    qa["pass"] = (
        qa["text_collision_count"] == 0
        and qa["out_of_bounds_count"] == 0
        and qa["panel_text_out_of_bounds_count"] == 0
        and qa["row_spacing_issue_count"] == 0
        and qa["ornament_count"] == 4
        and qa["top_rule_present"]
        and qa["panel_present"]
        and qa["footer_present"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return qa


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(EVIDENCE / "party_participation.csv")
    parties = parties.sort_values(["party_name", "party_uri"], kind="stable").reset_index(drop=True)

    slide_path = OUT / "prototype_party_slide_v2.png"
    qa = render_slide(parties, slide_path)
    (OUT / "prototype_party_slide_v2_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "status": "PASS" if qa["pass"] else "FAIL",
        "prototype_only": True,
        "visual_direction_gate": "pending_human_review",
        "slide": str(slide_path),
        "period": {"start": "2026-02-28", "end": "2026-08-28"},
        "division_count": 136,
        "party_count": len(parties),
        "ordering": "alphabetical",
        "metric": "recorded participation opportunities / eligible division opportunities",
        "denominator_display": "separate right-side recorded / eligible column",
        "canonical_reference": "EirePolitic dark analytical framing: floral corner ornaments, centered title, gold rule, analytical panel and split footer; bars retain party_issue_monthly_profile_v2 analytical treatment",
        "computer_visual_qa": {
            "text_collision_count": qa["text_collision_count"],
            "out_of_bounds_count": qa["out_of_bounds_count"],
            "panel_text_out_of_bounds_count": qa["panel_text_out_of_bounds_count"],
            "row_spacing_issue_count": qa["row_spacing_issue_count"],
            "ornament_count": qa["ornament_count"],
            "pass": qa["pass"],
        },
        "editorial_caveat": "Recorded voting participation does not by itself measure a TD's attendance, workload, effectiveness, or overall job performance.",
        "publication_enabled": False,
    }
    (OUT / "prototype_manifest_v2.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if qa["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
