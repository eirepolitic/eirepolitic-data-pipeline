from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WIDTH = 1080
HEIGHT = 1350
OUT_DIR = Path("instagram/templates/layouts")

STYLES = [
    ("classic_interlace", "Classic medium-complexity interlace corners"),
    ("rounded_scroll", "Rounded flowing Celtic scroll corners"),
    ("dense_manuscript", "Dense manuscript-style knot corners"),
    ("long_floral", "Long corner flourishes with knot heads"),
    ("minimal_triquetra", "Small clean triquetra-style corners"),
]


def line(x: int, y: int, w: int, h: int, width: int = 4, color: str = "#f4ead7") -> dict[str, Any]:
    return {"type": "line", "x": x, "y": y, "w": w, "h": h, "fill": color, "width": width}


def rect(x: int, y: int, w: int, h: int, radius: int, width: int = 4, color: str = "#f4ead7") -> dict[str, Any]:
    return {"type": "rectangle", "x": x, "y": y, "w": w, "h": h, "radius": radius, "fill": None, "stroke": color, "width": width}


def mirror_element(e: dict[str, Any], horizontal: bool, vertical: bool) -> dict[str, Any]:
    e = dict(e)
    if e["type"] == "line":
        x1, y1 = e["x"], e["y"]
        x2, y2 = x1 + e["w"], y1 + e["h"]
        if horizontal:
            x1, x2 = WIDTH - x1, WIDTH - x2
        if vertical:
            y1, y2 = HEIGHT - y1, HEIGHT - y2
        e["x"], e["y"], e["w"], e["h"] = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    else:
        if horizontal:
            e["x"] = WIDTH - e["x"] - e["w"]
        if vertical:
            e["y"] = HEIGHT - e["y"] - e["h"]
    return e


def corner_base(style: str) -> list[dict[str, Any]]:
    if style == "classic_interlace":
        return [
            rect(20, 20, 82, 82, 34, 6), rect(56, 20, 82, 82, 34, 6), rect(20, 56, 82, 82, 34, 6),
            rect(34, 34, 48, 48, 18, 5), line(92, 28, 170, 0, 6), line(128, 28, 36, 34, 6),
            line(164, 62, 42, -34, 6), line(206, 28, 50, 0, 6), line(28, 92, 0, 170, 6),
            line(28, 128, 34, 36, 6), line(62, 164, -34, 42, 6), line(28, 206, 0, 50, 6),
        ]
    if style == "rounded_scroll":
        return [
            rect(18, 18, 100, 74, 34, 7), rect(48, 48, 84, 84, 40, 7), rect(22, 98, 70, 92, 34, 7),
            line(104, 24, 148, 0, 7), line(130, 24, 34, 26, 7), line(164, 50, 38, -26, 7),
            line(32, 116, 0, 166, 7), line(32, 154, 28, 34, 7), line(60, 188, -28, 36, 7),
            rect(112, 18, 58, 34, 17, 5), rect(18, 112, 34, 58, 17, 5), line(190, 28, 66, -18, 5), line(28, 190, -18, 66, 5),
        ]
    if style == "dense_manuscript":
        return [
            rect(18, 18, 68, 68, 18, 5), rect(54, 18, 68, 68, 18, 5), rect(90, 18, 68, 68, 18, 5),
            rect(18, 54, 68, 68, 18, 5), rect(54, 54, 68, 68, 18, 5), rect(18, 90, 68, 68, 18, 5),
            line(34, 34, 116, 116, 5), line(150, 34, -116, 116, 5), line(74, 22, 0, 150, 5), line(22, 74, 150, 0, 5),
            line(138, 34, 80, -10, 5), line(34, 138, -10, 80, 5), rect(138, 12, 52, 28, 14, 4), rect(12, 138, 28, 52, 14, 4),
        ]
    if style == "long_floral":
        return [
            rect(20, 20, 76, 76, 28, 5), rect(54, 20, 76, 76, 28, 5), rect(20, 54, 76, 76, 28, 5),
            line(88, 26, 210, 0, 5), line(150, 26, 56, 30, 5), line(206, 56, 72, -30, 5), line(246, 26, 54, -14, 4),
            line(26, 88, 0, 210, 5), line(26, 150, 30, 56, 5), line(56, 206, -30, 72, 5), line(26, 246, -14, 54, 4),
            rect(112, 14, 70, 28, 14, 4), rect(14, 112, 28, 70, 14, 4),
        ]
    # minimal_triquetra
    return [
        rect(18, 18, 76, 56, 28, 6), rect(58, 18, 76, 56, 28, 6), rect(38, 52, 76, 56, 28, 6),
        line(68, 26, 92, 0, 6), line(100, 26, 30, 26, 6), line(130, 52, 30, -26, 6),
        line(26, 68, 0, 92, 6), line(26, 100, 26, 30, 6), line(52, 130, -26, 30, 6),
    ]


def corner_elements(style: str) -> list[dict[str, Any]]:
    base = corner_base(style)
    out: list[dict[str, Any]] = []
    for prefix, horiz, vert in [
        ("top_left", False, False), ("top_right", True, False),
        ("bottom_left", False, True), ("bottom_right", True, True),
    ]:
        for idx, e in enumerate(base):
            item = mirror_element(e, horiz, vert)
            item["id"] = f"celtic_{style}_{prefix}_{idx:02d}"
            out.append(item)
    return out


def core_elements() -> list[dict[str, Any]]:
    white = "{palette.text_primary}"
    return [
        {"id": "member_photo", "type": "image", "placeholder": "member_photo", "x": 248, "y": 92, "w": 584, "h": 560, "fit": "cover", "radius": 0, "background": "{palette.panel_background_alt}"},
        {"id": "photo_border", "type": "rectangle", "x": 242, "y": 86, "w": 596, "h": 572, "radius": 0, "fill": None, "stroke": "{palette.border}", "width": 6},
        {"id": "member_name", "type": "text", "placeholder": "member_name", "x": 80, "y": 680, "w": 920, "h": 116, "style": {"font_family": "default_bold", "font_size": 66, "min_font_size": 36, "color": white, "align": "center", "valign": "middle", "max_lines": 2, "shrink_to_fit": True, "line_spacing": 8}},
        {"id": "name_rule", "type": "line", "x": 160, "y": 814, "w": 760, "h": 0, "fill": "{palette.border}", "width": 6},
        {"id": "constituency_label", "type": "text", "text": "Constituency:", "x": 150, "y": 844, "w": 340, "h": 58, "style": {"font_size": 34, "min_font_size": 22, "color": white, "align": "right", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "constituency", "type": "text", "placeholder": "constituency", "x": 520, "y": 844, "w": 410, "h": 58, "style": {"font_size": 38, "min_font_size": 21, "color": white, "align": "left", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "party_label", "type": "text", "text": "Party:", "x": 150, "y": 936, "w": 340, "h": 58, "style": {"font_size": 34, "min_font_size": 22, "color": white, "align": "right", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "party", "type": "text", "placeholder": "party", "x": 520, "y": 936, "w": 410, "h": 58, "style": {"font_size": 38, "min_font_size": 20, "color": white, "align": "left", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "top_issue_label", "type": "text", "text": "Top Issue:", "x": 150, "y": 1028, "w": 340, "h": 58, "style": {"font_size": 34, "min_font_size": 22, "color": white, "align": "right", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "top_issue", "type": "text", "placeholder": "top_issue", "x": 520, "y": 1028, "w": 410, "h": 58, "style": {"font_family": "default_bold", "font_size": 38, "min_font_size": 20, "color": white, "align": "left", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "vote_participation", "type": "text", "placeholder": "vote_participation", "x": 150, "y": 1160, "w": 340, "h": 72, "style": {"font_family": "default_bold", "font_size": 62, "min_font_size": 38, "color": white, "align": "center", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "vote_label", "type": "text", "text": "Vote Participation %", "x": 130, "y": 1230, "w": 380, "h": 42, "style": {"font_size": 26, "min_font_size": 18, "color": white, "align": "center", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "speech_rank", "type": "text", "placeholder": "speech_rank", "x": 590, "y": 1160, "w": 340, "h": 72, "style": {"font_family": "default_bold", "font_size": 62, "min_font_size": 38, "color": white, "align": "center", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
        {"id": "speech_label", "type": "text", "text": "Speech Rank", "x": 570, "y": 1230, "w": 380, "h": 42, "style": {"font_size": 26, "min_font_size": 18, "color": white, "align": "center", "valign": "middle", "max_lines": 1, "shrink_to_fit": True}},
    ]


def build_template(style: str, variant_number: int, description: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "template_id": f"profile_card_main_celtic_corner_v{variant_number}",
        "description": description,
        "width": WIDTH,
        "height": HEIGHT,
        "palette": "eirepolitic_dark",
        "background": {"type": "solid", "color": "{palette.panel_background}"},
        "elements": corner_elements(style) + core_elements(),
    }


def build_all(out_dir: Path = OUT_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, (style, description) in enumerate(STYLES, start=1):
        path = out_dir / f"profile_card_main_celtic_corner_v{idx}.json"
        path.write_text(json.dumps(build_template(style, idx, description), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate five deterministic Celtic-corner profile-card JSON templates.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    print(json.dumps({"success": True, "templates": build_all(Path(args.out_dir))}, indent=2))


if __name__ == "__main__":
    main()
