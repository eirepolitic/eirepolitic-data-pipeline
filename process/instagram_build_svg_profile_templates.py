from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from process.instagram_build_celtic_svg_assets import build_all as build_svg_assets

WIDTH = 1080
HEIGHT = 1350
OUT_DIR = Path("instagram/templates/layouts")
ASSET_DIR = Path("instagram/assets/celtic_corners")

VARIANTS = [
    (1, "interlace_arch", 276, "High quality SVG classic interlace corner ornaments."),
    (2, "rounded_scroll", 300, "High quality SVG rounded scroll Celtic corner ornaments."),
    (3, "manuscript_panel", 256, "High quality SVG manuscript panel Celtic corner ornaments."),
    (4, "floral_long", 320, "High quality SVG long floral Celtic corner ornaments."),
    (5, "minimal_triquetra", 220, "High quality SVG minimal triquetra Celtic corner ornaments."),
]


def image(id_: str, source: str, x: int, y: int, w: int, h: int, flip_h: bool = False, flip_v: bool = False) -> dict[str, Any]:
    return {
        "id": id_,
        "type": "image",
        "source": source,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "fit": "stretch",
        "flip_horizontal": flip_h,
        "flip_vertical": flip_v,
    }


def corner_elements(asset: str, size: int) -> list[dict[str, Any]]:
    return [
        image("celtic_top_left", asset, 0, 0, size, size),
        image("celtic_top_right", asset, WIDTH - size, 0, size, size, True, False),
        image("celtic_bottom_left", asset, 0, HEIGHT - size, size, size, False, True),
        image("celtic_bottom_right", asset, WIDTH - size, HEIGHT - size, size, size, True, True),
    ]


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


def build_template(number: int, asset_name: str, size: int, description: str) -> dict[str, Any]:
    asset = f"{ASSET_DIR}/{asset_name}.svg"
    return {
        "schema_version": "1.0",
        "template_id": f"profile_card_main_svg_celtic_corner_v{number}",
        "description": description,
        "width": WIDTH,
        "height": HEIGHT,
        "palette": "eirepolitic_dark",
        "background": {"type": "solid", "color": "{palette.panel_background}"},
        "elements": corner_elements(asset, size) + core_elements(),
    }


def build_all(out_dir: Path = OUT_DIR) -> list[str]:
    build_svg_assets(ASSET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, asset_name, size, description in VARIANTS:
        path = out_dir / f"profile_card_main_svg_celtic_corner_v{number}.json"
        path.write_text(json.dumps(build_template(number, asset_name, size, description), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate profile-card templates that use high quality SVG Celtic corner assets.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    print(json.dumps({"success": True, "templates": build_all(Path(args.out_dir))}, indent=2))


if __name__ == "__main__":
    main()
