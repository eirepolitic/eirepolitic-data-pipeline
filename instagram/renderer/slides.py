from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from .charts import build_bar_chart
from .constants import DEFAULT_PALETTE, SlideGeometry
from .fonts import font_set
from .util import clamp_text, draw_wrapped_text, load_image_from_reference, rounded_panel


def merged_palette(context: dict[str, Any]) -> dict[str, str]:
    palette = dict(DEFAULT_PALETTE)
    palette.update(context.get("branding", {}).get("palette", {}))
    return palette


def new_slide(context: dict[str, Any]) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, str], dict[str, Any]]:
    width = int(context["slide_size"]["width"])
    height = int(context["slide_size"]["height"])
    image = Image.new("RGBA", (width, height), merged_palette(context)["bg"])
    draw = ImageDraw.Draw(image)
    return image, draw, merged_palette(context), font_set()


def draw_header(draw: ImageDraw.ImageDraw, slide: dict[str, Any], branding: dict[str, Any], fonts: dict, palette: dict, box: Sequence[int]) -> int:
    x0, y0, _, _ = box
    title = slide.get("title") or ""
    account = branding.get("account_name") or ""
    draw.text((x0, y0), title, font=fonts["headline"], fill=palette["text"])
    title_box = draw.textbbox((x0, y0), title, font=fonts["headline"])
    draw.text((x0, title_box[3] + 10), account, font=fonts["label"], fill=palette["muted"])
    return title_box[3] + 56


def draw_footer(draw: ImageDraw.ImageDraw, context: dict[str, Any], fonts: dict, palette: dict, y: int, width: int) -> None:
    footer = context.get("branding", {}).get("footer_note") or ""
    if footer:
        draw.text((72, y), footer, font=fonts["footer"], fill=palette["muted"])
    source = " | ".join(Path(value).name if "/" in value else value for value in context.get("datasets_used", {}).values())
    source = clamp_text(f"Sources: {source}", 80)
    bbox = draw.textbbox((0, 0), source, font=fonts["footer"])
    draw.text((width - bbox[2] - 72, y), source, font=fonts["footer"], fill=palette["muted"])


def render_overview_slide(context: dict[str, Any], slide: dict[str, Any]) -> Image.Image:
    image, draw, palette, fonts = new_slide(context)
    geo = SlideGeometry(width=image.width, height=image.height)
    cursor_y = draw_header(draw, slide, context["branding"], fonts, palette, (geo.margin_x, geo.margin_top, image.width - geo.margin_x, 0))

    hero_box = (geo.margin_x, cursor_y, image.width - geo.margin_x, cursor_y + 360)
    rounded_panel(draw, hero_box, geo.panel_radius, palette["panel_alt"], outline=palette["border"])
    img = load_image_from_reference(context["constituency"]["image_url"], hero_box[2]-hero_box[0]-36, hero_box[3]-hero_box[1]-36)
    if img:
        image.alpha_composite(img, (hero_box[0] + 18, hero_box[1] + 18))
    else:
        draw.text((hero_box[0] + 36, hero_box[1] + 42), context["constituency"]["name"], font=fonts["title"], fill=palette["text"])
        draw.text((hero_box[0] + 36, hero_box[1] + 135), "No constituency image indexed yet", font=fonts["body"], fill=palette["muted"])

    cursor_y = hero_box[3] + geo.gap
    draw.text((geo.margin_x, cursor_y), context["constituency"]["name"], font=fonts["title"], fill=palette["text"])
    cursor_y += 96

    metric_cards = [
        ("TDs", str(context["constituency"]["member_count"])),
        ("Parties", str(context["constituency"]["party_count"])),
        ("Issue-labelled speeches", str(context["constituency"]["speech_count"])),
        ("Top issue", context["constituency"]["top_issue_label"]),
    ]
    card_width = (image.width - geo.margin_x * 2 - geo.gap) // 2
    card_height = 150
    for idx, (label, value) in enumerate(metric_cards):
        row = idx // 2
        col = idx % 2
        x0 = geo.margin_x + col * (card_width + geo.gap)
        y0 = cursor_y + row * (card_height + geo.gap)
        box = (x0, y0, x0 + card_width, y0 + card_height)
        rounded_panel(draw, box, 26, palette["panel"], outline=palette["border"])
        draw.text((x0 + 24, y0 + 22), label, font=fonts["metric_label"], fill=palette["muted"])
        draw_wrapped_text(draw, value, fonts["metric"], palette["text"], (x0 + 24, y0 + 58, x0 + card_width - 24, y0 + card_height - 18), line_spacing=4, max_lines=2)

    footer_y = image.height - geo.margin_bottom
    draw_footer(draw, context, fonts, palette, footer_y - 28, image.width)
    return image


def render_member_profile_slide(context: dict[str, Any], slide: dict[str, Any]) -> Image.Image:
    image, draw, palette, fonts = new_slide(context)
    geo = SlideGeometry(width=image.width, height=image.height)
    cursor_y = draw_header(draw, slide, context["branding"], fonts, palette, (geo.margin_x, geo.margin_top, image.width - geo.margin_x, 0))

    left_box = (geo.margin_x, cursor_y, geo.margin_x + 360, cursor_y + 520)
    rounded_panel(draw, left_box, geo.panel_radius, palette["panel_alt"], outline=palette["border"])
    member_img = load_image_from_reference(context["member"]["photo_url"], left_box[2]-left_box[0]-24, left_box[3]-left_box[1]-24)
    if member_img:
        image.alpha_composite(member_img, (left_box[0] + 12, left_box[1] + 12))
    else:
        draw.text((left_box[0] + 28, left_box[1] + 42), "Photo unavailable", font=fonts["subhead"], fill=palette["muted"])

    right_box = (left_box[2] + geo.gap, cursor_y, image.width - geo.margin_x, cursor_y + 520)
    rounded_panel(draw, right_box, geo.panel_radius, palette["panel"], outline=palette["border"])
    draw_wrapped_text(draw, context["member"]["full_name"], fonts["headline"], palette["text"], (right_box[0] + 24, right_box[1] + 22, right_box[2] - 24, right_box[1] + 150), line_spacing=6, max_lines=2)
    draw.text((right_box[0] + 24, right_box[1] + 160), context["member"]["party"], font=fonts["subhead"], fill=palette["muted"])
    draw.text((right_box[0] + 24, right_box[1] + 214), context["member"]["constituency"], font=fonts["body"], fill=palette["muted"])
    draw.text((right_box[0] + 24, right_box[1] + 286), "Top issue", font=fonts["metric_label"], fill=palette["muted"])
    draw_wrapped_text(draw, context["member"]["top_issue_label"], fonts["metric"], palette["text"], (right_box[0] + 24, right_box[1] + 324, right_box[2] - 24, right_box[1] + 450), line_spacing=4, max_lines=2)
    draw.text((right_box[0] + 24, right_box[1] + 460), f"Issue-labelled speeches: {context['member']['speech_count']}", font=fonts["body"], fill=palette["muted"])

    bio_box = (geo.margin_x, cursor_y + 520 + geo.gap, image.width - geo.margin_x, image.height - 170)
    rounded_panel(draw, bio_box, geo.panel_radius, palette["panel"], outline=palette["border"])
    draw.text((bio_box[0] + 24, bio_box[1] + 22), "Background", font=fonts["subhead"], fill=palette["text"])
    draw_wrapped_text(
        draw,
        clamp_text(context["member"]["background"], 360),
        fonts["body"],
        palette["text"],
        (bio_box[0] + 24, bio_box[1] + 84, bio_box[2] - 24, bio_box[3] - 24),
        line_spacing=10,
        max_lines=8,
    )

    draw_footer(draw, context, fonts, palette, image.height - 100, image.width)
    return image


def render_top_issues_slide(context: dict[str, Any], slide: dict[str, Any]) -> Image.Image:
    image, draw, palette, fonts = new_slide(context)
    geo = SlideGeometry(width=image.width, height=image.height)
    cursor_y = draw_header(draw, slide, context["branding"], fonts, palette, (geo.margin_x, geo.margin_top, image.width - geo.margin_x, 0))

    panel_box = (geo.margin_x, cursor_y, image.width - geo.margin_x, image.height - 170)
    rounded_panel(draw, panel_box, geo.panel_radius, palette["panel"], outline=palette["border"])

    scope = slide.get("content", {}).get("scope", "constituency")
    rows = context["member_issue_rows"] if scope == "member" else context["constituency_issue_rows"]
    scope_title = context["member"]["full_name"] if scope == "member" else context["constituency"]["name"]
    chart = build_bar_chart(rows, panel_box[2]-panel_box[0]-36, panel_box[3]-panel_box[1]-110, palette, title=scope_title)
    image.alpha_composite(chart, (panel_box[0] + 18, panel_box[1] + 74))

    summary = "Counts show speeches with a classified issue label from the existing debate issue dataset."
    draw.text((panel_box[0] + 24, panel_box[1] + 20), summary, font=fonts["small"], fill=palette["muted"])
    draw_footer(draw, context, fonts, palette, image.height - 100, image.width)
    return image


def render_glossary_slide(context: dict[str, Any], slide: dict[str, Any]) -> Image.Image:
    image, draw, palette, fonts = new_slide(context)
    geo = SlideGeometry(width=image.width, height=image.height)
    cursor_y = draw_header(draw, slide, context["branding"], fonts, palette, (geo.margin_x, geo.margin_top, image.width - geo.margin_x, 0))

    boxes = [
        (geo.margin_x, cursor_y, image.width - geo.margin_x, cursor_y + 240),
        (geo.margin_x, cursor_y + 264, image.width - geo.margin_x, cursor_y + 504),
        (geo.margin_x, cursor_y + 528, image.width - geo.margin_x, image.height - 170),
    ]
    glossary_items = [
        ("Issues", context["glossary"].get("issues", "Issue labels come from the debate issue classifier.")),
        ("Vote participation", context["glossary"].get("vote_participation_pct", "Placeholder until a dedicated source table is connected.")),
        ("Speech rank", context["glossary"].get("speech_rank", "Placeholder until a dedicated source table is connected.")),
    ]
    for box, (title, body) in zip(boxes, glossary_items):
        rounded_panel(draw, box, geo.panel_radius, palette["panel"], outline=palette["border"])
        draw.text((box[0] + 24, box[1] + 22), title, font=fonts["subhead"], fill=palette["text"])
        draw_wrapped_text(draw, clamp_text(body, 260), fonts["body"], palette["text"], (box[0] + 24, box[1] + 84, box[2] - 24, box[3] - 24), line_spacing=10, max_lines=6)

    draw_footer(draw, context, fonts, palette, image.height - 100, image.width)
    return image


SLIDE_RENDERERS = {
    "constituency_overview": render_overview_slide,
    "member_profile": render_member_profile_slide,
    "top_issues": render_top_issues_slide,
    "glossary": render_glossary_slide,
}


def render_slides(context: dict[str, Any], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index, slide in enumerate(context["slides"], start=1):
        renderer = SLIDE_RENDERERS.get(slide["type"])
        if renderer is None:
            raise RuntimeError(f"Unsupported slide type: {slide['type']}")
        image = renderer(context, slide)
        out_path = png_dir / f"{index:02d}_{slide['key']}.png"
        image.convert("RGB").save(out_path, format="PNG")
        paths.append(out_path)
        manifest.append({
            "index": index,
            "key": slide["key"],
            "type": slide["type"],
            "path": str(out_path),
        })

    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
