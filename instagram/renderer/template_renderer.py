from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import FONT_CANDIDATES


PALETTE_ROOT = Path("instagram/templates/palettes")


@dataclass
class RenderResult:
    output_path: Path
    source_values_path: Path
    manifest_path: Path
    warnings: list[str]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_palette(palette_id: str) -> dict[str, str]:
    palette_path = PALETTE_ROOT / f"{palette_id}.json"
    if not palette_path.exists():
        raise FileNotFoundError(f"Missing palette: {palette_path}")
    data = load_json(palette_path)
    return dict(data.get("colors", {}))


def resolve_palette_value(value: Any, palette: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return palette.get(key, match.group(0))

    return re.sub(r"\{palette\.([A-Za-z0-9_]+)\}", replace, value)


def font_path(kind: str) -> str | None:
    key = "bold" if kind in {"default_bold", "bold"} else "regular"
    for path in FONT_CANDIDATES.get(key, []):
        if Path(path).exists():
            return path
    return None


def load_font(kind: str, size: int) -> ImageFont.ImageFont:
    path = font_path(kind)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if draw.textbbox((0, 0), probe, font=font)[2] <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def ellipsize_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    text = str(text or "")
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "…"
    while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > max_width:
        text = text[:-1]
    return (text.rstrip() or "") + suffix


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    style: Mapping[str, Any],
    width: int,
    height: int,
) -> tuple[ImageFont.ImageFont, list[str]]:
    size = int(style.get("font_size", 32))
    min_size = int(style.get("min_font_size", 16))
    max_lines = int(style.get("max_lines", 999))
    shrink = bool(style.get("shrink_to_fit", False))
    kind = str(style.get("font_family", "default_regular"))

    while True:
        font = load_font(kind, size)
        lines = text_lines(draw, text, font, width)
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = ellipsize_to_width(draw, lines[-1], font, width)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=int(style.get("line_spacing", 8))) if lines else (0, 0, 0, 0)
        fits_height = (bbox[3] - bbox[1]) <= height
        if not shrink or (fits_height and len(lines) <= max_lines) or size <= min_size:
            return font, lines
        size -= 2


def draw_text_element(draw: ImageDraw.ImageDraw, element: Mapping[str, Any], bindings: Mapping[str, Any], palette: Mapping[str, str], warnings: list[str]) -> None:
    placeholder = element.get("placeholder")
    text = str(bindings.get(placeholder, "") if placeholder else element.get("text", ""))
    if placeholder and placeholder not in bindings:
        warnings.append(f"missing_binding:{placeholder}")
    x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
    style = dict(element.get("style", {}))
    color = resolve_palette_value(style.get("color", "#000000"), palette)
    align = str(style.get("align", "left"))
    valign = str(style.get("valign", "top"))
    spacing = int(style.get("line_spacing", 8))

    font, lines = fit_text(draw, text, style, w, h)
    if not lines:
        return

    line_heights = []
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        total_height += line_height
    total_height += spacing * max(0, len(lines) - 1)

    cursor_y = y
    if valign == "middle":
        cursor_y = y + max(0, (h - total_height) // 2)
    elif valign == "bottom":
        cursor_y = y + max(0, h - total_height)

    for line, line_height in zip(lines, line_heights):
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        cursor_x = x
        if align == "center":
            cursor_x = x + max(0, (w - line_width) // 2)
        elif align == "right":
            cursor_x = x + max(0, w - line_width)
        draw.text((cursor_x, cursor_y), line, font=font, fill=color)
        cursor_y += line_height + spacing


def load_image(reference: str, warnings: list[str]) -> Image.Image | None:
    if not reference:
        warnings.append("missing_image_reference")
        return None
    try:
        if reference.startswith(("http://", "https://")):
            response = requests.get(reference, timeout=20)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        path = Path(reference)
        if path.exists():
            return Image.open(path).convert("RGBA")
        warnings.append(f"image_not_found:{reference}")
        return None
    except Exception as exc:  # pragma: no cover - network/path dependent
        warnings.append(f"image_load_error:{reference}:{exc}")
        return None


def rounded_mask(width: int, height: int, radius: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    return mask


def draw_image_element(base: Image.Image, draw: ImageDraw.ImageDraw, element: Mapping[str, Any], bindings: Mapping[str, Any], warnings: list[str]) -> None:
    placeholder = element.get("placeholder")
    reference = str(bindings.get(placeholder, "") if placeholder else element.get("source", ""))
    if placeholder and placeholder not in bindings:
        warnings.append(f"missing_binding:{placeholder}")
    x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
    image = load_image(reference, warnings)
    if image is None:
        draw.line((x + 24, y + 24, x + w - 24, y + h - 24), fill="#ffffff", width=3)
        draw.line((x + w - 24, y + 24, x + 24, y + h - 24), fill="#ffffff", width=3)
        return
    fit = element.get("fit", "cover")
    if fit == "contain":
        image.thumbnail((w, h), Image.Resampling.LANCZOS)
        pasted = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pasted.alpha_composite(image, ((w - image.width) // 2, (h - image.height) // 2))
        image = pasted
    elif fit == "stretch":
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    else:
        image = ImageOps.fit(image, (w, h), method=Image.Resampling.LANCZOS)
    radius = int(element.get("radius", 0) or 0)
    mask = rounded_mask(w, h, radius) if radius else image.getchannel("A")
    base.alpha_composite(image, (x, y), mask)


def draw_rectangle(draw: ImageDraw.ImageDraw, element: Mapping[str, Any], palette: Mapping[str, str]) -> None:
    x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
    fill = resolve_palette_value(element.get("fill", "#000000"), palette)
    outline = resolve_palette_value(element.get("outline"), palette)
    line_width = int(element.get("width", 1) or 1)
    radius = int(element.get("radius", 0) or 0)
    box = (x, y, x + w, y + h)
    if radius:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=line_width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=line_width)


def render_template(template: Mapping[str, Any], bindings: Mapping[str, Any], output_path: str | Path) -> RenderResult:
    palette_id = str(template.get("palette", "eirepolitic_dark"))
    palette = load_palette(palette_id)
    width = int(template["width"])
    height = int(template["height"])
    background = template.get("background", {})
    background_color = resolve_palette_value(background.get("color", "#ffffff"), palette)
    image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    warnings: list[str] = []

    for element in template.get("elements", []):
        element_type = element.get("type")
        if element_type == "rectangle":
            draw_rectangle(draw, element, palette)
        elif element_type == "text":
            draw_text_element(draw, element, bindings, palette, warnings)
        elif element_type == "image":
            draw_image_element(image, draw, element, bindings, warnings)
        elif element_type == "line":
            x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
            fill = resolve_palette_value(element.get("fill", "#ffffff"), palette)
            draw.line((x, y, x + w, y + h), fill=fill, width=int(element.get("width", 2)))
        else:
            warnings.append(f"unsupported_element:{element.get('id')}:{element_type}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")

    metadata_dir = output_path.parent.parent / "metadata" if output_path.parent.name == "png" else output_path.parent / "metadata"
    source_dir = metadata_dir / "source_values"
    manifest_dir = metadata_dir / "manifests"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    source_path = source_dir / f"{stem}.source_values.json"
    manifest_path = manifest_dir / f"{stem}.render_manifest.json"

    source_path.write_text(json.dumps({
        "template_id": template.get("template_id"),
        "palette": palette_id,
        "bindings": dict(bindings),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "success": True,
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "template_id": template.get("template_id"),
        "renderer_version": "1.0",
        "warnings": warnings,
    }, indent=2), encoding="utf-8")

    return RenderResult(output_path=output_path, source_values_path=source_path, manifest_path=manifest_path, warnings=warnings)
