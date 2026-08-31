from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import FONT_CANDIDATES


PALETTE_ROOT = Path("instagram/templates/palettes")


@dataclass
class RenderResult:
    output_path: Path
    source_values_path: Path
    manifest_path: Path
    warnings: list[str]
    text_metrics: list[dict[str, Any]]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml_or_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return data


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


def _font_size(font: ImageFont.ImageFont, fallback: int) -> int:
    return int(getattr(font, "size", fallback))


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
) -> tuple[ImageFont.ImageFont, list[str], dict[str, Any]]:
    requested_size = int(style.get("font_size", 32))
    size = requested_size
    min_size = int(style.get("min_font_size", 16))
    max_lines = int(style.get("max_lines", 999))
    shrink = bool(style.get("shrink_to_fit", False))
    kind = str(style.get("font_family", "default_regular"))
    spacing = int(style.get("line_spacing", 8))
    original_line_count = 0
    truncated = False

    while True:
        font = load_font(kind, size)
        wrapped_lines = text_lines(draw, text, font, width)
        original_line_count = len(wrapped_lines)
        too_many_lines = bool(max_lines and len(wrapped_lines) > max_lines)
        lines = wrapped_lines
        truncated = False

        if too_many_lines and (not shrink or size <= min_size):
            lines = wrapped_lines[:max_lines]
            if lines:
                lines[-1] = ellipsize_to_width(draw, lines[-1], font, width)
            truncated = True

        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing) if lines else (0, 0, 0, 0)
        fits_height = (bbox[3] - bbox[1]) <= height
        fits_width = (bbox[2] - bbox[0]) <= width
        fits_line_count = not too_many_lines

        if not shrink or (fits_height and fits_width and fits_line_count) or size <= min_size:
            actual_size = _font_size(font, size)
            return font, lines, {
                "requested_font_size": requested_size,
                "actual_font_size": actual_size,
                "minimum_font_size": min_size,
                "font_shrunk": actual_size < requested_size,
                "line_count": len(lines),
                "original_line_count": original_line_count,
                "truncated": truncated,
                "max_lines": max_lines,
            }
        size -= 2


def draw_text_element(
    draw: ImageDraw.ImageDraw,
    element: Mapping[str, Any],
    bindings: Mapping[str, Any],
    palette: Mapping[str, str],
    warnings: list[str],
) -> dict[str, Any]:
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

    font, lines, fit = fit_text(draw, text, style, w, h)
    metric: dict[str, Any] = {
        "element_id": str(element.get("id") or placeholder or "text"),
        "placeholder": placeholder,
        "source_text": text,
        "element_box": [x, y, x + w, y + h],
        **fit,
        "rendered_lines": lines,
        "rendered_bbox": None,
        "clipped": False,
    }
    if not lines:
        return metric

    line_boxes: list[tuple[int, int, int, int]] = []
    line_heights: list[int] = []
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_boxes.append(bbox)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        total_height += line_height
    total_height += spacing * max(0, len(lines) - 1)

    cursor_y = y
    if valign == "middle":
        cursor_y = y + max(0, (h - total_height) // 2)
    elif valign == "bottom":
        cursor_y = y + max(0, h - total_height)

    left = x + w
    top = y + h
    right = x
    bottom = y
    for line, line_height, line_bbox in zip(lines, line_heights, line_boxes):
        line_width = line_bbox[2] - line_bbox[0]
        cursor_x = x
        if align == "center":
            cursor_x = x + max(0, (w - line_width) // 2)
        elif align == "right":
            cursor_x = x + max(0, w - line_width)

        # Pillow's textbbox can have non-zero left/top bearings. Treat cursor_x/cursor_y
        # as the intended glyph-bbox origin so the rendered geometry matches fit_text.
        draw_x = cursor_x - line_bbox[0]
        draw_y = cursor_y - line_bbox[1]
        draw.text((draw_x, draw_y), line, font=font, fill=color)
        actual_bbox = draw.textbbox((draw_x, draw_y), line, font=font)
        left = min(left, actual_bbox[0])
        top = min(top, actual_bbox[1])
        right = max(right, actual_bbox[2])
        bottom = max(bottom, actual_bbox[3])
        cursor_y += line_height + spacing

    metric["rendered_bbox"] = [left, top, right, bottom]
    metric["clipped"] = bool(left < x or top < y or right > x + w or bottom > y + h)
    if metric["clipped"]:
        warnings.append(f"text_clipped:{metric['element_id']}")
    if metric["truncated"]:
        warnings.append(f"text_truncated:{metric['element_id']}")
    return metric


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
    except Exception as exc:  # pragma: no cover
        warnings.append(f"image_load_error:{reference}:{exc}")
        return None


def rounded_mask(width: int, height: int, radius: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    return mask


def draw_image_element(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    element: Mapping[str, Any],
    bindings: Mapping[str, Any],
    palette: Mapping[str, str],
    warnings: list[str],
) -> None:
    placeholder = element.get("placeholder")
    reference = str(bindings.get(placeholder, "") if placeholder else element.get("source", ""))
    if placeholder and placeholder not in bindings:
        warnings.append(f"missing_binding:{placeholder}")
    x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
    background = resolve_palette_value(element.get("background"), palette)
    if background:
        radius = int(element.get("radius", 0) or 0)
        box = (x, y, x + w, y + h)
        if radius:
            draw.rounded_rectangle(box, radius=radius, fill=background)
        else:
            draw.rectangle(box, fill=background)

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
    base.paste(image, (x, y), mask)


def draw_rectangle(draw: ImageDraw.ImageDraw, element: Mapping[str, Any], palette: Mapping[str, str]) -> None:
    x, y, w, h = [int(element.get(key, 0)) for key in ["x", "y", "w", "h"]]
    fill = resolve_palette_value(element.get("fill", "#000000"), palette)
    outline = resolve_palette_value(element.get("outline", element.get("stroke")), palette)
    line_width = int(element.get("width", element.get("stroke_width", 1)) or 1)
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
    text_metrics: list[dict[str, Any]] = []

    for element in template.get("elements", []):
        element_type = element.get("type")
        if element_type == "rectangle":
            draw_rectangle(draw, element, palette)
        elif element_type == "text":
            text_metrics.append(draw_text_element(draw, element, bindings, palette, warnings))
        elif element_type == "image":
            draw_image_element(image, draw, element, bindings, palette, warnings)
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
        "text_metrics": text_metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "success": True,
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "template_id": template.get("template_id"),
        "renderer_version": "1.1",
        "warnings": warnings,
        "text_metrics": text_metrics,
    }, indent=2), encoding="utf-8")

    return RenderResult(
        output_path=output_path,
        source_values_path=source_path,
        manifest_path=manifest_path,
        warnings=warnings,
        text_metrics=text_metrics,
    )


def render_template_file(
    template_path: str | Path,
    bindings_path: str | Path,
    output_path: str | Path,
    palette_override: str | None = None,
) -> dict[str, Any]:
    template = load_json(template_path)
    if palette_override:
        template = dict(template)
        template["palette"] = palette_override

    binding_doc = load_yaml_or_json(bindings_path)
    bindings = binding_doc.get("bindings", binding_doc)
    if not isinstance(bindings, dict):
        raise RuntimeError(f"bindings must be a mapping in {bindings_path}")

    result = render_template(template, bindings, output_path)
    return load_json(result.manifest_path)
