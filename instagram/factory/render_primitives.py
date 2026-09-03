from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1350
BG = "#0f2f24"
TEXT = "#f4ead7"
ACCENT = "#d8b45f"
MUTED = "#c8bda8"
TITLE_RULE_Y = 174
CORNER_DIR = Path("instagram/templates/assets")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int, *, bold: bool = True) -> ImageFont.ImageFont:
    for size in range(start, minimum - 1, -2):
        candidate = font(size, bold)
        if draw.textbbox((0, 0), text, font=candidate)[2] <= max_width:
            return candidate
    return font(minimum, bold)


def base_slide() -> Image.Image:
    missing = [name for name in ("corner_tl.png", "corner_tr.png", "corner_bl.png", "corner_br.png") if not (CORNER_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Approved Instagram corner assets are missing: {missing}")
    image = Image.new("RGB", (W, H), BG)
    for filename, position in [
        ("corner_tl.png", (0, 0)),
        ("corner_tr.png", (925, 0)),
        ("corner_bl.png", (0, 1195)),
        ("corner_br.png", (925, 1195)),
    ]:
        corner = Image.open(CORNER_DIR / filename).convert("RGBA").resize((155, 155), Image.Resampling.LANCZOS)
        image.paste(corner, position, corner)
    return image


def draw_title(image: Image.Image, title_lines: list[str]) -> None:
    draw = ImageDraw.Draw(image)
    if len(title_lines) == 1:
        draw.text((540, 100), title_lines[0], font=fit_font(draw, title_lines[0], 820, 56, 40), fill=TEXT, anchor="mm")
    else:
        title_font = font(45, True)
        draw.text((540, 76), title_lines[0], font=title_font, fill=TEXT, anchor="mm")
        draw.text((540, 128), title_lines[1], font=title_font, fill=TEXT, anchor="mm")
    draw.rectangle((112, TITLE_RULE_Y, 968, TITLE_RULE_Y + 4), fill=ACCENT)


def period_dates(period) -> str:
    return f"{period.start.day} {period.start.strftime('%b')} – {period.end.day} {period.end.strftime('%b')} {period.end.year}"


def remove_resampling_neutral_fringe(logo: Image.Image, *, max_spread: int = 18, max_value: int = 244) -> tuple[Image.Image, int]:
    rgb = logo.convert("RGB")
    cleaned: list[tuple[int, int, int]] = []
    changed = 0
    for pixel in rgb.getdata():
        low, high = min(pixel), max(pixel)
        if high <= max_value and (high - low) <= max_spread:
            cleaned.append((255, 255, 255))
            if pixel != (255, 255, 255):
                changed += 1
        else:
            cleaned.append(pixel)
    output = Image.new("RGB", rgb.size, "white")
    output.putdata(cleaned)
    return output, changed


def prepare_square_logo(
    logo: Image.Image,
    *,
    party_key: str,
    size: int,
    scale_overrides: dict[str, float] | None = None,
    neutral_cleanup_keys: set[str] | None = None,
) -> tuple[Image.Image, dict]:
    scale = float((scale_overrides or {}).get(party_key, 1.0))
    if scale > 1.0:
        crop_size = round(logo.width / scale)
        left = (logo.width - crop_size) // 2
        top = (logo.height - crop_size) // 2
        logo = logo.crop((left, top, left + crop_size, top + crop_size))
    logo = logo.resize((size, size), Image.Resampling.LANCZOS)
    cleaned = 0
    if party_key in (neutral_cleanup_keys or set()):
        logo, cleaned = remove_resampling_neutral_fringe(logo)
    return logo, {"artwork_scale": scale, "neutral_pixels_replaced": cleaned}


def contact_sheet(items: Iterable[tuple[str, Path]], out_path: Path, *, columns: int = 4) -> None:
    items = list(items)
    thumb_w, thumb_h, label_h, gap = 250, 312, 34, 18
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * (thumb_w + gap) + gap, rows * (thumb_h + label_h + gap) + gap), BG)
    draw = ImageDraw.Draw(canvas)
    label_font = font(18, True)
    for idx, (label, path) in enumerate(items):
        row, col = divmod(idx, columns)
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (thumb_w - image.width) // 2, y))
        draw.text((x + thumb_w // 2, y + thumb_h + 20), label, font=label_font, fill=TEXT, anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def carousel_sheet(items: Iterable[tuple[str, list[Path]]], out_path: Path) -> None:
    items = list(items)
    thumb_w, thumb_h, row_h, left = 162, 203, 225, 210
    canvas = Image.new("RGB", (left + 5 * thumb_w + 35, 30 + len(items) * row_h), BG)
    draw = ImageDraw.Draw(canvas)
    label_font = font(20, True)
    for row_idx, (label, paths) in enumerate(items):
        y = 20 + row_idx * row_h
        draw.text((20, y + thumb_h // 2), label, font=label_font, fill=TEXT, anchor="lm")
        for col_idx, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(image, (left + col_idx * thumb_w, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def draw_glossary(entries: list[tuple[str, str]], out_path: Path) -> dict:
    image = base_slide()
    draw_title(image, ["Glossary"])
    draw = ImageDraw.Draw(image)
    term_font, body_font = font(29, True), font(23)
    y = 225
    for term, body in entries:
        draw.text((135, y), term, font=term_font, fill=TEXT, anchor="la")
        bbox = draw.textbbox((135, y), term, font=term_font, anchor="la")
        underline_y = bbox[3] + 8
        draw.line((bbox[0], underline_y, bbox[2], underline_y), fill=ACCENT, width=2)
        body_y = underline_y + 22
        for line in textwrap.wrap(body, width=79):
            draw.text((135, body_y), line, font=body_font, fill=TEXT, anchor="la")
            body_y += 34
        y = body_y + 42
    if y > 1315:
        raise RuntimeError(f"Glossary overflowed slide: final y={y}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {"final_y": y, "entry_count": len(entries)}
