from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1350
BG = "#0f2f24"
TEXT = "#f4ead7"
ACCENT = "#d8b45f"
MUTED = "#c8bda8"
GRID = "#315448"
SOFT_GREEN = "#88a99a"
SOFT_GOLD = "#b79a5a"
TITLE_RULE_Y = 174
CONTENT_LEFT = 112
CONTENT_RIGHT = 968
FOOTER_Y = 1288


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int, *, bold: bool = True) -> ImageFont.ImageFont:
    for size in range(start, minimum - 1, -2):
        candidate = font(size, bold)
        if draw.textbbox((0, 0), text, font=candidate)[2] <= max_width:
            return candidate
    return font(minimum, bold)


def _draw_corner_motif(draw: ImageDraw.ImageDraw, x: int, y: int, flip_x: bool, flip_y: bool) -> None:
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    for offset, length, width in ((0, 118, 4), (18, 84, 3), (36, 52, 2)):
        x0 = x + sx * offset
        y0 = y + sy * offset
        draw.line((x0, y0, x0 + sx * length, y0), fill=ACCENT, width=width)
        draw.line((x0, y0, x0, y0 + sy * length), fill=ACCENT, width=width)


def base_slide() -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    _draw_corner_motif(draw, 22, 22, False, False)
    _draw_corner_motif(draw, W - 22, 22, True, False)
    _draw_corner_motif(draw, 22, H - 22, False, True)
    _draw_corner_motif(draw, W - 22, H - 22, True, True)
    return image


def draw_title(image: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(image)
    title_font = fit_font(draw, title, 820, 54, 38, bold=True)
    draw.text((W // 2, 100), title, font=title_font, fill=TEXT, anchor="mm")
    draw.rectangle((CONTENT_LEFT, TITLE_RULE_Y, CONTENT_RIGHT, TITLE_RULE_Y + 4), fill=ACCENT)


def draw_kicker(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.text((W // 2, 211), text.upper(), font=font(21, True), fill=MUTED, anchor="mm")


def draw_footer(image: Image.Image, source_text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.line((CONTENT_LEFT, 1254, CONTENT_RIGHT, 1254), fill=GRID, width=2)
    draw.text((W // 2, FOOTER_Y), source_text, font=font(19), fill=MUTED, anchor="mm")


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def render_latest_support(
    rows: list[dict[str, Any]],
    *,
    title: str,
    model_date: str,
    source_text: str,
    output_path: Path,
) -> None:
    image = base_slide()
    draw_title(image, title)
    draw_kicker(image, f"Modelled party support · {model_date}")
    draw = ImageDraw.Draw(image)

    label_x = 122
    bar_left = 352
    bar_right = 900
    max_value = max([float(row["high"]) for row in rows] + [30.0])
    axis_max = max(30.0, ((max_value // 5) + 1) * 5)
    chart_top = 278
    row_h = 108
    bar_h = 30

    for tick in range(0, int(axis_max) + 1, 10):
        x = bar_left + (bar_right - bar_left) * tick / axis_max
        draw.line((x, chart_top - 22, x, chart_top + row_h * len(rows) - 35), fill=GRID, width=1)
        draw.text((x, chart_top - 34), f"{tick}%", font=font(17), fill=MUTED, anchor="ms")

    for idx, row in enumerate(rows):
        y = chart_top + idx * row_h
        label = str(row["label"])
        value = float(row["value"])
        low = float(row["low"])
        high = float(row["high"])
        draw.text((label_x, y + 15), label, font=fit_font(draw, label, 210, 26, 20, bold=True), fill=TEXT, anchor="lm")
        value_x = bar_left + (bar_right - bar_left) * value / axis_max
        low_x = bar_left + (bar_right - bar_left) * low / axis_max
        high_x = bar_left + (bar_right - bar_left) * high / axis_max
        draw.rounded_rectangle((bar_left, y, value_x, y + bar_h), radius=8, fill=ACCENT)
        whisker_y = y + bar_h + 19
        draw.line((low_x, whisker_y, high_x, whisker_y), fill=TEXT, width=3)
        draw.line((low_x, whisker_y - 7, low_x, whisker_y + 7), fill=TEXT, width=2)
        draw.line((high_x, whisker_y - 7, high_x, whisker_y + 7), fill=TEXT, width=2)
        value_anchor = min(value_x + 16, 955)
        draw.text((value_anchor, y + 15), _pct(value), font=font(24, True), fill=TEXT, anchor="lm")

    draw.text((CONTENT_LEFT, 1168), "Whiskers show the published uncertainty range.", font=font(20), fill=MUTED, anchor="lm")
    draw_footer(image, source_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_change(
    rows: list[dict[str, Any]],
    *,
    title: str,
    previous_date: str,
    latest_date: str,
    source_text: str,
    output_path: Path,
) -> None:
    image = base_slide()
    draw_title(image, title)
    draw_kicker(image, f"{previous_date} → {latest_date} · percentage-point change")
    draw = ImageDraw.Draw(image)

    label_x = 122
    center_x = 585
    span = 315
    chart_top = 286
    row_h = 105
    bar_h = 28
    max_abs = max([abs(float(row["value"])) for row in rows] + [0.5])
    axis_max = max(0.5, round(max_abs * 1.25, 1))

    draw.line((center_x, chart_top - 34, center_x, chart_top + row_h * len(rows) - 35), fill=MUTED, width=2)
    draw.text((center_x, chart_top - 47), "0", font=font(17), fill=MUTED, anchor="ms")

    for idx, row in enumerate(rows):
        y = chart_top + idx * row_h
        label = str(row["label"])
        value = float(row["value"])
        draw.text((label_x, y + 14), label, font=fit_font(draw, label, 240, 26, 20, bold=True), fill=TEXT, anchor="lm")
        end_x = center_x + span * value / axis_max
        left, right = sorted((center_x, end_x))
        fill = ACCENT if value >= 0 else SOFT_GREEN
        draw.rounded_rectangle((left, y, right, y + bar_h), radius=7, fill=fill)
        text = f"{value:+.2f} pp"
        if value >= 0:
            draw.text((min(end_x + 14, 956), y + 14), text, font=font(22, True), fill=TEXT, anchor="lm")
        else:
            draw.text((max(end_x - 14, 346), y + 14), text, font=font(22, True), fill=TEXT, anchor="rm")

    draw.text((CONTENT_LEFT, 1168), "This compares consecutive model dates, not two individual opinion polls.", font=font(20), fill=MUTED, anchor="lm")
    draw_footer(image, source_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_trend(
    series: list[dict[str, Any]],
    *,
    title: str,
    start_date: str,
    latest_date: str,
    source_text: str,
    output_path: Path,
) -> None:
    image = base_slide()
    draw_title(image, title)
    draw_kicker(image, f"{start_date} → {latest_date} · same election cycle")
    draw = ImageDraw.Draw(image)

    plot_left, plot_right = 145, 875
    plot_top, plot_bottom = 300, 1090
    colors = [ACCENT, TEXT, MUTED, SOFT_GREEN, SOFT_GOLD]

    all_values = [float(point["value"]) for item in series for point in item.get("points", [])]
    if not all_values:
        raise RuntimeError("Trend slide has no points")
    y_min = max(0.0, min(all_values) - 2.0)
    y_max = max(all_values) + 2.0
    if y_max - y_min < 8:
        y_max = y_min + 8

    dates = sorted({str(point["date"]) for item in series for point in item.get("points", [])})
    date_index = {date: idx for idx, date in enumerate(dates)}
    denominator = max(1, len(dates) - 1)

    for step in range(5):
        value = y_min + (y_max - y_min) * step / 4
        y = plot_bottom - (plot_bottom - plot_top) * step / 4
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((plot_left - 20, y), f"{value:.0f}%", font=font(17), fill=MUTED, anchor="rm")

    for idx, item in enumerate(series):
        color = colors[idx % len(colors)]
        points = []
        for point in item.get("points", []):
            date = str(point["date"])
            value = float(point["value"])
            x = plot_left + (plot_right - plot_left) * date_index[date] / denominator
            y = plot_bottom - (plot_bottom - plot_top) * (value - y_min) / (y_max - y_min)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")
        elif points:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        if points:
            end_x, end_y = points[-1]
            label = str(item["label"])
            latest_value = float(item["points"][-1]["value"])
            draw.text((end_x + 14, end_y), f"{label} {_pct(latest_value)}", font=font(19, True), fill=color, anchor="lm")

    if dates:
        for idx in (0, len(dates) // 2, len(dates) - 1):
            date = dates[idx]
            x = plot_left + (plot_right - plot_left) * idx / denominator
            label = date[5:]
            draw.text((x, plot_bottom + 28), label, font=font(17), fill=MUTED, anchor="ma")

    draw.text((CONTENT_LEFT, 1168), "Lines show modelled support over time; they are not individual poll results.", font=font(20), fill=MUTED, anchor="lm")
    draw_footer(image, source_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
