from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageChops


def _ratio(value: float) -> float:
    return round(float(value), 4)


def _image_element_metrics(element: Mapping[str, Any], reference: str) -> dict[str, Any]:
    slot_width = int(element.get("w", 0))
    slot_height = int(element.get("h", 0))
    fit = str(element.get("fit", "cover"))
    metrics: dict[str, Any] = {
        "element_id": str(element.get("id") or "image"),
        "fit": fit,
        "slot_width": slot_width,
        "slot_height": slot_height,
        "source": reference,
    }
    if not reference or reference.startswith(("http://", "https://")):
        metrics["measurable"] = False
        return metrics
    path = Path(reference)
    if not path.is_file():
        metrics["measurable"] = False
        return metrics
    with Image.open(path) as source:
        source_width, source_height = source.size
    metrics.update({
        "measurable": True,
        "source_width": source_width,
        "source_height": source_height,
    })
    if source_width <= 0 or source_height <= 0 or slot_width <= 0 or slot_height <= 0:
        return metrics
    if fit == "contain":
        scale = min(slot_width / source_width, slot_height / source_height)
        rendered_width = source_width * scale
        rendered_height = source_height * scale
    else:
        rendered_width = slot_width
        rendered_height = slot_height
    metrics.update({
        "rendered_width": round(rendered_width, 2),
        "rendered_height": round(rendered_height, 2),
        "horizontal_fill_ratio": _ratio(rendered_width / slot_width),
        "vertical_fill_ratio": _ratio(rendered_height / slot_height),
        "area_fill_ratio": _ratio((rendered_width * rendered_height) / (slot_width * slot_height)),
        "source_aspect_ratio": _ratio(source_width / source_height),
        "slot_aspect_ratio": _ratio(slot_width / slot_height),
    })
    return metrics


def _content_bbox_metrics(output_path: Path) -> dict[str, Any]:
    with Image.open(output_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 10 else 0)
    bbox = difference.getbbox()
    if bbox is None:
        return {
            "content_detected": False,
            "canvas_width": width,
            "canvas_height": height,
            "occupied_height_ratio": 0.0,
            "top_whitespace_ratio": 1.0,
            "bottom_whitespace_ratio": 1.0,
        }
    left, top, right, bottom = bbox
    return {
        "content_detected": True,
        "canvas_width": width,
        "canvas_height": height,
        "content_bbox": [left, top, right, bottom],
        "occupied_height_ratio": _ratio((bottom - top) / height),
        "occupied_width_ratio": _ratio((right - left) / width),
        "top_whitespace_ratio": _ratio(top / height),
        "bottom_whitespace_ratio": _ratio((height - bottom) / height),
    }


def validate_slide_layout(
    *,
    template: Mapping[str, Any],
    bindings: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    template_rules = template.get("validation", {}) if isinstance(template.get("validation"), dict) else {}
    whitespace = _content_bbox_metrics(output_path)

    if not whitespace["content_detected"]:
        errors.append("no_non_background_content_detected")
    max_top = float(template_rules.get("max_top_whitespace_ratio", 0.12))
    max_bottom = float(template_rules.get("max_bottom_whitespace_ratio", 0.08))
    min_occupied = float(template_rules.get("min_occupied_height_ratio", 0.78))
    if whitespace["top_whitespace_ratio"] > max_top:
        errors.append(f"top_whitespace_ratio:{whitespace['top_whitespace_ratio']}>{max_top}")
    if whitespace["bottom_whitespace_ratio"] > max_bottom:
        errors.append(f"bottom_whitespace_ratio:{whitespace['bottom_whitespace_ratio']}>{max_bottom}")
    if whitespace["occupied_height_ratio"] < min_occupied:
        errors.append(f"occupied_height_ratio:{whitespace['occupied_height_ratio']}<{min_occupied}")

    media_metrics: list[dict[str, Any]] = []
    for element in template.get("elements", []):
        if not isinstance(element, dict) or element.get("type") != "image":
            continue
        placeholder = element.get("placeholder")
        reference = str(bindings.get(placeholder, "") if placeholder else element.get("source", ""))
        metrics = _image_element_metrics(element, reference)
        media_metrics.append(metrics)
        if not metrics.get("measurable") or metrics.get("fit") != "contain":
            continue
        rules = element.get("validation", {}) if isinstance(element.get("validation"), dict) else {}
        min_vertical = float(rules.get("min_vertical_fill_ratio", 0.88))
        min_area = float(rules.get("min_area_fill_ratio", 0.78))
        if float(metrics.get("vertical_fill_ratio", 0.0)) < min_vertical:
            errors.append(f"media_vertical_fill:{metrics['element_id']}:{metrics['vertical_fill_ratio']}<{min_vertical}")
        if float(metrics.get("area_fill_ratio", 0.0)) < min_area:
            errors.append(f"media_area_fill:{metrics['element_id']}:{metrics['area_fill_ratio']}<{min_area}")

    return {
        "success": not errors,
        "errors": errors,
        "warnings": warnings,
        "whitespace": whitespace,
        "media": media_metrics,
        "thresholds": {
            "max_top_whitespace_ratio": max_top,
            "max_bottom_whitespace_ratio": max_bottom,
            "min_occupied_height_ratio": min_occupied,
        },
    }


def validate_visual_manifest(
    *,
    visual_manifest: Mapping[str, Any] | None,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    if not visual_manifest:
        return {"success": True, "errors": [], "warnings": [], "metrics": {}}

    rules = template.get("validation", {}) if isinstance(template.get("validation"), dict) else {}
    readability = visual_manifest.get("readability", {}) if isinstance(visual_manifest.get("readability"), dict) else {}
    errors: list[str] = []

    metrics = {
        "plot_vertical_fill_ratio": float(visual_manifest.get("plot_vertical_fill_ratio", 0.0)),
        "plot_area_ratio": float(visual_manifest.get("plot_area_ratio", 0.0)),
        "category_label_font_size": float(readability.get("category_label_font_size", 0.0)),
        "value_label_font_size": float(readability.get("value_label_font_size", 0.0)),
        "axis_font_size": float(readability.get("axis_font_size", 0.0)),
        "bar_thickness_px": float(readability.get("bar_thickness_px", 0.0)),
        "max_wrapped_label_lines": int(readability.get("max_wrapped_label_lines", 0)),
        "max_value_label_x_ratio": float(readability.get("max_value_label_x_ratio", 0.0)),
        "displayed_item_count": int(readability.get("displayed_item_count", 0)),
    }
    thresholds = {
        "min_plot_vertical_fill_ratio": float(rules.get("min_plot_vertical_fill_ratio", 0.0)),
        "min_plot_area_ratio": float(rules.get("min_plot_area_ratio", 0.0)),
        "min_category_label_font_size": float(rules.get("min_category_label_font_size", 0.0)),
        "min_value_label_font_size": float(rules.get("min_value_label_font_size", 0.0)),
        "min_axis_font_size": float(rules.get("min_axis_font_size", 0.0)),
        "min_bar_thickness_px": float(rules.get("min_bar_thickness_px", 0.0)),
        "max_wrapped_label_lines": int(rules.get("max_wrapped_label_lines", 0)),
        "max_value_label_x_ratio": float(rules.get("max_value_label_x_ratio", 1.0)),
    }

    minimum_checks = (
        ("plot_vertical_fill_ratio", "min_plot_vertical_fill_ratio"),
        ("plot_area_ratio", "min_plot_area_ratio"),
        ("category_label_font_size", "min_category_label_font_size"),
        ("value_label_font_size", "min_value_label_font_size"),
        ("axis_font_size", "min_axis_font_size"),
        ("bar_thickness_px", "min_bar_thickness_px"),
    )
    for metric_name, threshold_name in minimum_checks:
        threshold = thresholds[threshold_name]
        if threshold and metrics[metric_name] < threshold:
            errors.append(f"{metric_name}:{metrics[metric_name]}<{threshold}")

    max_lines = thresholds["max_wrapped_label_lines"]
    if max_lines and metrics["max_wrapped_label_lines"] > max_lines:
        errors.append(f"max_wrapped_label_lines:{metrics['max_wrapped_label_lines']}>{max_lines}")

    max_x = thresholds["max_value_label_x_ratio"]
    if metrics["displayed_item_count"] > 0 and metrics["max_value_label_x_ratio"] > max_x:
        errors.append(f"max_value_label_x_ratio:{metrics['max_value_label_x_ratio']}>{max_x}")

    return {
        "success": not errors,
        "errors": errors,
        "warnings": [],
        "metrics": metrics,
        "thresholds": thresholds,
    }
