from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .catalogues import CatalogueSet, CatalogueValidationError, REPO_ROOT, load_catalogues

VALID_PROJECT_STATUSES = {"draft", "approved", "deprecated"}
VALID_DIRECTIONS = {"ascending", "descending"}
REQUIRED_TOP_LEVEL = {
    "project_id",
    "version",
    "status",
    "purpose",
    "granularity",
    "period",
    "slides",
    "validation",
    "output",
    "review",
    "schedule",
}
REQUIRED_GRANULARITY = {
    "grain",
    "key_fields",
    "label_field",
    "source_metric",
    "selector",
    "ordering",
}
REQUIRED_SLIDE = {
    "slide_id",
    "order",
    "post_type_id",
    "text",
    "accessibility",
    "fallback_behavior",
}
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ProjectValidationError(ValueError):
    """Raised when a Content Factory project specification is invalid."""


def load_project(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_file():
        raise ProjectValidationError(f"Project specification does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProjectValidationError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectValidationError("Project specification root must be a mapping")
    return payload


def _require_mapping(prefix: str, value: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be a mapping")
        return None
    return value


def _require_string(prefix: str, value: Any, errors: list[str]) -> str:
    text = str(value or "").strip()
    if not text:
        errors.append(f"{prefix} must be a non-empty string")
    return text


def _require_string_list(prefix: str, value: Any, errors: list[str], *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{prefix} must be a list of non-empty strings")
        return []
    if not allow_empty and not value:
        errors.append(f"{prefix} must not be empty")
    return value


def _provided_variables(slide: dict[str, Any]) -> set[str]:
    provided: set[str] = set()
    for section_name in ("text", "media"):
        section = slide.get(section_name)
        if isinstance(section, dict):
            provided.update(str(key) for key in section)
    if isinstance(slide.get("visual"), dict):
        provided.add("main_media")
    return provided


def _validate_granularity(
    project: dict[str, Any],
    catalogues: CatalogueSet,
    errors: list[str],
    warnings: list[str],
) -> None:
    granularity = _require_mapping("granularity", project.get("granularity"), errors)
    if granularity is None:
        return
    missing = sorted(REQUIRED_GRANULARITY - set(granularity))
    if missing:
        errors.append(f"granularity missing required fields: {', '.join(missing)}")
        return
    grain = _require_string("granularity.grain", granularity.get("grain"), errors)
    _require_string_list("granularity.key_fields", granularity.get("key_fields"), errors)
    _require_string("granularity.label_field", granularity.get("label_field"), errors)
    source_metric = _require_string("granularity.source_metric", granularity.get("source_metric"), errors)
    _require_mapping("granularity.selector", granularity.get("selector"), errors)
    ordering = _require_mapping("granularity.ordering", granularity.get("ordering"), errors)
    if ordering:
        _require_string("granularity.ordering.field", ordering.get("field"), errors)
        direction = _require_string("granularity.ordering.direction", ordering.get("direction"), errors)
        if direction and direction not in VALID_DIRECTIONS:
            errors.append(
                f"granularity.ordering.direction must be one of {sorted(VALID_DIRECTIONS)}; got {direction!r}"
            )
    metric = catalogues.metrics.get(source_metric)
    if metric is None:
        errors.append(f"granularity.source_metric references unknown metric_id: {source_metric}")
    else:
        if metric.get("status") == "deprecated":
            errors.append(f"granularity.source_metric references deprecated metric: {source_metric}")
        metric_grain = str(metric.get("grain") or "")
        if metric_grain and metric_grain not in {grain, "multi_grain"}:
            errors.append(
                f"granularity.grain {grain!r} is incompatible with source metric grain {metric_grain!r}"
            )
        if metric.get("status") == "draft":
            warnings.append(f"granularity.source_metric {source_metric} remains draft")


def _validate_visual(
    slide_prefix: str,
    slide: dict[str, Any],
    post_type: dict[str, Any],
    catalogues: CatalogueSet,
    errors: list[str],
    warnings: list[str],
) -> None:
    visual = slide.get("visual")
    if visual is None:
        return
    visual = _require_mapping(f"{slide_prefix}.visual", visual, errors)
    if visual is None:
        return
    visual_type_id = _require_string(f"{slide_prefix}.visual.visual_type_id", visual.get("visual_type_id"), errors)
    metric_id = _require_string(f"{slide_prefix}.visual.metric_id", visual.get("metric_id"), errors)
    visual_type = catalogues.visual_types.get(visual_type_id)
    metric = catalogues.metrics.get(metric_id)
    if visual_type is None:
        errors.append(f"{slide_prefix}.visual references unknown visual_type_id: {visual_type_id}")
    else:
        if visual_type.get("status") == "deprecated":
            errors.append(f"{slide_prefix}.visual references deprecated visual type: {visual_type_id}")
        if visual_type.get("status") == "draft":
            warnings.append(f"{slide_prefix}.visual type {visual_type_id} remains draft")
        bindings = visual.get("bindings")
        if not isinstance(bindings, dict):
            errors.append(f"{slide_prefix}.visual.bindings must be a mapping")
        else:
            missing_bindings = sorted(set(visual_type.get("required_bindings", [])) - set(bindings))
            if missing_bindings:
                errors.append(
                    f"{slide_prefix}.visual.bindings missing required bindings for {visual_type_id}: "
                    f"{', '.join(missing_bindings)}"
                )
        item_limit = visual.get("item_limit")
        if item_limit is not None:
            if not isinstance(item_limit, int) or isinstance(item_limit, bool) or item_limit < 1:
                errors.append(f"{slide_prefix}.visual.item_limit must be a positive integer")
            else:
                recommended = visual_type.get("recommended_item_count", {})
                maximum = recommended.get("max") if isinstance(recommended, dict) else None
                if isinstance(maximum, int) and item_limit > maximum:
                    warnings.append(
                        f"{slide_prefix}.visual.item_limit {item_limit} exceeds recommended maximum {maximum} "
                        f"for {visual_type_id}"
                    )
    if metric is None:
        errors.append(f"{slide_prefix}.visual references unknown metric_id: {metric_id}")
    else:
        if metric.get("status") == "deprecated":
            errors.append(f"{slide_prefix}.visual references deprecated metric: {metric_id}")
        if metric.get("status") == "draft":
            warnings.append(f"{slide_prefix}.visual metric {metric_id} remains draft")
    supported_slot_types = set(post_type.get("supported_slot_types", []))
    if "visual" not in supported_slot_types:
        errors.append(
            f"{slide_prefix}.post_type_id {post_type.get('post_type_id')} does not support a visual slot"
        )
    slots = post_type.get("slots", {})
    compatible_slots = [
        slot_id
        for slot_id, slot in slots.items()
        if isinstance(slot, dict) and slot.get("type") in {"visual", "visual_or_image"}
    ] if isinstance(slots, dict) else []
    if not compatible_slots:
        errors.append(f"{slide_prefix}.post_type_id has no declared visual-compatible slot")


def _validate_slides(
    project: dict[str, Any],
    catalogues: CatalogueSet,
    errors: list[str],
    warnings: list[str],
) -> None:
    slides = project.get("slides")
    if not isinstance(slides, list) or not slides:
        errors.append("slides must be a non-empty list")
        return
    slide_ids: set[str] = set()
    orders: set[int] = set()
    for position, slide in enumerate(slides, start=1):
        prefix = f"slides[{position}]"
        if not isinstance(slide, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        missing = sorted(REQUIRED_SLIDE - set(slide))
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue
        slide_id = _require_string(f"{prefix}.slide_id", slide.get("slide_id"), errors)
        if slide_id in slide_ids:
            errors.append(f"Duplicate slide_id: {slide_id}")
        slide_ids.add(slide_id)
        order = slide.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            errors.append(f"{prefix}.order must be a positive integer")
        elif order in orders:
            errors.append(f"Duplicate slide order: {order}")
        else:
            orders.add(order)
        post_type_id = _require_string(f"{prefix}.post_type_id", slide.get("post_type_id"), errors)
        post_type = catalogues.post_types.get(post_type_id)
        if post_type is None:
            errors.append(f"{prefix}.post_type_id references unknown post type: {post_type_id}")
            continue
        if post_type.get("status") == "deprecated":
            errors.append(f"{prefix}.post_type_id references deprecated post type: {post_type_id}")
        if post_type.get("status") == "draft":
            warnings.append(f"{prefix}.post_type_id {post_type_id} remains draft")
        for mapping_name in ("text", "accessibility", "fallback_behavior"):
            _require_mapping(f"{prefix}.{mapping_name}", slide.get(mapping_name), errors)
        has_visual = "visual" in slide
        has_media = "media" in slide
        if has_visual and has_media:
            errors.append(f"{prefix} cannot define both visual and media")
        provided = _provided_variables(slide)
        missing_variables = sorted(set(post_type.get("required_variables", [])) - provided)
        if missing_variables:
            errors.append(
                f"{prefix} does not provide required variables for {post_type_id}: {', '.join(missing_variables)}"
            )
        if has_visual:
            _validate_visual(prefix, slide, post_type, catalogues, errors, warnings)
        elif has_media and "image" not in set(post_type.get("supported_slot_types", [])):
            errors.append(f"{prefix}.post_type_id {post_type_id} does not support image media")
    if orders and orders != set(range(1, len(slides) + 1)):
        errors.append("slide order values must be contiguous starting at 1")


def _validate_control_sections(project: dict[str, Any], errors: list[str]) -> None:
    period = _require_mapping("period", project.get("period"), errors)
    if period:
        _require_string("period.mode", period.get("mode"), errors)
    validation = _require_mapping("validation", project.get("validation"), errors)
    if validation:
        scenarios = _require_string_list("validation.scenarios", validation.get("scenarios"), errors)
        required_scenarios = {"minimum", "maximum", "real_example"}
        missing = sorted(required_scenarios - set(scenarios))
        if missing:
            errors.append(f"validation.scenarios missing required scenarios: {', '.join(missing)}")
        _require_mapping("validation.real_example_selector", validation.get("real_example_selector"), errors)
        if validation.get("require_explicit_approval") is not True:
            errors.append("validation.require_explicit_approval must be true")
    output = _require_mapping("output", project.get("output"), errors)
    if output:
        _require_string("output.s3_prefix", output.get("s3_prefix"), errors)
        _require_string("output.preview_branch", output.get("preview_branch"), errors)
    review = _require_mapping("review", project.get("review"), errors)
    if review:
        if review.get("require_all_items_reviewed") is not True:
            errors.append("review.require_all_items_reviewed must be true")
        if not isinstance(review.get("allow_targeted_regeneration"), bool):
            errors.append("review.allow_targeted_regeneration must be boolean")
    schedule = _require_mapping("schedule", project.get("schedule"), errors)
    if schedule and not isinstance(schedule.get("enabled"), bool):
        errors.append("schedule.enabled must be boolean")


def validate_project(
    project: dict[str, Any] | None = None,
    *,
    project_path: str | Path | None = None,
    catalogues: CatalogueSet | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if project is None:
        if project_path is None:
            raise ProjectValidationError("project or project_path is required")
        try:
            project = load_project(project_path)
        except ProjectValidationError as exc:
            return {"success": False, "errors": [str(exc)], "warnings": []}
    if catalogues is None:
        try:
            catalogues = load_catalogues()
        except CatalogueValidationError as exc:
            return {
                "success": False,
                "errors": [f"Catalogue validation failed: {line}" for line in str(exc).splitlines()],
                "warnings": [],
            }
    missing_top = sorted(REQUIRED_TOP_LEVEL - set(project))
    if missing_top:
        errors.append(f"Project missing required top-level fields: {', '.join(missing_top)}")
        return {"success": False, "errors": errors, "warnings": warnings}
    project_id = _require_string("project_id", project.get("project_id"), errors)
    if project_id and not PROJECT_ID_PATTERN.fullmatch(project_id):
        errors.append("project_id must contain only lowercase letters, numbers, and underscores")
    version = project.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        errors.append("version must be a positive integer")
    status = _require_string("status", project.get("status"), errors)
    if status and status not in VALID_PROJECT_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_PROJECT_STATUSES)}; got {status!r}")
    _require_string("purpose", project.get("purpose"), errors)
    _validate_granularity(project, catalogues, errors, warnings)
    _validate_slides(project, catalogues, errors, warnings)
    _validate_control_sections(project, errors)
    if status == "approved":
        draft_warnings = [warning for warning in warnings if "remains draft" in warning]
        if draft_warnings:
            errors.append("Approved projects cannot reference draft catalogue entries")
    return {
        "success": not errors,
        "project_id": project_id or None,
        "version": version,
        "errors": errors,
        "warnings": warnings,
        "slide_count": len(project.get("slides", [])) if isinstance(project.get("slides"), list) else 0,
    }
