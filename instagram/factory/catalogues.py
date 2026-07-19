from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_DIR = REPO_ROOT / "instagram" / "catalogues"
DEFAULT_CATALOGUE_PATHS = {
    "post_types": CATALOGUE_DIR / "post_types.yml",
    "visual_types": CATALOGUE_DIR / "visual_types.yml",
    "metrics": CATALOGUE_DIR / "metrics.yml",
}
VALID_STATUSES = {"draft", "approved", "deprecated"}

POST_REQUIRED_FIELDS = {
    "post_type_id",
    "display_name",
    "description",
    "status",
    "layout_path",
    "supported_aspect_ratios",
    "supported_slot_types",
    "required_variables",
    "optional_variables",
    "text_limits",
    "slots",
    "fallback_rules",
    "example_preview",
    "version",
}
VISUAL_REQUIRED_FIELDS = {
    "visual_type_id",
    "display_name",
    "renderer",
    "template_path",
    "status",
    "intended_use",
    "required_bindings",
    "optional_bindings",
    "supported_data_shapes",
    "recommended_item_count",
    "supported_slot_fit",
    "known_limitations",
    "test_contact_sheet",
    "version",
}
METRIC_REQUIRED_FIELDS = {
    "metric_id",
    "display_name",
    "description",
    "status",
    "source_logical_keys",
    "layer",
    "grain",
    "measure_fields",
    "dimension_fields",
    "period_fields",
    "mapping_config",
    "freshness_expectation",
    "source_note",
    "validation_rules",
    "privacy_classification",
    "external_verification",
    "version",
}


class CatalogueValidationError(ValueError):
    """Raised when one or more catalogue entries are invalid."""


@dataclass(frozen=True)
class CatalogueSet:
    post_types: dict[str, dict[str, Any]]
    visual_types: dict[str, dict[str, Any]]
    metrics: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            "post_types": self.post_types,
            "visual_types": self.visual_types,
            "metrics": self.metrics,
        }


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise CatalogueValidationError(f"Catalogue file does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CatalogueValidationError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogueValidationError(f"Catalogue root must be a mapping: {resolved}")
    return payload


def _index_entries(
    entries: Any,
    *,
    catalogue_name: str,
    id_field: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        errors.append(f"{catalogue_name} must be a list")
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries, start=1):
        prefix = f"{catalogue_name}[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        entry_id = str(entry.get(id_field) or "").strip()
        if not entry_id:
            errors.append(f"{prefix} is missing {id_field}")
            continue
        if entry_id in indexed:
            errors.append(f"Duplicate {id_field}: {entry_id}")
            continue
        indexed[entry_id] = entry
    return indexed


def load_catalogues(
    paths: dict[str, str | Path] | None = None,
    *,
    validate: bool = True,
) -> CatalogueSet:
    selected = {**DEFAULT_CATALOGUE_PATHS, **(paths or {})}
    payloads = {name: _load_yaml(path) for name, path in selected.items()}
    errors: list[str] = []
    post_types = _index_entries(
        payloads["post_types"].get("post_types"),
        catalogue_name="post_types",
        id_field="post_type_id",
        errors=errors,
    )
    visual_types = _index_entries(
        payloads["visual_types"].get("visual_types"),
        catalogue_name="visual_types",
        id_field="visual_type_id",
        errors=errors,
    )
    metrics = _index_entries(
        payloads["metrics"].get("metrics"),
        catalogue_name="metrics",
        id_field="metric_id",
        errors=errors,
    )
    if errors:
        raise CatalogueValidationError("\n".join(errors))
    catalogues = CatalogueSet(post_types=post_types, visual_types=visual_types, metrics=metrics)
    if validate:
        report = validate_catalogues(catalogues=catalogues)
        if not report["success"]:
            raise CatalogueValidationError("\n".join(report["errors"]))
    return catalogues


def _missing_fields(entry: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(field for field in required if field not in entry)


def _validate_status(prefix: str, entry: dict[str, Any], errors: list[str]) -> None:
    status = str(entry.get("status") or "").strip()
    if status not in VALID_STATUSES:
        errors.append(f"{prefix}.status must be one of {sorted(VALID_STATUSES)}; got {status!r}")


def _validate_version(prefix: str, entry: dict[str, Any], errors: list[str]) -> None:
    version = entry.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        errors.append(f"{prefix}.version must be a positive integer")


def _validate_repo_reference(prefix: str, value: Any, errors: list[str]) -> Path | None:
    text = str(value or "").strip()
    if not text:
        errors.append(f"{prefix} must reference a repository file")
        return None
    path = _repo_path(text)
    if not path.is_file():
        errors.append(f"{prefix} does not exist: {text}")
        return None
    return path


def _validate_string_list(prefix: str, value: Any, errors: list[str], *, allow_empty: bool = True) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{prefix} must be a list of non-empty strings")
        return
    if not allow_empty and not value:
        errors.append(f"{prefix} must not be empty")


def _validate_post_types(entries: dict[str, dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    for entry_id, entry in entries.items():
        prefix = f"post_types.{entry_id}"
        missing = _missing_fields(entry, POST_REQUIRED_FIELDS)
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue
        _validate_status(prefix, entry, errors)
        _validate_version(prefix, entry, errors)
        _validate_string_list(f"{prefix}.supported_aspect_ratios", entry["supported_aspect_ratios"], errors, allow_empty=False)
        _validate_string_list(f"{prefix}.supported_slot_types", entry["supported_slot_types"], errors, allow_empty=False)
        _validate_string_list(f"{prefix}.required_variables", entry["required_variables"], errors)
        _validate_string_list(f"{prefix}.optional_variables", entry["optional_variables"], errors)
        layout_path = _validate_repo_reference(f"{prefix}.layout_path", entry["layout_path"], errors)
        if not isinstance(entry.get("slots"), dict):
            errors.append(f"{prefix}.slots must be a mapping")
        else:
            for slot_id, slot in entry["slots"].items():
                slot_prefix = f"{prefix}.slots.{slot_id}"
                if not isinstance(slot, dict):
                    errors.append(f"{slot_prefix} must be a mapping")
                    continue
                for dimension in ("width", "height"):
                    value = slot.get(dimension)
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        errors.append(f"{slot_prefix}.{dimension} must be a positive integer")
        if layout_path:
            try:
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"{prefix}.layout_path is not valid JSON: {exc}")
            else:
                if layout.get("template_id") != entry_id:
                    errors.append(
                        f"{prefix}.layout_path template_id {layout.get('template_id')!r} does not match {entry_id!r}"
                    )
                placeholders = {
                    str(element.get("placeholder"))
                    for element in layout.get("elements", [])
                    if isinstance(element, dict) and element.get("placeholder")
                }
                missing_placeholders = sorted(set(entry["required_variables"]) - placeholders)
                if missing_placeholders:
                    errors.append(
                        f"{prefix}.required_variables not present as layout placeholders: {', '.join(missing_placeholders)}"
                    )
        if entry["status"] == "approved" and not str(entry.get("example_preview") or "").strip():
            errors.append(f"{prefix}.example_preview is required for approved entries")
        elif entry["status"] == "draft" and not entry.get("example_preview"):
            warnings.append(f"{prefix} has no example preview yet")


def _validate_visual_types(entries: dict[str, dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    for entry_id, entry in entries.items():
        prefix = f"visual_types.{entry_id}"
        missing = _missing_fields(entry, VISUAL_REQUIRED_FIELDS)
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue
        _validate_status(prefix, entry, errors)
        _validate_version(prefix, entry, errors)
        for field in ("required_bindings", "optional_bindings", "supported_data_shapes", "known_limitations"):
            _validate_string_list(f"{prefix}.{field}", entry[field], errors, allow_empty=field != "supported_data_shapes")
        item_count = entry.get("recommended_item_count")
        if not isinstance(item_count, dict):
            errors.append(f"{prefix}.recommended_item_count must be a mapping")
        else:
            minimum, maximum = item_count.get("min"), item_count.get("max")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                errors.append(f"{prefix}.recommended_item_count.min must be a positive integer")
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                errors.append(f"{prefix}.recommended_item_count.max must be a positive integer")
            if isinstance(minimum, int) and isinstance(maximum, int) and minimum > maximum:
                errors.append(f"{prefix}.recommended_item_count.min cannot exceed max")
        template_path = _validate_repo_reference(f"{prefix}.template_path", entry["template_path"], errors)
        if template_path:
            template = _load_yaml(template_path)
            if template.get("template_id") != entry_id:
                errors.append(
                    f"{prefix}.template_path template_id {template.get('template_id')!r} does not match {entry_id!r}"
                )
            if template.get("renderer") != entry.get("renderer"):
                errors.append(
                    f"{prefix}.renderer {entry.get('renderer')!r} does not match template renderer {template.get('renderer')!r}"
                )
        test_dir = REPO_ROOT / "instagram" / "visuals" / "tests" / entry_id
        if not test_dir.is_dir() or not (test_dir / "cases.yml").is_file():
            errors.append(f"{prefix} has no deterministic test registry at {test_dir.relative_to(REPO_ROOT)}")
        if entry["status"] == "approved" and not str(entry.get("test_contact_sheet") or "").strip():
            errors.append(f"{prefix}.test_contact_sheet is required for approved entries")
        elif entry["status"] == "draft":
            warnings.append(f"{prefix} remains draft pending explicit visual approval")


def _validate_metrics(entries: dict[str, dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    for entry_id, entry in entries.items():
        prefix = f"metrics.{entry_id}"
        missing = _missing_fields(entry, METRIC_REQUIRED_FIELDS)
        if missing:
            errors.append(f"{prefix} missing required fields: {', '.join(missing)}")
            continue
        _validate_status(prefix, entry, errors)
        _validate_version(prefix, entry, errors)
        for field in ("source_logical_keys", "measure_fields", "dimension_fields", "period_fields"):
            _validate_string_list(
                f"{prefix}.{field}",
                entry[field],
                errors,
                allow_empty=field == "period_fields",
            )
        mapping_path = _validate_repo_reference(f"{prefix}.mapping_config", entry["mapping_config"], errors)
        if mapping_path:
            mapping = _load_yaml(mapping_path)
            if not mapping.get("mapping_id"):
                errors.append(f"{prefix}.mapping_config is missing mapping_id")
            transforms = mapping.get("transforms")
            if not isinstance(transforms, list) or not transforms:
                errors.append(f"{prefix}.mapping_config must define at least one transform")
        if not isinstance(entry.get("validation_rules"), dict):
            errors.append(f"{prefix}.validation_rules must be a mapping")
        if entry["status"] == "draft":
            warnings.append(f"{prefix} remains draft pending project-level factual validation")


def validate_catalogues(
    *,
    catalogues: CatalogueSet | None = None,
    paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if catalogues is None:
        try:
            catalogues = load_catalogues(paths=paths, validate=False)
        except CatalogueValidationError as exc:
            return {
                "success": False,
                "errors": str(exc).splitlines(),
                "warnings": [],
                "counts": {"post_types": 0, "visual_types": 0, "metrics": 0},
            }

    _validate_post_types(catalogues.post_types, errors, warnings)
    _validate_visual_types(catalogues.visual_types, errors, warnings)
    _validate_metrics(catalogues.metrics, errors, warnings)
    return {
        "success": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "post_types": len(catalogues.post_types),
            "visual_types": len(catalogues.visual_types),
            "metrics": len(catalogues.metrics),
        },
    }


def list_options(catalogues: CatalogueSet, names: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    selected = set(names or ("post_types", "visual_types", "metrics"))
    unknown = selected - {"post_types", "visual_types", "metrics"}
    if unknown:
        raise CatalogueValidationError(f"Unknown catalogue names: {', '.join(sorted(unknown))}")
    output: dict[str, list[dict[str, Any]]] = {}
    for name, entries in catalogues.as_dict().items():
        if name not in selected:
            continue
        id_field = {"post_types": "post_type_id", "visual_types": "visual_type_id", "metrics": "metric_id"}[name]
        output[name] = [
            {
                id_field: entry_id,
                "display_name": entry.get("display_name"),
                "description": entry.get("description") or entry.get("intended_use"),
                "status": entry.get("status"),
                "version": entry.get("version"),
            }
            for entry_id, entry in sorted(entries.items())
        ]
    return output
