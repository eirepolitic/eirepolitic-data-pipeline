from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.visuals.renderers.common import load_yaml, resolve_repo_path, rows_from_sample, utc_now, write_json


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _first_available_field(rows: list[dict[str, Any]], candidates: list[str]) -> str:
    available = set()
    for row in rows:
        available.update(str(key) for key in row.keys())
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"None of the candidate fields exist in input rows: {candidates}")


def _clean_label(value: Any, fallback: str = "Unknown") -> str:
    label = str(value or "").strip()
    if not label:
        return fallback
    lowered = label.lower()
    if lowered in {"nan", "none", "null", "unknown", "n/a"}:
        return fallback
    return label


def _apply_row_filters(rows: list[dict[str, Any]], filters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered = list(rows)
    metadata: list[dict[str, Any]] = []
    for filter_cfg in filters:
        operator = str(filter_cfg.get("operator") or "equals")
        field = str(filter_cfg.get("field") or "").strip()
        if not field:
            raise ValueError("Visual mapping row filter requires field")
        before = len(filtered)

        if operator == "equals":
            expected = str(filter_cfg.get("value", ""))
            filtered = [row for row in filtered if str(row.get(field, "")) == expected]
            selected_value: Any = expected
        elif operator == "latest_value":
            values = [str(row.get(field, "")).strip() for row in filtered if str(row.get(field, "")).strip()]
            if not values:
                raise ValueError(f"No non-empty values available for latest_value filter field: {field}")
            numeric_values = [(_as_float(value), value) for value in values]
            if all(parsed is not None for parsed, _ in numeric_values):
                selected_value = max(numeric_values, key=lambda item: float(item[0]))[1]
            else:
                selected_value = max(values)
            filtered = [row for row in filtered if str(row.get(field, "")).strip() == str(selected_value)]
        else:
            raise ValueError(f"Unsupported visual mapping row filter operator: {operator}")

        metadata.append(
            {
                "field": field,
                "operator": operator,
                "selected_value": selected_value,
                "input_rows": before,
                "output_rows": len(filtered),
            }
        )
    return filtered, metadata


def _count_by(rows: list[dict[str, Any]], transform: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_field = _first_available_field(rows, list(transform.get("label_field_candidates", [])))
    output_label_field = str(transform.get("output_label_field", "label"))
    output_value_field = str(transform.get("output_value_field", "value"))
    fallback_label = str(transform.get("fallback_label", "Unknown"))
    max_rows = int(transform.get("max_rows", 0) or 0)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        label = _clean_label(row.get(label_field), fallback=fallback_label)
        counts[label] += 1

    output_rows = [
        {output_label_field: label, output_value_field: count, "source_field": label_field}
        for label, count in counts.items()
    ]
    output_rows = sorted(output_rows, key=lambda row: int(row[output_value_field]), reverse=True)
    if max_rows and len(output_rows) > max_rows:
        kept = output_rows[: max_rows - 1]
        other_total = sum(int(row[output_value_field]) for row in output_rows[max_rows - 1 :])
        output_rows = kept + [{output_label_field: "Other", output_value_field: other_total, "source_field": label_field}]

    return output_rows, {"operation": "count_by", "label_field": label_field, "output_rows": len(output_rows)}


def _sum_by(rows: list[dict[str, Any]], transform: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_field = _first_available_field(rows, list(transform.get("label_field_candidates", [])))
    value_field = _first_available_field(rows, list(transform.get("value_field_candidates", [])))
    output_label_field = str(transform.get("output_label_field", "label"))
    output_value_field = str(transform.get("output_value_field", "value"))
    fallback_label = str(transform.get("fallback_label", "Unknown"))
    max_rows = int(transform.get("max_rows", 0) or 0)

    totals: dict[str, float] = defaultdict(float)
    skipped_invalid_values = 0
    for row in rows:
        value = _as_float(row.get(value_field))
        if value is None:
            skipped_invalid_values += 1
            continue
        label = _clean_label(row.get(label_field), fallback=fallback_label)
        totals[label] += value

    output_rows = [
        {
            output_label_field: label,
            output_value_field: f"{total:g}",
            "source_label_field": label_field,
            "source_value_field": value_field,
        }
        for label, total in totals.items()
    ]
    output_rows = sorted(output_rows, key=lambda row: float(row[output_value_field]), reverse=True)
    if max_rows and len(output_rows) > max_rows:
        kept = output_rows[: max_rows - 1]
        other_total = sum(float(row[output_value_field]) for row in output_rows[max_rows - 1 :])
        output_rows = kept + [
            {
                output_label_field: "Other",
                output_value_field: f"{other_total:g}",
                "source_label_field": label_field,
                "source_value_field": value_field,
            }
        ]

    return output_rows, {
        "operation": "sum_by",
        "label_field": label_field,
        "value_field": value_field,
        "output_rows": len(output_rows),
        "skipped_invalid_values": skipped_invalid_values,
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["label", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_visual_data(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_yaml(config_path)
    mapping_id = str(config.get("mapping_id") or config_path.stem)
    source_sample = {"input": config.get("input", {}) or {}}
    rows, input_metadata = rows_from_sample(source_sample)
    if not rows and bool(config.get("required_rows", True)):
        raise ValueError(f"No rows loaded for visual data mapping: {mapping_id}")

    filtered_rows, filter_metadata = _apply_row_filters(rows, list(config.get("row_filters", []) or []))
    if not filtered_rows and bool(config.get("required_rows", True)):
        raise ValueError(f"No rows remain after visual data mapping filters: {mapping_id}")

    results: list[dict[str, Any]] = []
    for transform in config.get("transforms", []) or []:
        transform_id = str(transform.get("id") or transform.get("operation") or "transform")
        operation = str(transform.get("operation", "count_by"))
        if operation == "count_by":
            output_rows, transform_metadata = _count_by(filtered_rows, transform)
        elif operation == "sum_by":
            output_rows, transform_metadata = _sum_by(filtered_rows, transform)
        else:
            raise ValueError(f"Unsupported visual data transform operation: {operation}")

        output_path = str(transform["output_path"])
        _write_csv(output_path, output_rows)
        results.append(
            {
                "id": transform_id,
                "operation": operation,
                "output_path": output_path,
                "row_count": len(output_rows),
                "metadata": transform_metadata,
            }
        )

    manifest = {
        "success": True,
        "mapping_id": mapping_id,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "created_at": utc_now(),
        "input": input_metadata,
        "source_row_count": len(rows),
        "filtered_row_count": len(filtered_rows),
        "row_filters": filter_metadata,
        "results": results,
        "review_only": True,
    }
    manifest_path = config.get("manifest_path")
    if manifest_path:
        write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chart-ready CSVs for Instagram visual renderers.")
    parser.add_argument("--config", required=True, help="Visual data mapping YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_visual_data(args.config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
