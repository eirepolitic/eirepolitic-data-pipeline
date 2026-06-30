from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.renderer.constants import DEFAULT_BUCKET, DEFAULT_REGION
from instagram.visuals.renderers.common import load_yaml, utc_now, write_json

DEFAULT_RANGE_BYTES = 262_144
DEFAULT_SAMPLE_ROWS = 25


def _s3_client(region: str):
    return boto3.client("s3", region_name=region)


def _candidate_keys(input_cfg: dict[str, Any]) -> list[str]:
    if input_cfg.get("key"):
        return [str(input_cfg["key"])]
    return [str(key) for key in input_cfg.get("keys", [])]


def _read_s3_prefix(bucket: str, region: str, key: str, range_bytes: int) -> tuple[str, dict[str, Any]]:
    response = _s3_client(region).get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes=0-{max(range_bytes - 1, 0)}",
    )
    body = response["Body"].read().decode("utf-8-sig", errors="replace")
    metadata = {
        "bucket": bucket,
        "region": region,
        "key": key,
        "range_bytes_requested": range_bytes,
        "content_length": response.get("ContentLength"),
        "content_range": response.get("ContentRange"),
        "etag": response.get("ETag"),
        "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
    }
    return body, metadata


def _first_available_prefix(input_cfg: dict[str, Any], range_bytes: int) -> tuple[str, dict[str, Any]]:
    bucket = str(input_cfg.get("bucket") or os.environ.get("INSTAGRAM_VISUAL_S3_BUCKET") or DEFAULT_BUCKET)
    region = str(input_cfg.get("region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION)
    checked: list[dict[str, Any]] = []
    last_error: str | None = None

    for key in _candidate_keys(input_cfg):
        try:
            text, metadata = _read_s3_prefix(bucket, region, key, range_bytes)
            metadata["checked"] = checked + [{"key": key, "available": True}]
            return text, metadata
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "Unknown")
            checked.append({"key": key, "available": False, "error_code": code})
            last_error = code
            if code not in {"NoSuchKey", "404", "NotFound"}:
                raise

    raise FileNotFoundError(f"No available S3 keys found. checked={checked}; last_error={last_error}")


def _normalise_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def _profile_rows(text: str, max_rows: int, include_sampled_values: bool = False) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    columns = list(reader.fieldnames or [])
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if index >= max_rows:
            break
        rows.append(_normalise_row(row))

    non_empty_counts = {column: 0 for column in columns}
    blank_counts = {column: 0 for column in columns}
    examples: dict[str, list[str]] = {column: [] for column in columns}
    value_counters: dict[str, Counter[str]] = {column: Counter() for column in columns}

    for row in rows:
        for column in columns:
            value = str(row.get(column, "")).strip()
            if value:
                non_empty_counts[column] += 1
                if include_sampled_values:
                    value_counters[column][value] += 1
                    if len(examples[column]) < 3 and value not in examples[column]:
                        examples[column].append(value)
            else:
                blank_counts[column] += 1

    likely_numeric: list[str] = []
    for column in columns:
        values = [str(row.get(column, "")).strip().replace(",", "") for row in rows if str(row.get(column, "")).strip()]
        if values and all(_is_float(value) for value in values):
            likely_numeric.append(column)

    profile = {
        "columns": columns,
        "column_count": len(columns),
        "sample_row_count": len(rows),
        "range_may_be_truncated": not text.endswith("\n"),
        "non_empty_counts": non_empty_counts,
        "blank_counts": blank_counts,
        "sampled_values_included": include_sampled_values,
        "likely_numeric_columns": likely_numeric,
    }
    if include_sampled_values:
        profile["example_values"] = examples
        profile["top_values"] = {
            column: [{"value": value, "count": count} for value, count in counter.most_common(5)]
            for column, counter in value_counters.items()
        }
    return profile


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _mapping_hints(config: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    column_set = set(columns)
    hints: list[dict[str, Any]] = []
    for transform in config.get("transforms", []):
        label_candidates = [str(value) for value in transform.get("label_field_candidates", [])]
        value_candidates = [str(value) for value in transform.get("value_field_candidates", [])]
        hints.append(
            {
                "transform_id": transform.get("id"),
                "operation": transform.get("operation"),
                "label_candidates": label_candidates,
                "label_matches": [candidate for candidate in label_candidates if candidate in column_set],
                "value_candidates": value_candidates,
                "value_matches": [candidate for candidate in value_candidates if candidate in column_set],
            }
        )
    return {"transforms": hints}


def profile_config(config_path: str | Path, range_bytes: int, sample_rows: int, include_sampled_values: bool = False) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = load_yaml(path)
    input_cfg = dict(config.get("input", {}))
    text, s3_metadata = _first_available_prefix(input_cfg, range_bytes)
    row_profile = _profile_rows(text, sample_rows, include_sampled_values=include_sampled_values)
    return {
        "mapping_id": config.get("mapping_id"),
        "config_path": str(path.relative_to(REPO_ROOT)),
        "profiled_at": utc_now(),
        "review_only": True,
        "download_strategy": "s3_get_object_range_prefix_only",
        "s3": s3_metadata,
        "schema": row_profile,
        "mapping_hints": _mapping_hints(config, row_profile["columns"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile S3 CSV schemas for Instagram visual mappings without downloading full datasets.")
    parser.add_argument("--config", action="append", required=True, help="Mapping YAML config to profile. Can be passed more than once.")
    parser.add_argument("--output", default="generated_visual_data/s3_schema_profile.json")
    parser.add_argument("--range-bytes", type=int, default=DEFAULT_RANGE_BYTES)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument(
        "--include-sampled-values",
        action="store_true",
        default=False,
        help="Include example/top sampled values in the JSON profile. Off by default to avoid exposing raw row values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [
        profile_config(
            config,
            args.range_bytes,
            args.sample_rows,
            include_sampled_values=args.include_sampled_values,
        )
        for config in args.config
    ]
    payload = {
        "success": True,
        "created_at": utc_now(),
        "review_only": True,
        "publishes_content": False,
        "range_bytes": args.range_bytes,
        "sample_rows": args.sample_rows,
        "sampled_values_included": args.include_sampled_values,
        "profile_count": len(profiles),
        "profiles": profiles,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
