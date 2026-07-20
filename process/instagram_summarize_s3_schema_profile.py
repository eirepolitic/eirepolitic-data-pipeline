from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RAW_SAMPLE_FIELDS = {"example_values", "top_values"}


def _fmt_list(values: list[Any], limit: int = 12) -> str:
    if not values:
        return "_none_"
    rendered = [f"`{str(value)}`" for value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"… {len(values) - limit} more")
    return ", ".join(rendered)


def _raw_sample_field_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("sampled_values_included"):
        errors.append("Top-level sampled_values_included is true.")
    for profile_index, profile in enumerate(payload.get("profiles", []) or []):
        schema = profile.get("schema", {}) or {}
        mapping_id = profile.get("mapping_id") or f"profile_{profile_index}"
        if schema.get("sampled_values_included"):
            errors.append(f"{mapping_id}: schema sampled_values_included is true.")
        for field in sorted(RAW_SAMPLE_FIELDS):
            if field in schema:
                errors.append(f"{mapping_id}: schema contains raw sampled field `{field}`.")
    return errors


def _profile_section(profile: dict[str, Any]) -> list[str]:
    mapping_id = str(profile.get("mapping_id") or "unknown_mapping")
    config_path = str(profile.get("config_path") or "")
    schema = profile.get("schema", {}) or {}
    s3 = profile.get("s3", {}) or {}
    columns = schema.get("columns", []) or []
    mapping_hints = (profile.get("mapping_hints", {}) or {}).get("transforms", []) or []

    lines = [
        f"## {mapping_id}",
        "",
        f"- Config: `{config_path}`",
        f"- S3 key: `{s3.get('key', '')}`",
        f"- Bucket: `{s3.get('bucket', '')}`",
        f"- Region: `{s3.get('region', '')}`",
        f"- Last modified: `{s3.get('last_modified', '')}`",
        f"- Content range: `{s3.get('content_range', '')}`",
        f"- Column count: `{schema.get('column_count', 0)}`",
        f"- Sample rows inspected: `{schema.get('sample_row_count', 0)}`",
        f"- Sampled values included: `{schema.get('sampled_values_included', False)}`",
        f"- Range may be truncated: `{schema.get('range_may_be_truncated')}`",
        "",
        "### Columns",
        "",
        _fmt_list([str(column) for column in columns], limit=40),
        "",
    ]

    likely_numeric = schema.get("likely_numeric_columns", []) or []
    lines += [
        "### Likely numeric columns",
        "",
        _fmt_list([str(column) for column in likely_numeric], limit=40),
        "",
    ]

    if mapping_hints:
        lines += ["### Mapping candidate matches", ""]
        for hint in mapping_hints:
            lines += [
                f"- Transform `{hint.get('transform_id', '')}` / `{hint.get('operation', '')}`",
                f"  - Label matches: {_fmt_list(hint.get('label_matches', []) or [])}",
                f"  - Value matches: {_fmt_list(hint.get('value_matches', []) or [])}",
            ]
        lines.append("")

    non_empty_counts = schema.get("non_empty_counts", {}) or {}
    blank_counts = schema.get("blank_counts", {}) or {}

    lines += [
        "### Sample column coverage",
        "",
        "Raw sampled field values are omitted from this public preview summary by default.",
        "",
        "| Column | Non-empty sampled rows | Blank sampled rows |",
        "| --- | ---: | ---: |",
    ]
    for column in columns[:60]:
        lines.append(f"| `{column}` | {non_empty_counts.get(column, 0)} | {blank_counts.get(column, 0)} |")
    if len(columns) > 60:
        lines.append(f"| _{len(columns) - 60} more columns omitted from Markdown table_ |  |  |")
    lines.append("")
    return lines


def build_markdown(profile_path: str | Path, output_path: str | Path, allow_sampled_values: bool = False) -> dict[str, Any]:
    profile_path = Path(profile_path)
    output_path = Path(output_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))

    raw_sample_errors = _raw_sample_field_errors(payload)
    if raw_sample_errors and not allow_sampled_values:
        raise ValueError("Public S3 schema summary refused sampled raw values: " + "; ".join(raw_sample_errors))

    profiles = payload.get("profiles", []) or []

    lines = [
        "# S3 schema profile summary",
        "",
        "Review-only schema snapshot for Instagram visual smoke mappings.",
        "",
        f"- Created at: `{payload.get('created_at', '')}`",
        f"- Profile count: `{payload.get('profile_count', len(profiles))}`",
        f"- Range bytes per file: `{payload.get('range_bytes', '')}`",
        f"- Sample rows per file: `{payload.get('sample_rows', '')}`",
        f"- Sampled values included: `{payload.get('sampled_values_included', False)}`",
        "- Download strategy: S3 prefix range read only; full datasets are not downloaded.",
        "- Privacy: raw sampled field values are omitted by default.",
        "- Publishing: this does not publish, schedule, or approve Instagram content.",
        "",
    ]

    for profile in profiles:
        lines.extend(_profile_section(profile))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "success": True,
        "profile_path": str(profile_path),
        "output_path": str(output_path),
        "profile_count": len(profiles),
        "raw_sample_guard_checked": True,
        "raw_sample_guard_errors": raw_sample_errors,
        "allow_sampled_values": allow_sampled_values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown summary from an S3 schema profile JSON artifact.")
    parser.add_argument("--profile", default="generated_visual_data/s3_schema_profile.json")
    parser.add_argument("--output", default="generated_visual_data/s3_schema_profile.md")
    parser.add_argument(
        "--allow-sampled-values",
        action="store_true",
        default=False,
        help="Allow profile JSON containing raw sampled fields. Off by default for public preview summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_markdown(args.profile, args.output, allow_sampled_values=args.allow_sampled_values)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
