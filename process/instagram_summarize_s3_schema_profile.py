from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_list(values: list[Any], limit: int = 12) -> str:
    if not values:
        return "_none_"
    rendered = [f"`{str(value)}`" for value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"… {len(values) - limit} more")
    return ", ".join(rendered)


def _fmt_examples(values: list[Any], limit: int = 3) -> str:
    if not values:
        return ""
    return "; ".join(str(value).replace("\n", " ")[:80] for value in values[:limit])


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

    example_values = schema.get("example_values", {}) or {}
    non_empty_counts = schema.get("non_empty_counts", {}) or {}
    blank_counts = schema.get("blank_counts", {}) or {}
    top_values = schema.get("top_values", {}) or {}

    lines += [
        "### Sample column coverage",
        "",
        "| Column | Non-empty | Blank | Examples | Top sampled values |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for column in columns[:60]:
        examples = _fmt_examples(example_values.get(column, []) or [])
        top = top_values.get(column, []) or []
        top_text = "; ".join(f"{entry.get('value', '')} ({entry.get('count', 0)})" for entry in top[:5])
        lines.append(
            f"| `{column}` | {non_empty_counts.get(column, 0)} | {blank_counts.get(column, 0)} | {examples} | {top_text} |"
        )
    if len(columns) > 60:
        lines.append(f"| _{len(columns) - 60} more columns omitted from Markdown table_ |  |  |  |  |")
    lines.append("")
    return lines


def build_markdown(profile_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    profile_path = Path(profile_path)
    output_path = Path(output_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
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
        "- Download strategy: S3 prefix range read only; full datasets are not downloaded.",
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown summary from an S3 schema profile JSON artifact.")
    parser.add_argument("--profile", default="generated_visual_data/s3_schema_profile.json")
    parser.add_argument("--output", default="generated_visual_data/s3_schema_profile.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_markdown(args.profile, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
