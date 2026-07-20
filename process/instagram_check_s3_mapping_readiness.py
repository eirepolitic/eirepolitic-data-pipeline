from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.visuals.renderers.common import utc_now, write_json

COUNT_ONLY_OPERATIONS = {"count_by"}
VALUE_REQUIRED_OPERATIONS = {"sum_by"}


def _required_fields(operation: str) -> dict[str, bool]:
    if operation in COUNT_ONLY_OPERATIONS:
        return {"label": True, "value": False}
    if operation in VALUE_REQUIRED_OPERATIONS:
        return {"label": True, "value": True}
    return {"label": True, "value": True}


def _check_transform(profile: dict[str, Any], transform: dict[str, Any]) -> dict[str, Any]:
    operation = str(transform.get("operation") or "")
    requirements = _required_fields(operation)
    label_matches = [str(value) for value in transform.get("label_matches", [])]
    value_matches = [str(value) for value in transform.get("value_matches", [])]
    schema = profile.get("schema", {}) or {}
    sample_row_count = int(schema.get("sample_row_count") or 0)
    column_count = int(schema.get("column_count") or 0)

    errors: list[str] = []
    warnings: list[str] = []

    if column_count <= 0:
        errors.append("No CSV columns were detected in the sampled S3 prefix.")
    if sample_row_count <= 0:
        errors.append("No sample rows were available in the sampled S3 prefix.")
    if requirements["label"] and not label_matches:
        errors.append("No configured label field candidate matched the S3 schema.")
    if requirements["value"] and not value_matches:
        errors.append("No configured value field candidate matched the S3 schema.")
    if operation not in COUNT_ONLY_OPERATIONS | VALUE_REQUIRED_OPERATIONS:
        warnings.append(f"Unknown operation `{operation}`; label and value candidates were both treated as required.")

    return {
        "transform_id": transform.get("transform_id"),
        "operation": operation,
        "required": requirements,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "label_candidates": transform.get("label_candidates", []),
        "label_matches": label_matches,
        "value_candidates": transform.get("value_candidates", []),
        "value_matches": value_matches,
    }


def _check_profile(profile: dict[str, Any]) -> dict[str, Any]:
    transforms = (profile.get("mapping_hints", {}) or {}).get("transforms", []) or []
    checks = [_check_transform(profile, transform) for transform in transforms]
    errors: list[str] = []
    warnings: list[str] = []

    if not transforms:
        errors.append("No transforms were found in mapping hints.")

    for check in checks:
        for error in check["errors"]:
            errors.append(f"{check.get('transform_id')}: {error}")
        for warning in check["warnings"]:
            warnings.append(f"{check.get('transform_id')}: {warning}")

    return {
        "mapping_id": profile.get("mapping_id"),
        "config_path": profile.get("config_path"),
        "s3_key": (profile.get("s3", {}) or {}).get("key"),
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "transform_checks": checks,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# S3 mapping readiness",
        "",
        "Review-only readiness check for S3-backed Instagram visual mappings.",
        "",
        f"- Created at: `{payload.get('created_at', '')}`",
        f"- Overall ready: `{payload.get('ready')}`",
        f"- Profile count: `{payload.get('profile_count', 0)}`",
        "- Failure scope: readiness failures mark only the non-blocking S3 smoke status as failed.",
        "- Publishing: this does not publish, schedule, or approve Instagram content.",
        "",
    ]

    for profile in payload.get("profiles", []) or []:
        lines += [
            f"## {profile.get('mapping_id')}",
            "",
            f"- Config: `{profile.get('config_path', '')}`",
            f"- S3 key: `{profile.get('s3_key', '')}`",
            f"- Ready: `{profile.get('ready')}`",
            "",
        ]
        if profile.get("errors"):
            lines += ["### Errors", ""]
            lines.extend(f"- {error}" for error in profile["errors"])
            lines.append("")
        if profile.get("warnings"):
            lines += ["### Warnings", ""]
            lines.extend(f"- {warning}" for warning in profile["warnings"])
            lines.append("")

        lines += [
            "### Transform checks",
            "",
            "| Transform | Operation | Ready | Label matches | Value matches |",
            "| --- | --- | --- | --- | --- |",
        ]
        for check in profile.get("transform_checks", []) or []:
            label_matches = ", ".join(f"`{value}`" for value in check.get("label_matches", [])) or "_none_"
            value_matches = ", ".join(f"`{value}`" for value in check.get("value_matches", [])) or "_none_"
            lines.append(
                f"| `{check.get('transform_id', '')}` | `{check.get('operation', '')}` | `{check.get('ready')}` | {label_matches} | {value_matches} |"
            )
        lines.append("")

    return "\n".join(lines)


def check_readiness(profile_path: str | Path, output_json: str | Path, output_markdown: str | Path) -> dict[str, Any]:
    profile_path = Path(profile_path)
    source = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = [_check_profile(profile) for profile in source.get("profiles", [])]
    ready = all(profile["ready"] for profile in profiles) and bool(profiles)
    payload = {
        "success": ready,
        "ready": ready,
        "created_at": utc_now(),
        "review_only": True,
        "publishes_content": False,
        "profile_path": str(profile_path),
        "profile_count": len(profiles),
        "profiles": profiles,
    }
    write_json(output_json, payload)
    output_markdown = Path(output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(build_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check readiness of S3 visual mapping fields from a schema profile artifact.")
    parser.add_argument("--profile", default="generated_visual_data/s3_schema_profile.json")
    parser.add_argument("--output-json", default="generated_visual_data/s3_mapping_readiness.json")
    parser.add_argument("--output-markdown", default="generated_visual_data/s3_mapping_readiness.md")
    parser.add_argument("--fail-on-not-ready", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_readiness(args.profile, args.output_json, args.output_markdown)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.fail_on_not_ready and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
