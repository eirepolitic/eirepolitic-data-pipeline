from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import boto3
import pandas as pd
import requests
import yaml
from openai import OpenAI


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
DEFAULT_REGION = os.getenv("AWS_REGION", "ca-central-1")
DEFAULT_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
DEFAULT_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")
DEFAULT_VALIDATION_MODEL = os.getenv("OPENAI_VALIDATION_MODEL", "gpt-4.1-mini")
METRICS_KEY = os.getenv("MEMBER_PROFILE_METRICS_INPUT_KEY", "processed/members/member_profile_metrics_2025.csv")
CONTENT_TYPE_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
SUFFIX_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


VALIDATION_SCHEMA = {
    "name": "member_profile_template_validation",
    "type": "json_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "template_fidelity_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "text_legibility_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "formatting_issues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "suspect_text": {
                "type": "array",
                "items": {"type": "string"},
            },
            "needs_second_pass": {"type": "boolean"},
            "correction_instructions": {"type": "string"},
        },
        "required": [
            "template_fidelity_score",
            "text_legibility_score",
            "formatting_issues",
            "suspect_text",
            "needs_second_pass",
            "correction_instructions",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", default="generated_visual_tests/option5_member_profile_ai")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--validation-model", default=DEFAULT_VALIDATION_MODEL)
    return parser.parse_args()


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def slugify(value: str) -> str:
    return "-".join(str(value or "").strip().lower().replace("/", " ").replace("_", " ").split())


def is_url(value: str) -> bool:
    parsed = urlparse(str(value or ""))
    return parsed.scheme in {"http", "https"}


def infer_suffix_from_url_or_content_type(source: str, content_type: Optional[str] = None) -> str:
    if content_type:
        normalized = content_type.split(";")[0].strip().lower()
        if normalized in CONTENT_TYPE_TO_SUFFIX:
            return CONTENT_TYPE_TO_SUFFIX[normalized]

    parsed = urlparse(str(source or ""))
    url_suffix = Path(parsed.path).suffix.lower()
    if url_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if url_suffix == ".jpeg" else url_suffix

    guessed, _ = mimetypes.guess_type(parsed.path)
    if guessed in CONTENT_TYPE_TO_SUFFIX:
        return CONTENT_TYPE_TO_SUFFIX[guessed]

    return ".png"


def ensure_destination_suffix(destination: Path, suffix: str) -> Path:
    if destination.suffix.lower() == suffix.lower():
        return destination
    if destination.suffix:
        return destination.with_suffix(suffix)
    return destination.parent / f"{destination.name}{suffix}"


def download_to_path(source: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_url(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        suffix = infer_suffix_from_url_or_content_type(source, response.headers.get("Content-Type"))
        final_destination = ensure_destination_suffix(destination, suffix)
        final_destination.write_bytes(response.content)
        return final_destination

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing local file: {source}")
    suffix = source_path.suffix.lower() or ".png"
    final_destination = ensure_destination_suffix(destination, suffix)
    shutil.copy2(source_path, final_destination)
    return final_destination


def image_file_tuple(path: Path) -> tuple[str, bytes, str]:
    suffix = path.suffix.lower()
    mime = SUFFIX_TO_MIME.get(suffix, "image/png")
    return (path.name, path.read_bytes(), mime)


def data_url_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = SUFFIX_TO_MIME.get(suffix, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_member(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series:
    cfg = spec.get("selection", {})
    exclude_names = {str(name).strip().lower() for name in cfg.get("exclude_names", [])}
    candidates = df.copy()
    candidates = candidates[candidates["photo_url"].fillna("").astype(str).str.strip() != ""].copy()
    if exclude_names:
        candidates = candidates[~candidates["full_name"].fillna("").str.lower().isin(exclude_names)].copy()

    if candidates.empty:
        raise RuntimeError("No member candidates with photo_url found after exclusions.")

    order_by = cfg.get("order_by", ["speech_count_2025", "full_name"])
    ascending = cfg.get("ascending", [False, True])
    candidates = candidates.sort_values(by=order_by, ascending=ascending)
    return candidates.iloc[0]


def exact_visible_values(member: pd.Series) -> Dict[str, str]:
    return {
        "full_name": str(member.get("full_name") or ""),
        "constituency": str(member.get("constituency") or ""),
        "party": str(member.get("party") or ""),
        "top_issue": str(member.get("top_issue_2025") or ""),
        "vote_participation_pct": f"{int(member.get('vote_participation_pct_2025') or 0)}%",
        "speech_rank": str(int(member.get("speech_rank_2025") or 0)),
    }


def build_prompt_v1(member: pd.Series, spec: Dict[str, Any]) -> str:
    voice = spec.get("prompt", {}).get("voice", {})
    exact = exact_visible_values(member)

    lines = [
        "Use the first image as the master template. Preserve its overall layout, border, decorative corner ornaments, color palette, spacing, typography style, framing, and composition as closely as possible.",
        "Use the second image only as the replacement portrait for the framed photo area.",
        "Do not redesign the slide.",
        "Replace the old portrait and old text with the following exact visible values:",
        f"- Full name: {exact['full_name']}",
        f"- Constituency: {exact['constituency']}",
        f"- Party: {exact['party']}",
        f"- Top Issue: {exact['top_issue']}",
        f"- Vote Participation %: {exact['vote_participation_pct']}",
        f"- Speech Rank: {exact['speech_rank']}",
        "Keep the slide in portrait format and retain the same approximate text placements and hierarchy.",
        "Do not add extra badges, logos, labels, charts, or new decorative concepts.",
        "Do not change the border ornament style.",
        "Do not add made-up values.",
    ]

    if voice:
        lines.append(
            "Visual tone: "
            f"clean={voice.get('clean', True)}, restrained={voice.get('restrained', True)}, premium={voice.get('premium', True)}."
        )

    return "\n".join(lines)


def build_validation_prompt(source_values: Dict[str, Any]) -> str:
    member = source_values["selected_member"]
    exact = {
        "full_name": member["full_name"],
        "constituency": member["constituency"],
        "party": member["party"],
        "top_issue": member["top_issue_2025"],
        "vote_participation_pct": f"{member['vote_participation_pct_2025']}%",
        "speech_rank": str(member["speech_rank_2025"]),
    }
    return "\n".join(
        [
            "You are validating an experimental template-based infographic edit.",
            "Image A is the original template.",
            "Image B is the first-pass edited output.",
            "Check whether Image B correctly preserves the template layout and styling while replacing the content with the provided source truth.",
            "Return JSON only matching the requested schema.",
            "Source truth:",
            f"- Full name: {exact['full_name']}",
            f"- Constituency: {exact['constituency']}",
            f"- Party: {exact['party']}",
            f"- Top Issue: {exact['top_issue']}",
            f"- Vote Participation %: {exact['vote_participation_pct']}",
            f"- Speech Rank: {exact['speech_rank']}",
            "Focus on layout drift from template, alignment issues, spacing issues, multiline wrapping, and text that looks wrong, truncated, malformed, or suspicious.",
        ]
    )


def build_prompt_v2(member: pd.Series, validation_report: Dict[str, Any]) -> str:
    exact = exact_visible_values(member)
    correction_instructions = validation_report.get("correction_instructions") or "Move the output closer to the template, improve alignment, spacing, and multiline formatting, and preserve exact values."
    formatting_issues = validation_report.get("formatting_issues") or []
    suspect_text = validation_report.get("suspect_text") or []

    lines = [
        "Use the first image as the master template.",
        "Use the second image only as the replacement portrait for the framed photo area.",
        "Use the third image as the first-pass draft that needs correction.",
        "Create a corrected second-pass version of the member profile slide.",
        "Preserve the original template layout, border ornaments, spacing, hierarchy, and styling as closely as possible.",
        "Do not redesign the slide.",
        "Use these exact visible values:",
        f"- Full name: {exact['full_name']}",
        f"- Constituency: {exact['constituency']}",
        f"- Party: {exact['party']}",
        f"- Top Issue: {exact['top_issue']}",
        f"- Vote Participation %: {exact['vote_participation_pct']}",
        f"- Speech Rank: {exact['speech_rank']}",
        f"Correction priorities: {correction_instructions}",
    ]

    if formatting_issues:
        lines.append("Formatting issues to fix:")
        lines.extend(f"- {issue}" for issue in formatting_issues)

    if suspect_text:
        lines.append("Suspect text to correct if visible:")
        lines.extend(f"- {item}" for item in suspect_text)

    lines.extend(
        [
            "Important:",
            "- Keep the portrait frame structure.",
            "- Keep the bottom metrics layout.",
            "- Improve alignment and spacing.",
            "- Fix multiline wrapping if needed.",
            "- Do not invent or alter values.",
        ]
    )

    return "\n".join(lines)


def parse_response_json_text(response: Any) -> Dict[str, Any]:
    text = getattr(response, "output_text", None)
    if text:
        return json.loads(text)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        # Best effort fallback
        for item in dumped.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        return json.loads(content["text"])
    raise RuntimeError("Could not parse JSON validation output from response.")


def validate_v1(
    client: OpenAI,
    *,
    validation_model: str,
    template_path: Path,
    v1_path: Path,
    source_values: Dict[str, Any],
) -> tuple[Dict[str, Any], Any]:
    prompt = build_validation_prompt(source_values)
    response = client.responses.create(
        model=validation_model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url_for_image(template_path), "detail": "high"},
                    {"type": "input_image", "image_url": data_url_for_image(v1_path), "detail": "high"},
                ],
            }
        ],
        text={"format": VALIDATION_SCHEMA},
    )
    parsed = parse_response_json_text(response)
    return parsed, response


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    df = read_csv_from_s3(DEFAULT_BUCKET, METRICS_KEY)
    member = select_member(df, spec)

    run_slug = f"{slugify(str(member['full_name']))}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = Path(args.output_root) / run_slug
    inputs_dir = run_root / "inputs"
    output_dir = run_root / "outputs"
    metadata_dir = run_root / "metadata"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    template_path = download_to_path(spec["template_image_source"], inputs_dir / "template_image.png")
    member_photo_path = download_to_path(str(member["photo_url"]), inputs_dir / "member_photo.png")

    source_values = {
        "selected_member": {
            "member_code": str(member.get("member_code") or ""),
            "full_name": str(member.get("full_name") or ""),
            "constituency": str(member.get("constituency") or ""),
            "party": str(member.get("party") or ""),
            "photo_url": str(member.get("photo_url") or ""),
            "top_issue_2025": str(member.get("top_issue_2025") or ""),
            "top_issue_count_2025": int(member.get("top_issue_count_2025") or 0),
            "vote_participation_pct_2025": int(member.get("vote_participation_pct_2025") or 0),
            "distinct_votes_participated_2025": int(member.get("distinct_votes_participated_2025") or 0),
            "all_distinct_vote_ids_2025": int(member.get("all_distinct_vote_ids_2025") or 0),
            "speech_count_2025": int(member.get("speech_count_2025") or 0),
            "speech_rank_2025": int(member.get("speech_rank_2025") or 0),
        },
        "risk_notes": [
            "This is an experimental image-edit test, not a trusted source of text accuracy.",
            "Visible values must be checked against source_values.json during review.",
            "Text rendering remains high risk even when the layout resembles the template.",
            "A second-pass correction is always run for comparison in this test variant.",
        ],
    }
    write_json(metadata_dir / "source_values.json", source_values)

    client = OpenAI()

    prompt_v1 = build_prompt_v1(member, spec)
    (metadata_dir / "prompt_v1.txt").write_text(prompt_v1, encoding="utf-8")
    v1_result = client.images.edit(
        model=args.model,
        image=[image_file_tuple(template_path), image_file_tuple(member_photo_path)],
        prompt=prompt_v1,
        size=args.size,
    )

    v1_b64 = v1_result.data[0].b64_json
    if not v1_b64:
        raise RuntimeError("No image payload returned by first-pass image edit request.")
    v1_path = output_dir / "member_profile_ai_edit_v1.png"
    v1_path.write_bytes(base64.b64decode(v1_b64))
    write_json(
        metadata_dir / "openai_response_v1.json",
        v1_result.model_dump() if hasattr(v1_result, "model_dump") else {"raw_result": str(v1_result)},
    )

    validation_report, validation_response = validate_v1(
        client,
        validation_model=args.validation_model,
        template_path=template_path,
        v1_path=v1_path,
        source_values=source_values,
    )
    validation_report["second_pass_policy"] = "always_run"
    validation_report["validation_model"] = args.validation_model
    write_json(metadata_dir / "validation_report.json", validation_report)
    write_json(
        metadata_dir / "validation_response.json",
        validation_response.model_dump() if hasattr(validation_response, "model_dump") else {"raw_result": str(validation_response)},
    )

    prompt_v2 = build_prompt_v2(member, validation_report)
    (metadata_dir / "prompt_v2.txt").write_text(prompt_v2, encoding="utf-8")
    v2_result = client.images.edit(
        model=args.model,
        image=[image_file_tuple(template_path), image_file_tuple(member_photo_path), image_file_tuple(v1_path)],
        prompt=prompt_v2,
        size=args.size,
    )

    v2_b64 = v2_result.data[0].b64_json
    if not v2_b64:
        raise RuntimeError("No image payload returned by second-pass image edit request.")
    v2_path = output_dir / "member_profile_ai_edit_v2.png"
    v2_path.write_bytes(base64.b64decode(v2_b64))
    # Convenience alias for latest output
    final_path = output_dir / "member_profile_ai_edit.png"
    final_path.write_bytes(v2_path.read_bytes())
    write_json(
        metadata_dir / "openai_response_v2.json",
        v2_result.model_dump() if hasattr(v2_result, "model_dump") else {"raw_result": str(v2_result)},
    )

    print(json.dumps(
        {
            "run_root": str(run_root.resolve()),
            "output_image_v1": str(v1_path.resolve()),
            "output_image_v2": str(v2_path.resolve()),
            "output_image_latest": str(final_path.resolve()),
            "selected_member": str(member.get("full_name") or ""),
            "validation_model": args.validation_model,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
