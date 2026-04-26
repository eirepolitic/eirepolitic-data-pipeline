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
METRICS_KEY = os.getenv("MEMBER_PROFILE_METRICS_INPUT_KEY", "processed/members/member_profile_metrics_2025.csv")
CONTENT_TYPE_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", default="generated_visual_tests/option5_member_profile_ai")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
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


def build_prompt(member: pd.Series, spec: Dict[str, Any]) -> str:
    voice = spec.get("prompt", {}).get("voice", {})
    exact = {
        "full_name": str(member.get("full_name") or ""),
        "constituency": str(member.get("constituency") or ""),
        "party": str(member.get("party") or ""),
        "top_issue": str(member.get("top_issue_2025") or ""),
        "vote_participation_pct": f"{int(member.get('vote_participation_pct_2025') or 0)}%",
        "speech_rank": str(int(member.get("speech_rank_2025") or 0)),
    }

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
            f"clean={voice.get('clean', True)}, restrained={voice.get('restrained', True)}, "
            f"premium={voice.get('premium', True)}."
        )

    return "\n".join(lines)


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

    prompt = build_prompt(member, spec)
    (metadata_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

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
        ],
    }
    (metadata_dir / "source_values.json").write_text(json.dumps(source_values, indent=2, ensure_ascii=False), encoding="utf-8")

    client = OpenAI()
    with template_path.open("rb") as template_fh, member_photo_path.open("rb") as member_photo_fh:
        result = client.images.edit(
            model=args.model,
            image=[template_fh, member_photo_fh],
            prompt=prompt,
            size=args.size,
        )

    b64_json = result.data[0].b64_json
    if not b64_json:
        raise RuntimeError("No image payload returned by image edit request.")
    image_bytes = base64.b64decode(b64_json)
    final_path = output_dir / "member_profile_ai_edit.png"
    final_path.write_bytes(image_bytes)

    response_dump = result.model_dump() if hasattr(result, "model_dump") else {"raw_result": str(result)}
    (metadata_dir / "openai_response.json").write_text(json.dumps(response_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {
            "run_root": str(run_root.resolve()),
            "output_image": str(final_path.resolve()),
            "selected_member": str(member.get("full_name") or ""),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
