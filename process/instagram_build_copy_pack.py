from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REVIEW_TABLE = "generated_posts/member_profile_batch_v1_fixture/review/review_table.csv"
DEFAULT_COPY_DIR = "generated_posts/member_profile_batch_v1_fixture/copy"
DEFAULT_HASHTAGS = [
    "#EirePolitic",
    "#IrishPolitics",
    "#DailEireann",
    "#Oireachtas",
    "#DataPolitics",
]


def slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def clean(value: Any, default: str = "N/A") -> str:
    if pd.isna(value):
        return default
    text = str(value or "").strip()
    return text if text else default


def normalize_hashtags(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_HASHTAGS
    tags = []
    for item in raw.split(","):
        tag = item.strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        tags.append(tag)
    return tags or DEFAULT_HASHTAGS


def build_caption(row: pd.Series, hashtags: list[str]) -> str:
    full_name = clean(row.get("full_name"))
    party = clean(row.get("party"))
    constituency = clean(row.get("constituency"))
    top_issue = clean(row.get("top_issue_2025"), "No classified issue yet")
    vote = clean(row.get("vote_participation_pct_2025"))
    speech_count = clean(row.get("speech_count_2025"), "0")
    speech_rank = clean(row.get("speech_rank_2025"))

    return "\n".join([
        f"TD profile: {full_name} ({party}, {constituency}).",
        "",
        f"Top 2025 debate issue in this dataset: {top_issue}.",
        f"Vote participation: {vote}.",
        f"Speech activity: {speech_count} issue-labelled speeches; rank {speech_rank}.",
        "",
        "Source: Oireachtas data pipeline. Review before publishing.",
        "",
        " ".join(hashtags),
    ])


def build_alt_text(row: pd.Series) -> str:
    full_name = clean(row.get("full_name"))
    party = clean(row.get("party"))
    constituency = clean(row.get("constituency"))
    top_issue = clean(row.get("top_issue_2025"), "No classified issue yet")
    vote = clean(row.get("vote_participation_pct_2025"))
    speech_count = clean(row.get("speech_count_2025"), "0")
    speech_rank = clean(row.get("speech_rank_2025"))
    return (
        f"Profile card for {full_name}, {party} TD for {constituency}. "
        f"The card lists top 2025 debate issue as {top_issue}, vote participation as {vote}, "
        f"and speech activity as {speech_count} issue-labelled speeches with rank {speech_rank}."
    )


def build_safety_notes(row: pd.Series) -> str:
    notes: list[str] = []
    if clean(row.get("publish_ready"), "no").lower() != "yes":
        notes.append("publish_ready is not yes")
    if clean(row.get("needs_photo_check"), "no").lower() == "yes":
        notes.append("photo needs checking")
    warnings = clean(row.get("warnings"), "")
    if warnings:
        notes.append(f"render warnings: {warnings}")
    return "; ".join(notes)


def build_copy_pack(review_table: str | Path, copy_dir: str | Path, hashtags: list[str]) -> dict[str, Any]:
    review_table = Path(review_table)
    copy_dir = Path(copy_dir)
    copy_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(review_table)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        full_name = clean(row.get("full_name"))
        slug = slugify(full_name)
        caption = build_caption(row, hashtags)
        alt_text = build_alt_text(row)
        safety_notes = build_safety_notes(row)

        caption_path = copy_dir / f"{slug}.caption.txt"
        alt_path = copy_dir / f"{slug}.alt_text.txt"
        caption_path.write_text(caption, encoding="utf-8")
        alt_path.write_text(alt_text, encoding="utf-8")

        rows.append({
            "slug": slug,
            "full_name": full_name,
            "party": clean(row.get("party")),
            "constituency": clean(row.get("constituency")),
            "output_file_rel": clean(row.get("output_file_rel"), ""),
            "caption_file": str(caption_path),
            "alt_text_file": str(alt_path),
            "caption": caption,
            "alt_text": alt_text,
            "hashtags": " ".join(hashtags),
            "publish_ready": clean(row.get("publish_ready"), "no"),
            "review_status": clean(row.get("review_status"), "needs_review"),
            "safety_notes": safety_notes,
        })

    captions_csv = copy_dir / "captions.csv"
    pd.DataFrame(rows).to_csv(captions_csv, index=False, encoding="utf-8-sig")
    manifest = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_table": str(review_table),
        "copy_dir": str(copy_dir),
        "captions_csv": str(captions_csv),
        "items": rows,
        "notes": [
            "Captions and alt text are deterministic draft copy only.",
            "Do not publish until review_status and publish_ready are manually updated in the review table.",
            "Check metrics, names, constituencies, photos, and warning flags before publishing.",
        ],
    }
    manifest_path = copy_dir / "copy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic captions and alt text from an Instagram review table.")
    parser.add_argument("--review-table", default=DEFAULT_REVIEW_TABLE)
    parser.add_argument("--copy-dir", default=DEFAULT_COPY_DIR)
    parser.add_argument("--hashtags", help="Comma-separated hashtags. Defaults to core eirepolitic tags.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_copy_pack(args.review_table, args.copy_dir, normalize_hashtags(args.hashtags))
    print(json.dumps({"success": True, "copy_dir": result["copy_dir"], "items": len(result["items"])}, indent=2))


if __name__ == "__main__":
    main()
