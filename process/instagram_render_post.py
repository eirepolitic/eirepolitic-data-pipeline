"""
instagram_render_post.py

Generate a test Instagram constituency carousel from existing S3 datasets.

Outputs:
  generated_posts/<post_slug>/
    html/
    png/
    post_context.json

This script:
- loads a YAML post spec
- reads existing datasets from S3
- builds a constituency context
- renders HTML with Jinja2
- exports PNG screenshots with Playwright

Scope:
- visuals only
- no caption generation
- no publishing automation
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import boto3
import pandas as pd
import yaml
from botocore.exceptions import ClientError
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import async_playwright


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
DEFAULT_REGION = os.getenv("AWS_REGION", "ca-central-1")
TEMPLATES_DIR = Path("instagram/templates")

DATASET_CANDIDATES = {
    "members": ["raw/members/oireachtas_members_34th_dail.csv"],
    "member_summaries": ["processed/members/members_summaries.csv"],
    "member_photos": [
        "processed/members/member_photos/members_photo_urls.csv",
        "processed/members/members_photo_urls.csv",
    ],
    "debate_issues": ["processed/debates/debate_speeches_classified.csv"],
    "constituency_images": ["processed/constituencies/constituency_images.csv"],
}

TEMPLATE_BY_TYPE = {
    "constituency_overview": "slide_overview.html",
    "top_issues": "slide_top_issues.html",
    "member_profile": "slide_member_profile.html",
    "methodology": "slide_methodology.html",
}


class S3CSVLoader:
    def __init__(self, bucket: str, region: str) -> None:
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.used_keys: Dict[str, Optional[str]] = {}

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def read_first_csv(self, label: str, keys: Iterable[str]) -> pd.DataFrame:
        for key in keys:
            if not self._exists(key):
                continue
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            text = obj["Body"].read().decode("utf-8-sig", errors="replace")
            self.used_keys[label] = key
            return pd.read_csv(io.StringIO(text))
        self.used_keys[label] = None
        return pd.DataFrame()


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(td|teachta d[aá]la|minister|deputy)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_constituency(value: Any) -> str:
    return normalize_name(value)


def coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        if not isinstance(value, str):
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
        text = str(value or "").strip()
        if text:
            return text
    return None


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return 0


def build_issue_counts(df_debate: pd.DataFrame, member_lookup: Dict[str, Dict[str, Any]], constituency_key: str) -> Counter:
    counter: Counter = Counter()
    if df_debate.empty:
        return counter

    speaker_col = "Speaker Name" if "Speaker Name" in df_debate.columns else "speaker_name"
    issue_col = None
    for candidate in ["issue", "Issue", "issue_label", "category", "label"]:
        if candidate in df_debate.columns:
            issue_col = candidate
            break
    if speaker_col not in df_debate.columns or not issue_col:
        return counter

    for _, row in df_debate.iterrows():
        speaker_key = normalize_name(row.get(speaker_col, ""))
        issue = str(row.get(issue_col, "") or "").strip()
        if not speaker_key or not issue or issue.upper() == "NONE":
            continue
        member = member_lookup.get(speaker_key)
        if not member:
            continue
        if member.get("constituency_key") != constituency_key:
            continue
        counter[issue] += 1
    return counter


def pick_constituency_image(df_images: pd.DataFrame, constituency_name: str) -> Optional[str]:
    if df_images.empty:
        return None
    key = normalize_constituency(constituency_name)
    for _, row in df_images.iterrows():
        filename = normalize_constituency(row.get("filename", ""))
        if filename == key or key in filename or filename in key:
            return coalesce_text(row.get("url"), row.get("s3_url"))
    return None


def build_member_table(df_members: pd.DataFrame, df_photos: pd.DataFrame, df_summaries: pd.DataFrame) -> pd.DataFrame:
    if df_members.empty:
        raise RuntimeError("Members dataset is required for constituency carousel generation.")

    base = df_members.copy()
    base["member_key"] = base["full_name"].map(normalize_name)
    base["constituency_key"] = base["constituency"].map(normalize_constituency)

    if not df_photos.empty:
        photos = df_photos.copy()
        if "member_code" in photos.columns and "photo_url" in photos.columns:
            photo_cols = ["member_code", "photo_url"]
            base = base.merge(photos[photo_cols].drop_duplicates(subset=["member_code"]), on="member_code", how="left")
        elif "full_name" in photos.columns and "photo_url" in photos.columns:
            photos["member_key"] = photos["full_name"].map(normalize_name)
            base = base.merge(
                photos[["member_key", "photo_url"]].drop_duplicates(subset=["member_key"]),
                on="member_key",
                how="left",
            )

    if not df_summaries.empty:
        summaries = df_summaries.copy()
        if "member_code" in summaries.columns and "background" in summaries.columns:
            summary_cols = ["member_code", "background"]
            base = base.merge(
                summaries[summary_cols].drop_duplicates(subset=["member_code"]),
                on="member_code",
                how="left",
            )
        elif "full_name" in summaries.columns and "background" in summaries.columns:
            summaries["member_key"] = summaries["full_name"].map(normalize_name)
            base = base.merge(
                summaries[["member_key", "background"]].drop_duplicates(subset=["member_key"]),
                on="member_key",
                how="left",
            )

    return base


def build_post_context(spec: Dict[str, Any], loader: S3CSVLoader) -> Dict[str, Any]:
    datasets = {label: loader.read_first_csv(label, keys) for label, keys in DATASET_CANDIDATES.items()}

    df_members = build_member_table(datasets["members"], datasets["member_photos"], datasets["member_summaries"])
    constituency_name = spec["data"]["constituency"]
    constituency_key = normalize_constituency(constituency_name)

    member_lookup = {
        normalize_name(row.get("full_name")): row.to_dict()
        for _, row in df_members.iterrows()
        if str(row.get("full_name", "")).strip()
    }

    issue_counts = build_issue_counts(datasets["debate_issues"], member_lookup, constituency_key)

    members_in_constituency = df_members[df_members["constituency_key"] == constituency_key].copy()
    if members_in_constituency.empty:
        available = sorted(set(df_members.get("constituency", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:20]
        raise RuntimeError(
            f"No members matched constituency '{constituency_name}'. Sample available constituencies: {available}"
        )

    speech_count_map = Counter()
    if not datasets["debate_issues"].empty:
        debate_df = datasets["debate_issues"]
        speaker_col = "Speaker Name" if "Speaker Name" in debate_df.columns else "speaker_name"
        if speaker_col in debate_df.columns:
            for speaker in debate_df[speaker_col].fillna(""):
                key = normalize_name(speaker)
                member = member_lookup.get(key)
                if member and member.get("constituency_key") == constituency_key:
                    speech_count_map[key] += 1

    members_in_constituency["speech_count"] = members_in_constituency["member_key"].map(lambda x: speech_count_map.get(x, 0))

    requested_member = coalesce_text(spec["data"].get("member_name"))
    selected_member_row = None
    if requested_member:
        requested_key = normalize_name(requested_member)
        matches = members_in_constituency[members_in_constituency["member_key"] == requested_key]
        if not matches.empty:
            selected_member_row = matches.iloc[0]
    if selected_member_row is None:
        members_in_constituency = members_in_constituency.sort_values(
            by=["speech_count", "full_name"], ascending=[False, True]
        )
        selected_member_row = members_in_constituency.iloc[0]

    issues = []
    max_issue_count = max(issue_counts.values()) if issue_counts else 0
    for label, count in issue_counts.most_common(spec["data"].get("issue_limit", 5)):
        percent = round((count / max_issue_count) * 100, 2) if max_issue_count else 0
        issues.append({"label": label, "count": count, "percent": percent})

    top_issue = issues[0] if issues else {"label": "TODO: no issue data", "count": 0, "percent": 0}

    constituency = {
        "name": constituency_name,
        "member_count": int(len(members_in_constituency)),
        "party_count": int(members_in_constituency["party"].fillna("").replace("", pd.NA).dropna().nunique()),
        "speech_count": int(sum(issue_counts.values())),
        "image_url": pick_constituency_image(datasets["constituency_images"], constituency_name),
    }

    member = {
        "full_name": coalesce_text(selected_member_row.get("full_name")) or "TODO: member",
        "party": coalesce_text(selected_member_row.get("party")),
        "constituency": coalesce_text(selected_member_row.get("constituency")) or constituency_name,
        "photo_url": coalesce_text(selected_member_row.get("photo_url")),
        "background": coalesce_text(selected_member_row.get("background")),
        "speech_count": safe_int(selected_member_row.get("speech_count")),
    }

    lead_member = {
        "full_name": member["full_name"],
    }

    datasets_used = []
    for label, key in loader.used_keys.items():
        if key:
            datasets_used.append(f"s3://{loader.bucket}/{key}")
        else:
            datasets_used.append(f"{label}: TODO missing")

    return {
        "post": spec["post"],
        "branding": spec["branding"],
        "slide_size": spec["post"]["slide_size"],
        "slides": spec["slides"],
        "constituency": constituency,
        "issues": issues,
        "member": member,
        "lead_member": lead_member,
        "top_issue": top_issue,
        "datasets_used": datasets_used,
    }


def render_slides(spec: Dict[str, Any], context: Dict[str, Any], output_dir: Path) -> List[Path]:
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered_paths: List[Path] = []
    for idx, slide in enumerate(spec["slides"], start=1):
        template_name = TEMPLATE_BY_TYPE[slide["type"]]
        template = env.get_template(template_name)
        payload = dict(context)
        payload["slide"] = slide
        html = template.render(**payload)
        file_name = f"{idx:02d}_{slide['key']}.html"
        out_path = html_dir / file_name
        out_path.write_text(html, encoding="utf-8")
        rendered_paths.append(out_path)
    return rendered_paths


async def screenshot_html_files(html_paths: List[Path], output_dir: Path, width: int, height: int) -> None:
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        for html_path in html_paths:
            await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            png_path = png_dir / f"{html_path.stem}.png"
            await page.screenshot(path=str(png_path), full_page=False)
        await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to YAML post spec")
    parser.add_argument("--constituency", help="Optional override for constituency name")
    parser.add_argument("--member-name", help="Optional override for member name")
    parser.add_argument("--output-dir", help="Optional override for output root")
    parser.add_argument("--skip-screenshots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    if args.constituency:
        spec.setdefault("data", {})["constituency"] = args.constituency
    if args.member_name:
        spec.setdefault("data", {})["member_name"] = args.member_name
    if args.output_dir:
        spec.setdefault("post", {})["output_root"] = args.output_dir

    output_root = Path(spec["post"].get("output_root", "generated_posts"))
    post_slug = spec["post"]["slug"]
    output_dir = output_root / post_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = S3CSVLoader(bucket=DEFAULT_BUCKET, region=DEFAULT_REGION)
    context = build_post_context(spec, loader)

    context_path = output_dir / "post_context.json"
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")

    html_paths = render_slides(spec, context, output_dir)

    if not args.skip_screenshots:
        asyncio.run(
            screenshot_html_files(
                html_paths=html_paths,
                output_dir=output_dir,
                width=int(spec["post"]["slide_size"]["width"]),
                height=int(spec["post"]["slide_size"]["height"]),
            )
        )

    print(f"Done. Output folder: {output_dir}")
    for html_path in html_paths:
        print(f"HTML: {html_path}")
        png_path = output_dir / "png" / f"{html_path.stem}.png"
        if png_path.exists():
            print(f"PNG:  {png_path}")


if __name__ == "__main__":
    main()
