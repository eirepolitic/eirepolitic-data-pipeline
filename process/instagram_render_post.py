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
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    "glossary": "slide_glossary.html",
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


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = strip_accents(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(td|teachta dail|teachta dala|minister|deputy)\b", " ", text)
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


def format_metric(value: Any, suffix: str = "") -> str:
    text = coalesce_text(value)
    if not text:
        return f"TODO{suffix}" if suffix else "TODO"
    return f"{text}{suffix}"


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


def pick_issue_column(df_debate: pd.DataFrame) -> Optional[str]:
    for candidate in [
        "PoliticalIssues",
        "political_issues",
        "issue",
        "Issue",
        "issue_label",
        "category",
        "label",
    ]:
        if candidate in df_debate.columns:
            return candidate
    return None


def pick_speaker_column(df_debate: pd.DataFrame) -> Optional[str]:
    if "Speaker Name" in df_debate.columns:
        return "Speaker Name"
    if "speaker_name" in df_debate.columns:
        return "speaker_name"
    return None


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
            base = base.merge(
                photos[["member_code", "photo_url"]].drop_duplicates(subset=["member_code"]),
                on="member_code",
                how="left",
            )
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
            base = base.merge(
                summaries[["member_code", "background"]].drop_duplicates(subset=["member_code"]),
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


def build_issue_records(df_debate: pd.DataFrame, member_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if df_debate.empty:
        return []

    speaker_col = pick_speaker_column(df_debate)
    issue_col = pick_issue_column(df_debate)
    if not speaker_col or not issue_col:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df_debate.iterrows():
        speaker_key = normalize_name(row.get(speaker_col, ""))
        issue = str(row.get(issue_col, "") or "").strip()
        if not speaker_key or not issue or issue.upper() == "NONE":
            continue
        member = member_lookup.get(speaker_key)
        if not member:
            continue
        records.append(
            {
                "member_key": speaker_key,
                "constituency_key": member.get("constituency_key"),
                "issue": issue,
            }
        )
    return records


def issue_counts_from_records(records: List[Dict[str, Any]], *, constituency_key: Optional[str] = None, member_key: Optional[str] = None) -> Counter:
    counter: Counter = Counter()
    for rec in records:
        if constituency_key and rec.get("constituency_key") != constituency_key:
            continue
        if member_key and rec.get("member_key") != member_key:
            continue
        counter[rec["issue"]] += 1
    return counter


def build_chart_axis(max_value: int) -> Dict[str, Any]:
    if max_value <= 0:
        return {"ticks": [0, 1], "grid_lines": [{"left_pct": 0}, {"left_pct": 100}], "max_value": 1}

    rough_step = max(1, math.ceil(max_value / 3))
    magnitude = 10 ** max(0, len(str(rough_step)) - 1)
    step = max(1, math.ceil(rough_step / magnitude) * magnitude)
    axis_max = step * math.ceil(max_value / step)
    ticks = list(range(0, axis_max + 1, step))
    if ticks[-1] != axis_max:
        ticks.append(axis_max)
    grid_lines = [{"left_pct": round((tick / axis_max) * 100, 4)} for tick in ticks]
    return {"ticks": ticks, "grid_lines": grid_lines, "max_value": axis_max}


def make_issue_rows(counter: Counter, limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    top = counter.most_common(limit)
    max_value = top[0][1] if top else 0
    axis = build_chart_axis(max_value)
    axis_max = axis["max_value"]
    for label, count in top:
        percent = round((count / axis_max) * 100, 4) if axis_max else 0
        rows.append({"label": label, "count": count, "percent": percent})
    return rows, axis


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

    members_in_constituency = df_members[df_members["constituency_key"] == constituency_key].copy()
    if members_in_constituency.empty:
        available = sorted(set(df_members.get("constituency", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:20]
        raise RuntimeError(
            f"No members matched constituency '{constituency_name}'. Sample available constituencies: {available}"
        )

    issue_records = build_issue_records(datasets["debate_issues"], member_lookup)
    constituency_issue_counts = issue_counts_from_records(issue_records, constituency_key=constituency_key)

    speech_count_map = Counter(rec["member_key"] for rec in issue_records if rec.get("constituency_key") == constituency_key)
    members_in_constituency["speech_count"] = members_in_constituency["member_key"].map(lambda x: speech_count_map.get(x, 0))

    requested_member = coalesce_text(spec["data"].get("member_name"))
    selected_member_row = None
    if requested_member:
        requested_key = normalize_name(requested_member)
        matches = members_in_constituency[members_in_constituency["member_key"] == requested_key]
        if not matches.empty:
            selected_member_row = matches.iloc[0]
    if selected_member_row is None:
        members_in_constituency = members_in_constituency.sort_values(by=["speech_count", "full_name"], ascending=[False, True])
        selected_member_row = members_in_constituency.iloc[0]

    member_key = selected_member_row["member_key"]
    member_issue_counts = issue_counts_from_records(issue_records, member_key=member_key)

    constituency_top_issue = constituency_issue_counts.most_common(1)[0][0] if constituency_issue_counts else "TODO"
    member_top_issue = member_issue_counts.most_common(1)[0][0] if member_issue_counts else "TODO"

    metrics = spec.get("data", {}).get("metrics", {})
    constituency = {
        "name": constituency_name,
        "member_count": int(len(members_in_constituency)),
        "party_count": int(members_in_constituency["party"].fillna("").replace("", pd.NA).dropna().nunique()),
        "speech_count": int(sum(constituency_issue_counts.values())),
        "image_url": pick_constituency_image(datasets["constituency_images"], constituency_name),
        "top_issue_label": constituency_top_issue,
        "vote_participation_pct": format_metric(metrics.get("vote_participation_pct"), "%"),
        "speech_rank": format_metric(metrics.get("constituency_speech_rank") or metrics.get("speech_rank")),
    }

    member = {
        "full_name": coalesce_text(selected_member_row.get("full_name")) or "TODO: member",
        "party": coalesce_text(selected_member_row.get("party")),
        "constituency": coalesce_text(selected_member_row.get("constituency")) or constituency_name,
        "photo_url": coalesce_text(selected_member_row.get("photo_url")),
        "background": coalesce_text(selected_member_row.get("background")),
        "speech_count": safe_int(selected_member_row.get("speech_count")),
        "top_issue_label": member_top_issue,
        "vote_participation_pct": format_metric(metrics.get("vote_participation_pct"), "%"),
        "speech_rank": format_metric(metrics.get("speech_rank")),
        "member_key": member_key,
    }

    top_issue = {
        "label": constituency_top_issue,
        "count": int(constituency_issue_counts.most_common(1)[0][1]) if constituency_issue_counts else 0,
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
        "style": spec.get("style", {}),
        "data": spec.get("data", {}),
        "slide_size": spec["post"]["slide_size"],
        "slides": spec["slides"],
        "constituency": constituency,
        "member": member,
        "top_issue": top_issue,
        "datasets_used": datasets_used,
        "glossary": spec.get("data", {}).get("glossary", {}),
        "constituency_issue_counts": dict(constituency_issue_counts),
        "member_issue_counts": dict(member_issue_counts),
    }


def prepare_slide_payload(base_context: Dict[str, Any], slide: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(base_context)
    payload["slide"] = slide

    if slide["type"] == "top_issues":
        content = slide.get("content", {})
        scope = content.get("scope", "constituency")
        issue_limit = int(content.get("issue_limit") or base_context.get("data", {}).get("issue_limit", 8))
        if scope == "member":
            counter = Counter(base_context.get("member_issue_counts", {}))
            title = base_context["member"]["full_name"]
        else:
            counter = Counter(base_context.get("constituency_issue_counts", {}))
            title = base_context["constituency"]["name"]
        rows, axis = make_issue_rows(counter, issue_limit)
        payload["issues"] = rows
        payload["axis"] = axis
        payload["issues_scope_title"] = title

    return payload


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
    enabled_slides = [slide for slide in spec["slides"] if slide.get("enabled", True)]
    for idx, slide in enumerate(enabled_slides, start=1):
        template_name = TEMPLATE_BY_TYPE[slide["type"]]
        template = env.get_template(template_name)
        payload = prepare_slide_payload(context, slide)
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
