from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from PIL import Image, ImageDraw, ImageFont

from instagram.visuals.renderers import horizontal_bar

S3_BUCKET = "eirepolitic-data-pipeline"
MEMBER_KEY = "raw/members/oireachtas_members_34th_dail.csv"
CLASSIFIED_KEY_CANDIDATES = [
    "processed/debates/debate_speeches_classified.csv",
    "processed/debates/debate_speeches_classified_latest.csv",
]
JAN_START = "2026-01-01"
JAN_END = "2026-01-31"


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "unknown"


def _csv_rows(body: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))


def _read_s3_csv(s3: Any, key: str) -> list[dict[str, str]]:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return _csv_rows(obj["Body"].read())


def _first_existing_csv(s3: Any, keys: list[str]) -> tuple[str, list[dict[str, str]]]:
    last_error: Exception | None = None
    for key in keys:
        try:
            return key, _read_s3_csv(s3, key)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read any classified speech key: {keys}; last_error={last_error}")


def _parse_date(value: str) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _field(row: dict[str, str], names: list[str]) -> str:
    lowered = {k.lower().strip(): k for k in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return (row.get(key) or "").strip()
    return ""


def _load_records(mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s3 = boto3.client("s3")
    members = _read_s3_csv(s3, MEMBER_KEY)
    classified_key, speeches = _first_existing_csv(s3, CLASSIFIED_KEY_CANDIDATES)

    member_party: dict[str, str] = {}
    party_member_counts: Counter[str] = Counter()
    for row in members:
        name = _field(row, ["Full Name", "Member Name", "Name", "full_name"])
        party = _field(row, ["Party", "party", "Party Name"])
        if not name or not party:
            continue
        member_party[name.casefold()] = party
        party_member_counts[party] += 1

    party_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    party_total_counts: Counter[str] = Counter()
    january_rows = 0
    matched_rows = 0
    seen_categories: set[str] = set()

    for row in speeches:
        date_value = _field(row, ["Debate Date", "debate_date", "Date"])
        date_iso = _parse_date(date_value)
        if not date_iso or not (JAN_START <= date_iso <= JAN_END):
            continue
        january_rows += 1
        speaker = _field(row, ["Speaker", "Speaker Name", "speaker_name", "Member Name"])
        category = _field(row, ["Issue Category", "issue_category", "Category", "classification"])
        if not speaker or not category:
            continue
        party = member_party.get(speaker.casefold())
        if not party:
            continue
        matched_rows += 1
        seen_categories.add(category)
        party_category_counts[party][category] += 1
        party_total_counts[party] += 1

    parties = sorted(party_member_counts)
    categories = sorted(seen_categories)
    if not parties or not categories:
        raise RuntimeError("January normalization source produced no parties/categories")

    baselines: dict[str, float] = {}
    if mode == "share_pp":
        for category in categories:
            shares = []
            for party in parties:
                total = party_total_counts[party]
                shares.append((party_category_counts[party][category] / total) if total else 0.0)
            baselines[category] = sum(shares) / len(shares)
    elif mode == "per_td":
        for category in categories:
            rates = []
            for party in parties:
                td_count = party_member_counts[party]
                rates.append((party_category_counts[party][category] / td_count) if td_count else 0.0)
            baselines[category] = sum(rates) / len(rates)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    records: list[dict[str, Any]] = []
    for party in parties:
        td_count = int(party_member_counts[party])
        total = int(party_total_counts[party])
        rows: list[dict[str, Any]] = []
        for category in categories:
            count = int(party_category_counts[party][category])
            baseline = baselines[category]
            if mode == "share_pp":
                actual = (count / total) if total else 0.0
                delta = (actual - baseline) * 100.0
                display_value = delta
                value_label = f"+{delta:.1f} pp" if delta > 0 else f"{delta:.1f} pp"
            else:
                actual = (count / td_count) if td_count else 0.0
                delta = actual - baseline
                display_value = delta
                value_label = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            if delta > 0:
                rows.append({
                    "label": category,
                    "value": display_value,
                    "raw_count": count,
                    "actual": actual,
                    "baseline": baseline,
                    "value_label": value_label,
                })
        rows.sort(key=lambda item: item["value"], reverse=True)
        records.append({
            "party": party,
            "party_key": slugify(party),
            "member_count": td_count,
            "speech_count": total,
            "period_start": JAN_START,
            "period_end": JAN_END,
            "rows": rows[:7],
            "mode": mode,
            "source_batch_id": "latest-classified-production",
            "classified_s3_key": classified_key,
        })

    manifest = {
        "source_bucket": S3_BUCKET,
        "member_key": MEMBER_KEY,
        "classified_key": classified_key,
        "period_start": JAN_START,
        "period_end": JAN_END,
        "january_rows": january_rows,
        "matched_rows": matched_rows,
        "party_count": len(parties),
        "category_count": len(categories),
        "mode": mode,
    }
    return records, manifest


def load_party_share_overindex_records(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _load_records("share_pp")


def load_party_per_td_overindex_records(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _load_records("per_td")


def build_context(record: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return dict(record)


def render_cover(path: Path, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1032, 1210), "#0f2f24")
    draw = ImageDraw.Draw(image)
    try:
        party_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 68)
        number_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 74)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        party_font = number_font = label_font = small_font = ImageFont.load_default()

    cx, cy, radius = 516, 375, 225
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline="#d8b45f", width=10)
    party = str(context.get("party", ""))
    words = party.split()
    if len(words) > 1:
        split = max(1, len(words)//2)
        lines = [" ".join(words[:split]), " ".join(words[split:])]
    else:
        lines = [party]
    if len(lines) == 1:
        draw.text((cx, cy), lines[0], font=party_font, fill="#f4ead7", anchor="mm")
    else:
        draw.text((cx, cy-46), lines[0], font=party_font, fill="#f4ead7", anchor="mm")
        draw.text((cx, cy+46), lines[1], font=party_font, fill="#f4ead7", anchor="mm")

    mode = context.get("mode")
    if mode == "share_pp":
        big = f"{int(context.get('speech_count',0)):,}"
        label = "JANUARY SPEECHES"
        second_big = "VS AVG"
        second_label = "PARTY ISSUE SHARE"
    else:
        big = f"{int(context.get('member_count',0)):,}"
        label = "CURRENT TDS"
        second_big = "PER TD"
        second_label = "ISSUE OVER-INDEX"

    draw.text((270, 785), big, font=number_font, fill="#f4ead7", anchor="mm")
    draw.text((270, 852), label, font=label_font, fill="#d8b45f", anchor="mm")
    draw.text((762, 785), second_big, font=number_font, fill="#f4ead7", anchor="mm")
    draw.text((762, 852), second_label, font=label_font, fill="#d8b45f", anchor="mm")
    draw.line((215, 945, 817, 945), fill="#d8b45f", width=3)
    draw.text((516, 1000), "JANUARY 2026", font=label_font, fill="#d8b45f", anchor="mm")
    draw.text((516, 1048), "1 Jan – 31 Jan 2026", font=small_font, fill="#f4ead7", anchor="mm")
    image.save(path, format="PNG")


def render_assets(item_dir: Path, context: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    assets = item_dir / "assets"
    manifests = item_dir / "manifests"
    metadata_dir = item_dir / "metadata"
    assets.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cover = assets / "cover.png"
    render_cover(cover, context)

    rows = context.get("rows") or []
    if not rows:
        raise ValueError(f"No above-average categories for {context.get('party')}")

    sample = {
        "visual_id": f"{context.get('party_key')}-{context.get('mode')}",
        "bindings": {"label": "label", "value": "value"},
        "source_note": "January 2026 Dáil speeches · Houses of the Oireachtas / Eirepolitic classification",
    }
    template = {
        "template_id": "horizontal_bar_draft_v1",
        "params": {
            "width": 1032,
            "height": 1210,
            "max_items": 7,
            "sort": "descending",
            "value_format": "decimal",
        },
        "palette": {
            "background": "#0f2f24",
            "panel": "#0f2f24",
            "text": "#f4ead7",
            "muted": "#c8bda8",
            "accent": "#d8b45f",
            "grid": "#f4ead7",
        },
    }
    visual_png = assets / "issue_profile.png"
    visual_metadata = metadata_dir / "visual.json"
    visual_manifest_path = manifests / "visual_manifest.json"
    visual_manifest = horizontal_bar.render(
        template,
        sample,
        rows,
        visual_png,
        visual_metadata,
        visual_manifest_path,
        {
            "data_origin": "real_s3",
            "source_key": context.get("classified_s3_key"),
            "period_start": JAN_START,
            "period_end": JAN_END,
            "mode": context.get("mode"),
        },
    )
    return {
        "cover": str(cover),
        "issue_profile": str(visual_png),
        "visual_manifest": visual_manifest,
    }


def media_for_slide(slide: dict[str, Any], asset_result: dict[str, Any]) -> str | None:
    slide_id = str(slide.get("slide_id"))
    return asset_result.get(slide_id)


class PartyShareOverindexAdapter:
    def load_records(self, **kwargs: Any):
        return load_party_share_overindex_records(**kwargs)
    def build_context(self, record: dict[str, Any], **kwargs: Any):
        return build_context(record, **kwargs)
    def render_assets(self, item_dir: Path, context: dict[str, Any], project: dict[str, Any]):
        return render_assets(item_dir, context, project)
    def media_for_slide(self, slide: dict[str, Any], asset_result: dict[str, Any]):
        return media_for_slide(slide, asset_result)


class PartyPerTDOverindexAdapter(PartyShareOverindexAdapter):
    def load_records(self, **kwargs: Any):
        return load_party_per_td_overindex_records(**kwargs)
