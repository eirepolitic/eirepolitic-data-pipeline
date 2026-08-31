from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.visuals.renderers import horizontal_bar
from instagram.visuals.renderers.common import load_yaml

from .catalogues import REPO_ROOT
from .constituency_pilot import first_field, load_source_rows, normalize_text
from .historical_sources import annotate_current_records


def build_party_records(
    members: list[dict[str, Any]],
    speeches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    member_name_field = first_field(members, ["full_name", "member_name", "name", "showAs"], "member name")
    party_field = first_field(members, ["party", "party_name", "latest_party_name"], "party")
    speaker_field = first_field(speeches, ["Speaker Name", "speaker_name", "speaker", "showAs"], "speaker")
    issue_field = first_field(
        speeches,
        ["PoliticalIssues", "issue_category", "political_issues", "issue", "topic", "category", "label"],
        "issue",
    )

    member_lookup: dict[str, str] = {}
    party_members: dict[str, set[str]] = defaultdict(set)
    for row in members:
        member_name = str(row.get(member_name_field) or "").strip()
        party = str(row.get(party_field) or "").strip()
        key = normalize_text(member_name)
        if not key or not party:
            continue
        member_lookup[key] = party
        party_members[party].add(member_name)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    matched_speeches = 0
    unmatched_speeches = 0
    ignored_empty_issue = 0
    for row in speeches:
        issue = str(row.get(issue_field) or "").strip()
        if not issue or issue.upper() in {"NONE", "N/A", "UNKNOWN", "UNCLASSIFIED"}:
            ignored_empty_issue += 1
            continue
        party = member_lookup.get(normalize_text(row.get(speaker_field)))
        if not party:
            unmatched_speeches += 1
            continue
        counts[party][issue] += 1
        matched_speeches += 1

    records: list[dict[str, Any]] = []
    for party in sorted(counts):
        issue_counts = counts[party]
        rows = [
            {"label": label, "value": value}
            for label, value in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        if not rows:
            continue
        records.append({
            "party": party,
            "party_key": normalize_text(party).replace(" ", "-"),
            "member_names": sorted(party_members.get(party, set())),
            "member_count": len(party_members.get(party, set())),
            "issue_rows": rows,
            "issue_count": len(rows),
            "speech_count": sum(issue_counts.values()),
            "max_issue_label_length": max(len(row["label"]) for row in rows),
        })

    if not records:
        raise ValueError("No party issue records could be built from the selected data")

    join_manifest = {
        "member_name_field": member_name_field,
        "party_field": party_field,
        "speaker_field": speaker_field,
        "issue_field": issue_field,
        "member_rows": len(members),
        "speech_rows": len(speeches),
        "matched_speeches": matched_speeches,
        "unmatched_speeches": unmatched_speeches,
        "ignored_empty_issue": ignored_empty_issue,
        "party_count": len(records),
    }
    return records, join_manifest


def load_party_records(data_source: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_party_records(members, speeches)
    date_field = first_field(speeches, ["Debate Date", "debate_date", "date"], "debate date")
    dates = sorted(str(row.get(date_field) or "").strip() for row in speeches if str(row.get(date_field) or "").strip())
    if not dates:
        raise ValueError("No debate dates available for party issue profile period")
    for record in records:
        record["period_start"] = dates[0]
        record["period_end"] = dates[-1]
    join_manifest["date_field"] = date_field
    join_manifest["period_start"] = dates[0]
    join_manifest["period_end"] = dates[-1]
    annotate_current_records(records, source_manifest)
    return records, source_manifest, join_manifest


def build_party_context(record: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in record["issue_rows"][:7]]
    return {
        **record,
        "display_label": record["party"],
        "item_key": record["party_key"],
        "issue_rows": rows,
        "issue_count": len(rows),
        "scenario": record.get("scenario", "batch_item"),
        "synthetic": bool(record.get("synthetic", False)),
        "no_publication": True,
    }


def _write_party_cover(path: Path, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1032, 1210), "#173d30")
    draw = ImageDraw.Draw(image)
    try:
        number_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 124)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 38)
    except OSError:
        number_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.rounded_rectangle((34, 34, 998, 1176), radius=42, fill="#214a3b", outline="#d8b45f", width=5)
    metrics = [
        (context.get("member_count", 0), "CURRENT TDS"),
        (context.get("speech_count", 0), "CLASSIFIED SPEECHES"),
    ]
    centers = [280, 650]
    for (value, label), center_y in zip(metrics, centers):
        draw.text((516, center_y - 45), f"{int(value):,}", font=number_font, fill="#f4ead7", anchor="mm")
        draw.text((516, center_y + 75), label, font=label_font, fill="#d8b45f", anchor="mm")
    period = f"{context.get('period_start', '')} – {context.get('period_end', '')}"
    draw.text((516, 930), period, font=label_font, fill="#f4ead7", anchor="mm")
    draw.text((516, 1000), "DEBATE COVERAGE", font=label_font, fill="#d8b45f", anchor="mm")
    image.save(path, format="PNG")


def render_party_assets(item_dir: Path, context: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    assets = item_dir / "assets"
    cover = assets / "cover.png"
    visual = assets / "visual.png"
    _write_party_cover(cover, context)
    template = load_yaml(REPO_ROOT / "instagram/visuals/templates/horizontal_bar_draft_v1.yml")
    sample = {
        "visual_id": f"party_issue_profile_{context['scenario']}",
        "bindings": {"label": "label", "value": "value"},
        "filters": [],
        "grouping": {"grain": "party", "key": context["party_key"]},
        "source_note": f"Dáil speeches · {context.get('period_start', '')} – {context.get('period_end', '')} · Houses of the Oireachtas / Eirepolitic classification",
        "attribution": {"source": "Houses of the Oireachtas / Eirepolitic classification"},
    }
    visual_manifest = horizontal_bar.render(
        template,
        sample,
        context["issue_rows"],
        visual,
        item_dir / "metadata/visual.json",
        item_dir / "manifests/visual_manifest.json",
        {
            "scenario": context["scenario"],
            "synthetic": context["synthetic"],
            "data_origin": context.get("data_origin"),
            "source_batch_id": context.get("source_batch_id"),
        },
    )
    return {"paths": {"cover": cover, "visual": visual}, "visual_manifest": visual_manifest}


def party_media_for_slide(slide: dict[str, Any], assets: dict[str, Path]) -> Path:
    return assets["visual"] if slide.get("visual") else assets["cover"]
