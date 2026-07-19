from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar
from instagram.visuals.renderers.common import load_yaml, rows_from_sample, write_json

from .catalogues import REPO_ROOT, load_catalogues
from .project import load_project, validate_project

MEMBER_KEYS = [
    "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv",
    "raw/members/oireachtas_members_34th_dail.csv",
]
DEBATE_KEYS = [
    "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv",
    "processed/debates/debate_speeches_classified.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(td|teachta dail|teachta dala|minister|deputy)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_field(rows: list[dict[str, Any]], candidates: list[str], label: str) -> str:
    available: set[str] = set()
    for row in rows:
        available.update(str(key) for key in row)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"No {label} field found. Tried: {candidates}; available: {sorted(available)}")


def read_local_csv(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_source_rows(data_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if data_source == "local":
        members = read_local_csv("tests/fixtures/instagram/members.csv")
        speeches = read_local_csv("tests/fixtures/instagram/debate_issues.csv")
        return members, speeches, {
            "mode": "local",
            "members": "tests/fixtures/instagram/members.csv",
            "speeches": "tests/fixtures/instagram/debate_issues.csv",
        }
    if data_source != "s3":
        raise ValueError(f"Unsupported data source: {data_source}")
    members, member_meta = rows_from_sample(
        {"input": {"mode": "s3_csv_first_available", "keys": MEMBER_KEYS, "required": True}}
    )
    speeches, speech_meta = rows_from_sample(
        {"input": {"mode": "s3_csv_first_available", "keys": DEBATE_KEYS, "required": True}}
    )
    return members, speeches, {"mode": "s3", "members": member_meta, "speeches": speech_meta}


def build_constituency_records(
    members: list[dict[str, Any]],
    speeches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    member_name_field = first_field(members, ["full_name", "member_name", "name", "showAs"], "member name")
    constituency_field = first_field(members, ["constituency", "constituency_name", "latest_constituency_name"], "constituency")
    speaker_field = first_field(speeches, ["Speaker Name", "speaker_name", "speaker", "showAs"], "speaker")
    issue_field = first_field(
        speeches,
        ["PoliticalIssues", "issue_category", "political_issues", "issue", "topic", "category", "label"],
        "issue",
    )

    member_lookup: dict[str, str] = {}
    constituency_members: dict[str, set[str]] = defaultdict(set)
    for row in members:
        member_name = str(row.get(member_name_field) or "").strip()
        constituency = str(row.get(constituency_field) or "").strip()
        key = normalize_text(member_name)
        if not key or not constituency:
            continue
        member_lookup[key] = constituency
        constituency_members[constituency].add(member_name)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    matched_speeches = 0
    unmatched_speeches = 0
    ignored_empty_issue = 0
    for row in speeches:
        issue = str(row.get(issue_field) or "").strip()
        if not issue or issue.upper() in {"NONE", "N/A", "UNKNOWN", "UNCLASSIFIED"}:
            ignored_empty_issue += 1
            continue
        constituency = member_lookup.get(normalize_text(row.get(speaker_field)))
        if not constituency:
            unmatched_speeches += 1
            continue
        counts[constituency][issue] += 1
        matched_speeches += 1

    records: list[dict[str, Any]] = []
    for constituency in sorted(counts):
        issue_counts = counts[constituency]
        if not issue_counts:
            continue
        rows = [
            {"label": label, "value": value}
            for label, value in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        records.append(
            {
                "constituency": constituency,
                "constituency_key": normalize_text(constituency).replace(" ", "-"),
                "member_names": sorted(constituency_members.get(constituency, set())),
                "member_count": len(constituency_members.get(constituency, set())),
                "issue_rows": rows,
                "issue_count": len(rows),
                "speech_count": sum(issue_counts.values()),
                "max_issue_label_length": max(len(row["label"]) for row in rows),
            }
        )

    manifest = {
        "member_name_field": member_name_field,
        "constituency_field": constituency_field,
        "speaker_field": speaker_field,
        "issue_field": issue_field,
        "member_rows": len(members),
        "speech_rows": len(speeches),
        "matched_speeches": matched_speeches,
        "unmatched_speeches": unmatched_speeches,
        "ignored_empty_issue": ignored_empty_issue,
        "constituency_count": len(records),
    }
    if not records:
        raise ValueError("No constituency issue records could be built from the selected data")
    return records, manifest


def _complexity(record: dict[str, Any]) -> int:
    return (
        len(record["constituency"])
        + record["issue_count"] * 12
        + record["max_issue_label_length"]
        + min(record["speech_count"], 100)
    )


def build_scenarios(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    minimum_source = min(
        records,
        key=lambda row: (row["issue_count"], row["speech_count"], len(row["constituency"]), row["constituency"]),
    )
    complexity_values = sorted(_complexity(record) for record in records)
    target = median(complexity_values)
    real_source = min(records, key=lambda row: (abs(_complexity(row) - target), row["constituency"]))

    longest_constituency = max(records, key=lambda row: (len(row["constituency"]), row["constituency"]))
    all_issue_rows = [
        {
            "label": issue["label"],
            "value": issue["value"],
            "originating_constituency": record["constituency"],
        }
        for record in records
        for issue in record["issue_rows"]
    ]
    longest_labels = sorted(all_issue_rows, key=lambda row: (-len(row["label"]), -int(row["value"]), row["label"]))[:8]
    largest_values = sorted(all_issue_rows, key=lambda row: (-int(row["value"]), -len(row["label"]), row["label"]))
    synthetic_rows: list[dict[str, Any]] = []
    for index, label_row in enumerate(longest_labels):
        value_row = largest_values[index % len(largest_values)]
        synthetic_rows.append({"label": label_row["label"], "value": value_row["value"]})

    return {
        "minimum": {
            **minimum_source,
            "scenario": "minimum",
            "synthetic": True,
            "no_publication": True,
            "source_fields": {
                "constituency": minimum_source["constituency"],
                "issue_rows": minimum_source["constituency"],
            },
        },
        "maximum": {
            **longest_constituency,
            "issue_rows": synthetic_rows,
            "issue_count": len(synthetic_rows),
            "speech_count": sum(int(row["value"]) for row in synthetic_rows),
            "scenario": "maximum",
            "synthetic": True,
            "no_publication": True,
            "source_fields": {
                "constituency": longest_constituency["constituency"],
                "issue_labels": [row["originating_constituency"] for row in longest_labels],
                "issue_values": [row["originating_constituency"] for row in largest_values[: len(synthetic_rows)]],
            },
        },
        "real_example": {
            **real_source,
            "scenario": "real_example",
            "synthetic": False,
            "no_publication": True,
            "source_fields": {"constituency": real_source["constituency"], "issue_rows": real_source["constituency"]},
        },
    }


def write_cover_asset(path: Path, scenario: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 860
    image = Image.new("RGB", (width, height), "#173d30")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    draw.rounded_rectangle((70, 70, width - 70, height - 70), radius=42, fill="#214a3b", outline="#d8b45f", width=5)
    title = scenario["constituency"]
    draw.multiline_text((width // 2, 285), title, font=title_font, fill="#f4ead7", anchor="mm", align="center", spacing=12)
    detail = f"{scenario.get('member_count', 0)} current members · {scenario.get('speech_count', 0)} classified speeches"
    draw.text((width // 2, 560), detail, font=body_font, fill="#cbbf9f", anchor="mm")
    marker = "SYNTHETIC TEST" if scenario["synthetic"] else "REAL DATA EXAMPLE"
    draw.text((width // 2, 680), marker, font=body_font, fill="#d8b45f", anchor="mm")
    image.save(path, format="PNG")


def render_visual(path: Path, metadata_path: Path, manifest_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    template = load_yaml(REPO_ROOT / "instagram/visuals/templates/horizontal_bar_draft_v1.yml")
    sample = {
        "visual_id": f"constituency_issue_profile_{scenario['scenario']}",
        "bindings": {"label": "label", "value": "value"},
        "filters": [],
        "grouping": {"grain": "constituency", "key": scenario["constituency_key"]},
        "source_note": "Synthetic validation context" if scenario["synthetic"] else "Joined Oireachtas member and classified debate data",
        "attribution": {"source": "Houses of the Oireachtas / Eirepolitic classification"},
    }
    return horizontal_bar.render(
        template,
        sample,
        scenario["issue_rows"],
        path,
        metadata_path,
        manifest_path,
        {"scenario": scenario["scenario"], "synthetic": scenario["synthetic"]},
    )


def _replace_tokens(value: Any, scenario: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.replace("{{ constituency }}", scenario["constituency"])
    return value


def build_contact_sheet(image_paths: list[Path], output_path: Path, labels: list[str]) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_width = 360
    thumb_height = 450
    label_height = 54
    canvas = Image.new("RGB", (thumb_width * len(images), thumb_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail((thumb_width, thumb_height))
        x = index * thumb_width + (thumb_width - image.width) // 2
        y = label_height + (thumb_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * thumb_width + thumb_width // 2, 24), label, fill="black", anchor="mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def render_project_tests(
    project_path: str | Path,
    *,
    data_source: str = "local",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    project = load_project(project_path)
    validation = validate_project(project=project, catalogues=load_catalogues())
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))
    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    scenarios = build_scenarios(records)
    root = Path(output_root or project.get("output", {}).get("local_root") or "generated_factory_tests")
    if not root.is_absolute():
        root = REPO_ROOT / root
    template_path = REPO_ROOT / "instagram/templates/layouts/title_text_media_v1.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    slide_outputs: dict[str, dict[str, str]] = defaultdict(dict)
    scenario_manifests: dict[str, Any] = {}

    for scenario_name, scenario in scenarios.items():
        scenario_dir = root / scenario_name
        assets_dir = scenario_dir / "assets"
        cover_asset = assets_dir / "constituency_cover.png"
        visual_asset = assets_dir / "issue_profile.png"
        write_cover_asset(cover_asset, scenario)
        visual_manifest = render_visual(
            visual_asset,
            scenario_dir / "metadata/issue_profile.visual.json",
            scenario_dir / "manifests/issue_profile.visual_manifest.json",
            scenario,
        )
        rendered_slides: list[Path] = []
        for slide in sorted(project["slides"], key=lambda item: item["order"]):
            slide_id = slide["slide_id"]
            media_path = cover_asset if slide_id == "cover" else visual_asset
            bindings = {
                key: _replace_tokens(value, scenario)
                for key, value in slide.get("text", {}).items()
            }
            bindings["main_media"] = str(media_path)
            output_path = scenario_dir / f"{slide['order']:02d}_{slide_id}.png"
            result = render_template(template, bindings, output_path)
            if result.warnings:
                raise ValueError(f"Render warnings for {scenario_name}/{slide_id}: {result.warnings}")
            rendered_slides.append(output_path)
            slide_outputs[slide_id][scenario_name] = str(output_path.relative_to(root))
        build_contact_sheet(
            rendered_slides,
            scenario_dir / "contact_sheet.png",
            [slide["slide_id"] for slide in sorted(project["slides"], key=lambda item: item["order"])],
        )
        scenario_manifest = {
            "scenario": scenario_name,
            "constituency": scenario["constituency"],
            "synthetic": scenario["synthetic"],
            "no_publication": scenario["no_publication"],
            "source_fields": scenario["source_fields"],
            "member_names": scenario.get("member_names", []),
            "issue_rows": scenario["issue_rows"],
            "slides": [str(path.relative_to(root)) for path in rendered_slides],
            "visual_manifest": visual_manifest,
        }
        write_json(scenario_dir / "scenario_manifest.json", scenario_manifest)
        scenario_manifests[scenario_name] = scenario_manifest

    for slide_id, outputs in slide_outputs.items():
        paths = [root / outputs[name] for name in ("minimum", "maximum", "real_example")]
        build_contact_sheet(paths, root / "contact_sheets" / f"{slide_id}.png", ["minimum", "maximum", "real example"])

    manifest = {
        "success": True,
        "created_at": utc_now(),
        "project_id": project["project_id"],
        "project_version": project["version"],
        "data_source": data_source,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "scenario_manifests": scenario_manifests,
        "slide_contact_sheets": {
            slide_id: f"contact_sheets/{slide_id}.png" for slide_id in slide_outputs
        },
        "warnings": validation["warnings"],
        "review_state": "needs_review",
        "approved": False,
    }
    write_json(root / "project_validation_manifest.json", manifest)
    return manifest
