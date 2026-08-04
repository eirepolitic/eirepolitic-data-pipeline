from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constituency_pilot import (
    MEMBER_SOURCE,
    SPEECH_SOURCE,
    build_constituency_records,
    first_field,
    load_source_rows,
    normalize_text,
    render_visual,
    write_cover_asset,
)
from .historical_sources import annotate_current_records, load_historical_joined_records
from .party_adapter import (
    build_party_context,
    load_party_records,
    party_media_for_slide,
    render_party_assets,
)
from .validation_scenarios import select_real_category_value_scenarios

HistoricalLoader = Callable[
    [str, dict[str, Any], dict[str, Any]],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


@dataclass(frozen=True)
class FactoryAdapter:
    adapter_id: str
    load_records: Callable[[str], tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]
    build_context: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    build_scenarios: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, dict[str, Any]]]
    render_assets: Callable[[Path, dict[str, Any], dict[str, Any]], dict[str, Any]]
    media_for_slide: Callable[[dict[str, Any], dict[str, Path]], Path]
    load_historical_records: HistoricalLoader | None = None


def _constituency_load_records(data_source: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    return annotate_current_records(records, source_manifest), source_manifest, join_manifest


def _constituency_historical_records(
    data_source: str,
    project: dict[str, Any],
    source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_historical_joined_records(
        data_source=data_source,
        project=project,
        current_source_manifest=source_manifest,
        member_logical_key=MEMBER_SOURCE,
        speech_logical_key=SPEECH_SOURCE,
        build_records=build_constituency_records,
    )


def _build_party_records_from_rows(
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

    return records, {
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


def _party_load_records(data_source: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records, source_manifest, join_manifest = load_party_records(data_source)
    return annotate_current_records(records, source_manifest), source_manifest, join_manifest


def _party_historical_records(
    data_source: str,
    project: dict[str, Any],
    source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_historical_joined_records(
        data_source=data_source,
        project=project,
        current_source_manifest=source_manifest,
        member_logical_key=MEMBER_SOURCE,
        speech_logical_key=SPEECH_SOURCE,
        build_records=_build_party_records_from_rows,
    )


def _constituency_context(record: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in record["issue_rows"][:7]]
    constituency = str(record["constituency"])
    constituency_key = str(record["constituency_key"])
    return {
        **record,
        project["granularity"]["label_field"]: constituency,
        "display_label": constituency,
        "display_constituency": record.get("display_constituency", constituency),
        "result_constituency": record.get("result_constituency", constituency),
        "result_constituency_key": record.get("result_constituency_key", constituency_key),
        "result_issue_count": record.get("result_issue_count", record.get("issue_count", len(rows))),
        "result_speech_count": record.get("result_speech_count", record.get("speech_count", 0)),
        "item_key": constituency_key,
        "issue_rows": rows,
        "issue_count": len(rows),
        "scenario": record.get("scenario", "batch_item"),
        "synthetic": bool(record.get("synthetic", False)),
        "no_publication": True,
    }


def _constituency_scenarios(records: list[dict[str, Any]], project: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return select_real_category_value_scenarios(
        records,
        key_field="constituency_key",
        label_field="constituency",
        max_items=7,
    )


def _constituency_assets(item_dir: Path, context: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    assets_dir = item_dir / "assets"
    cover_asset = assets_dir / "cover.png"
    visual_asset = assets_dir / "visual.png"
    write_cover_asset(cover_asset, context)
    visual_manifest = render_visual(
        visual_asset,
        item_dir / "metadata/visual.json",
        item_dir / "manifests/visual_manifest.json",
        context,
    )
    return {
        "paths": {"cover": cover_asset, "visual": visual_asset},
        "visual_manifest": visual_manifest,
    }


def _constituency_media(slide: dict[str, Any], assets: dict[str, Path]) -> Path:
    return assets["visual"] if slide.get("visual") else assets["cover"]


def _party_scenarios(records: list[dict[str, Any]], project: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return select_real_category_value_scenarios(
        records,
        key_field="party_key",
        label_field="party",
        max_items=7,
    )


ADAPTERS: dict[str, FactoryAdapter] = {
    "constituency_issue_profile_v1": FactoryAdapter(
        adapter_id="constituency_issue_profile_v1",
        load_records=_constituency_load_records,
        build_context=_constituency_context,
        build_scenarios=_constituency_scenarios,
        render_assets=_constituency_assets,
        media_for_slide=_constituency_media,
        load_historical_records=_constituency_historical_records,
    ),
    "party_issue_profile_v1": FactoryAdapter(
        adapter_id="party_issue_profile_v1",
        load_records=_party_load_records,
        build_context=build_party_context,
        build_scenarios=_party_scenarios,
        render_assets=render_party_assets,
        media_for_slide=party_media_for_slide,
        load_historical_records=_party_historical_records,
    ),
}


def get_adapter(project: dict[str, Any]) -> FactoryAdapter:
    adapter_id = str(project.get("factory", {}).get("adapter") or project.get("project_id"))
    try:
        return ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(
            f"No factory adapter registered for '{adapter_id}'. "
            "Add a project adapter without changing the generic orchestrator."
        ) from exc
