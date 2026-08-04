from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constituency_pilot import (
    DEBATE_KEYS,
    MEMBER_KEYS,
    build_constituency_records,
    load_source_rows,
    render_visual,
    write_cover_asset,
)
from .historical_sources import annotate_current_records, load_historical_joined_records
from .party_adapter import (
    build_party_context,
    build_party_records,
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
    annotate_current_records(records, source_manifest)
    return records, source_manifest, join_manifest


def _constituency_historical(
    data_source: str,
    project: dict[str, Any],
    current_source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_historical_joined_records(
        data_source=data_source,
        project=project,
        current_source_manifest=current_source_manifest,
        member_logical_key=MEMBER_KEYS[0],
        speech_logical_key=DEBATE_KEYS[0],
        build_records=build_constituency_records,
    )


def _party_historical(
    data_source: str,
    project: dict[str, Any],
    current_source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_historical_joined_records(
        data_source=data_source,
        project=project,
        current_source_manifest=current_source_manifest,
        member_logical_key=MEMBER_KEYS[0],
        speech_logical_key=DEBATE_KEYS[0],
        build_records=build_party_records,
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
    return {"paths": {"cover": cover_asset, "visual": visual_asset}, "visual_manifest": visual_manifest}


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
        load_historical_records=_constituency_historical,
    ),
    "party_issue_profile_v1": FactoryAdapter(
        adapter_id="party_issue_profile_v1",
        load_records=load_party_records,
        build_context=build_party_context,
        build_scenarios=_party_scenarios,
        render_assets=render_party_assets,
        media_for_slide=party_media_for_slide,
        load_historical_records=_party_historical,
    ),
}


def get_adapter(project: dict[str, Any]) -> FactoryAdapter:
    adapter_id = str(project.get("factory", {}).get("adapter") or project.get("project_id"))
    try:
        return ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(
            f"No factory adapter registered for '{adapter_id}'. Add a project adapter without changing the generic orchestrator."
        ) from exc
