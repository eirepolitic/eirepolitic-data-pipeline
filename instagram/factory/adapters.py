from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .constituency_pilot import (
    DEBATE_KEYS,
    MEMBER_KEYS,
    build_constituency_records,
    load_source_rows,
    render_visual,
)
from .historical_sources import annotate_current_records, load_historical_joined_records
from .party_adapter import (
    build_party_context,
    build_party_records,
    load_party_records,
    party_media_for_slide,
    render_party_assets,
)
from .party_overindex_adapter import (
    build_context as build_party_overindex_context,
    load_party_per_td_overindex_records,
    load_party_share_overindex_records,
    media_for_slide as party_overindex_media_for_slide,
    render_assets as render_party_overindex_assets,
)
from .validation_density import select_density_aware_category_value_scenarios

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
        "display_constituency": constituency,
        "display_constituency_key": constituency_key,
        "result_constituency": constituency,
        "result_constituency_key": constituency_key,
        "result_issue_count": int(record.get("issue_count", len(rows))),
        "result_speech_count": int(record.get("speech_count", 0)),
        "item_key": constituency_key,
        "issue_rows": rows,
        "issue_count": len(rows),
        "speech_count": sum(int(row.get("value", 0)) for row in rows),
        "scenario": record.get("scenario", "batch_item"),
        "synthetic": bool(record.get("synthetic", False)),
        "no_publication": True,
    }


def _constituency_scenarios(records: list[dict[str, Any]], project: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return select_density_aware_category_value_scenarios(
        records,
        key_field="constituency_key",
        label_field="constituency",
        max_items=7,
        dense_min_items=5,
        maximum_min_items=6,
    )


def _write_constituency_cover(path: Path, context: dict[str, Any]) -> None:
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
        (context.get("issue_count", 0), "ISSUE CATEGORIES SHOWN"),
    ]
    for (value, label), center_y in zip(metrics, (255, 600, 945)):
        draw.text((516, center_y - 45), f"{int(value):,}", font=number_font, fill="#f4ead7", anchor="mm")
        draw.text((516, center_y + 75), label, font=label_font, fill="#d8b45f", anchor="mm")
    image.save(path, format="PNG")


def _constituency_assets(item_dir: Path, context: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    assets_dir = item_dir / "assets"
    cover_asset = assets_dir / "cover.png"
    visual_asset = assets_dir / "visual.png"
    _write_constituency_cover(cover_asset, context)
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
    return select_density_aware_category_value_scenarios(
        records,
        key_field="party_key",
        label_field="party",
        max_items=7,
        dense_min_items=5,
        maximum_min_items=6,
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
    "party_issue_share_overindex_v1": FactoryAdapter(
        adapter_id="party_issue_share_overindex_v1",
        load_records=load_party_share_overindex_records,
        build_context=build_party_overindex_context,
        build_scenarios=_party_scenarios,
        render_assets=render_party_overindex_assets,
        media_for_slide=party_overindex_media_for_slide,
    ),
    "party_issue_per_td_overindex_v1": FactoryAdapter(
        adapter_id="party_issue_per_td_overindex_v1",
        load_records=load_party_per_td_overindex_records,
        build_context=build_party_overindex_context,
        build_scenarios=_party_scenarios,
        render_assets=render_party_overindex_assets,
        media_for_slide=party_overindex_media_for_slide,
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
