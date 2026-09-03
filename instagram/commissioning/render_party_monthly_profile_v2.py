from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset
from instagram.factory.run_party_monthly_profile_v2 import (
    COVER_TITLE,
    LOGO_BORDER_WIDTH,
    LOGO_SCALE_OVERRIDES,
    LOGO_SIZE,
    LOGO_TOP,
    _display_party_name,
    _prepare_logo,
)
from instagram.factory.periods import resolve_monthly_period
from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar

PERIOD = "2026-07"
SOURCE_ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v1/period={PERIOD}")
OUTPUT_ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v2/period={PERIOD}")
EXPECTED_PARTY_COUNT = 11
EXPECTED_SLIDE_COUNT = 55

VARIANT_2_SOURCE_COMMIT = "7f10fb0454d0da6d58b4c5c5b2e5aacc39844052"
VARIANT_2_SOURCE_RUN = 203
VARIANT_2_SOURCE_RUN_ID = 33448590338
VARIANT_2_RENDERER_BLOB = "28c6d77cce90449dca2d5a6862b42badb01fee8b"
VARIANT_2_LAYOUT_BLOB = "c409545caafa0a36c79297e530f1c1ad1d7f784b"
VARIANT_2_TEMPLATE_RENDERER_BLOB = "662d8457d325d35552d08a43a11aee9e678f2704"
TITLE_MEDIA_LAYOUT = Path("instagram/templates/layouts/title_text_media_v1.json")
PRESENTATION_LABELS_PATH = Path("instagram/reference/issue_presentation_labels.yml")
MIN_VISUAL_ROWS = 4

ANALYTICAL_TITLES = {
    "02_most_discussed_issues": "Most Discussed Issues",
    "03_more_than_average": "Issues Discussed More Than Average",
    "04_more_per_td": "Issues Discussed More Than Average per TD",
}

V2_GLOSSARY = [
    (
        "Most Discussed Issues",
        "The issues this party/group talked about most often during the month, based on the number of classified speeches.",
    ),
    (
        "Issues Discussed More Than Average",
        "Issues that made up a larger share of this party/group's classified speeches than the simple average across parties. Values show percentage points above average.",
    ),
    (
        "Issues Discussed More Than Average per TD",
        "Issues where this party/group recorded more classified speeches per TD than the simple average across parties. This adjusts the comparison for party size.",
    ),
    (
        "Classified Speeches",
        "Speech segments assigned to an issue category. Counts show how often an issue was discussed, not the party/group's position on it.",
    ),
]


def _render_cover(path: Path, data: dict, s3, period) -> dict:
    party = str(data["party"])
    party_key = str(data["party_key"])
    speech_count = int(data["classified_speeches"])
    td_count = int(data["td_count"])

    image = profile._base_slide()
    profile._draw_title(image, [COVER_TITLE, period.label])
    draw = ImageDraw.Draw(image)

    asset = resolve_party_asset(party)
    if asset.party_key != party_key:
        raise RuntimeError(
            f"Registry party_key mismatch for {party!r}: source={party_key!r}, registry={asset.party_key!r}"
        )
    logo, asset_lineage = fetch_logo(s3, asset)
    logo, logo_scale, neutral_cleanup_pixels = _prepare_logo(logo, party_key)

    logo_left = (profile.W - LOGO_SIZE) // 2
    logo_right = logo_left + LOGO_SIZE - 1
    logo_bottom = LOGO_TOP + LOGO_SIZE - 1
    image.paste(logo, (logo_left, LOGO_TOP))
    draw.rectangle(
        (logo_left, LOGO_TOP, logo_right, logo_bottom),
        outline=profile.ACCENT,
        width=LOGO_BORDER_WIDTH,
    )

    number_font = profile._font(72, True)
    label_font = profile._font(25, True)
    small_font = profile._font(24)
    avg = speech_count / td_count if td_count else 0.0
    draw.text((294, 955), f"{speech_count:,}", font=number_font, fill=profile.TEXT, anchor="mm")
    draw.text((294, 1022), "CLASSIFIED SPEECHES", font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.text((786, 955), f"{avg:.1f}", font=number_font, fill=profile.TEXT, anchor="mm")
    draw.text((786, 1022), "AVG SPEECHES PER TD", font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.line((239, 1115, 841, 1115), fill=profile.ACCENT, width=3)
    draw.text((540, 1170), period.label.upper(), font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.text((540, 1218), profile._period_dates(period), font=small_font, fill=profile.TEXT, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

    return {
        "party_asset": asset_lineage,
        "cover_title": COVER_TITLE,
        "cover_title_period": period.label,
        "display_party_name": _display_party_name(party),
        "logo_geometry": {
            "square_size": [LOGO_SIZE, LOGO_SIZE],
            "top": LOGO_TOP,
            "centered": True,
            "artwork_scale": logo_scale,
            "border": {
                "enabled": True,
                "color": profile.ACCENT,
                "width_px": LOGO_BORDER_WIDTH,
                "position": "inside_square",
            },
        },
        "logo_resampling_cleanup": {
            "enabled": party_key == "social-democrats",
            "neutral_pixels_replaced": neutral_cleanup_pixels,
            "purpose": "remove_neutral_gray_pixels_introduced_by_lanczos_downscaling",
        },
    }


def _render_v2_glossary(path: Path) -> None:
    image = profile._base_slide()
    profile._draw_title(image, ["Glossary"])
    draw = ImageDraw.Draw(image)
    term_font = profile._font(29, True)
    body_font = profile._font(23)
    y = 225

    for term, body in V2_GLOSSARY:
        draw.text((135, y), term, font=term_font, fill=profile.TEXT, anchor="la")
        bbox = draw.textbbox((135, y), term, font=term_font, anchor="la")
        underline_y = bbox[3] + 8
        draw.line((bbox[0], underline_y, bbox[2], underline_y), fill=profile.ACCENT, width=2)
        body_y = underline_y + 22
        for line in textwrap.wrap(body, width=79):
            draw.text((135, body_y), line, font=body_font, fill=profile.TEXT, anchor="la")
            body_y += 34
        y = body_y + 42

    if y > 1315:
        raise RuntimeError(f"V2 glossary overflowed slide: final y={y}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _variant_2_template(value_format: str) -> dict:
    return {
        "template_id": "horizontal_bar_draft_v1",
        "params": {
            "width": 1032,
            "height": 1210,
            "max_items": 7,
            "sort": "descending",
            "value_format": value_format,
            "min_visual_rows": MIN_VISUAL_ROWS,
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


def _render_variant_2_chart(
    output_path: Path,
    *,
    party: str,
    party_key: str,
    period,
    rows: list[dict],
    slide_title: str,
    value_format: str,
    metric_id: str,
) -> dict:
    party_dir = output_path.parent.parent
    assets_dir = party_dir / "assets" / "variant-2"
    metadata_dir = party_dir / "metadata" / "variant-2"
    visual_path = assets_dir / f"{output_path.stem}-visual.png"
    visual_metadata = metadata_dir / f"{output_path.stem}-visual.json"
    visual_manifest_path = metadata_dir / f"{output_path.stem}-visual-manifest.json"
    assets_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample = {
        "visual_id": f"{party_key}-{metric_id}-variant-2",
        "bindings": {"label": "label", "value": "value"},
        "source_note": f"{period.label} Dáil speeches · Houses of the Oireachtas / Eirepolitic classification",
    }
    input_metadata = {
        "data_origin": "QA-passed July 2026 party monthly profile metrics",
        "source_metrics_manifest": str(SOURCE_ROOT / "parties" / party_key / "manifest.json"),
        "period_start": period.start.isoformat(),
        "period_end": period.end.isoformat(),
        "metric_id": metric_id,
        "presentation_labels_path": str(PRESENTATION_LABELS_PATH),
        "approved_visual_variant": 2,
        "approved_visual_source_run": VARIANT_2_SOURCE_RUN,
        "approved_visual_source_run_id": VARIANT_2_SOURCE_RUN_ID,
        "min_visual_rows": MIN_VISUAL_ROWS,
    }
    visual_manifest = horizontal_bar.render(
        _variant_2_template(value_format),
        sample,
        rows,
        visual_path,
        visual_metadata,
        visual_manifest_path,
        input_metadata,
    )
    if visual_manifest.get("warnings"):
        raise RuntimeError(
            f"Variant 2 visual warnings for {party}/{metric_id}: {visual_manifest['warnings']}"
        )

    layout = json.loads(TITLE_MEDIA_LAYOUT.read_text(encoding="utf-8"))
    final_result = render_template(
        layout,
        {"slide_title": slide_title, "main_media": str(visual_path)},
        output_path,
    )
    if final_result.warnings:
        raise RuntimeError(
            f"Variant 2 outer-layout warnings for {party}/{metric_id}: {final_result.warnings}"
        )

    return {
        "variant": 2,
        "source_family": "Matplotlib final January commissioning",
        "source_commit": VARIANT_2_SOURCE_COMMIT,
        "source_run_number": VARIANT_2_SOURCE_RUN,
        "source_run_id": VARIANT_2_SOURCE_RUN_ID,
        "renderer": "instagram.visuals.renderers.horizontal_bar",
        "renderer_blob": VARIANT_2_RENDERER_BLOB,
        "outer_layout": "title_text_media_v1",
        "outer_layout_blob": VARIANT_2_LAYOUT_BLOB,
        "template_renderer_blob": VARIANT_2_TEMPLATE_RENDERER_BLOB,
        "slide_title": slide_title,
        "metric_id": metric_id,
        "value_format": value_format,
        "min_visual_rows": MIN_VISUAL_ROWS,
        "visual_asset": str(visual_path.relative_to(OUTPUT_ROOT)),
        "visual_metadata": str(visual_metadata.relative_to(OUTPUT_ROOT)),
        "visual_manifest": str(visual_manifest_path.relative_to(OUTPUT_ROOT)),
        "readability": visual_manifest.get("readability"),
        "warnings": visual_manifest.get("warnings") or [],
    }


def main() -> None:
    source_run = json.loads((SOURCE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    source_party_manifests = sorted((SOURCE_ROOT / "parties").glob("*/manifest.json"))
    if len(source_party_manifests) != EXPECTED_PARTY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTY_COUNT} source party manifests; found {len(source_party_manifests)}"
        )
    if source_run.get("qa") != {"slide_count": 55, "passed": 55, "failed": 0}:
        raise RuntimeError(f"Source July v1 batch is not 55/55 QA-passed: {source_run.get('qa')}")
    readiness = source_run.get("readiness") or {}
    if readiness.get("matched_classified_rows") != 2009 or readiness.get("unmatched_classified_rows") != 0:
        raise RuntimeError(f"Unexpected source July v1 readiness: {readiness}")

    period = resolve_monthly_period(PERIOD)
    s3 = boto3.client("s3", region_name="ca-central-1")
    party_manifests: list[dict] = []
    qa_rows: list[dict] = []
    cover_paths: list[tuple[str, Path]] = []
    raw_paths: list[tuple[str, Path]] = []
    share_paths: list[tuple[str, Path]] = []
    per_td_paths: list[tuple[str, Path]] = []
    carousel_items: list[tuple[str, list[Path]]] = []

    for source_manifest_path in source_party_manifests:
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        party = str(source["party"])
        display_party = _display_party_name(party)
        key = str(source["party_key"])
        slides_dir = OUTPUT_ROOT / "parties" / key / "slides"
        paths = [
            slides_dir / "01_cover.png",
            slides_dir / "02_most_discussed_issues.png",
            slides_dir / "03_more_than_average.png",
            slides_dir / "04_more_per_td.png",
            slides_dir / "05_glossary.png",
        ]

        cover_lineage = _render_cover(paths[0], source, s3, period)
        analytical_visuals = {
            "02_most_discussed_issues": _render_variant_2_chart(
                paths[1],
                party=party,
                party_key=key,
                period=period,
                rows=source["raw_counts"],
                slide_title=ANALYTICAL_TITLES["02_most_discussed_issues"],
                value_format="integer",
                metric_id="raw_counts",
            ),
            "03_more_than_average": _render_variant_2_chart(
                paths[2],
                party=party,
                party_key=key,
                period=period,
                rows=source["share_vs_average"],
                slide_title=ANALYTICAL_TITLES["03_more_than_average"],
                value_format="plus_pp_1",
                metric_id="share_vs_average",
            ),
            "04_more_per_td": _render_variant_2_chart(
                paths[3],
                party=party,
                party_key=key,
                period=period,
                rows=source["per_td_vs_average"],
                slide_title=ANALYTICAL_TITLES["04_more_per_td"],
                value_format="plus_per_td_2",
                metric_id="per_td_vs_average",
            ),
        }
        _render_v2_glossary(paths[4])

        manifest = {
            **source,
            "project_id": "party_issue_monthly_profile_v2",
            "source_metrics_manifest": str(source_manifest_path),
            "source_metrics_project_id": source_run.get("project_id"),
            "party_asset_registry": "configs/reference/party_assets_v1.csv",
            **cover_lineage,
            "analytical_slide_visual_source": "Variant 2 — Matplotlib final January commissioning, run #203",
            "analytical_visual_variant": 2,
            "analytical_titles": ANALYTICAL_TITLES,
            "glossary_terms": [term for term, _ in V2_GLOSSARY],
            "analytical_visuals": analytical_visuals,
            "slides": [str(path.relative_to(OUTPUT_ROOT)) for path in paths],
            "review_state": "pending_human_review",
            "publication_enabled": False,
        }
        manifest_path = OUTPUT_ROOT / "parties" / key / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        party_manifests.append(manifest)

        for slide_no, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                ok = image.size == (profile.W, profile.H)
            qa_rows.append(
                {
                    "party": party,
                    "party_key": key,
                    "slide": slide_no,
                    "path": str(path.relative_to(OUTPUT_ROOT)),
                    "dimensions_ok": ok,
                    "status": "PASS" if ok else "FAIL",
                }
            )
        cover_paths.append((display_party, paths[0]))
        raw_paths.append((display_party, paths[1]))
        share_paths.append((display_party, paths[2]))
        per_td_paths.append((display_party, paths[3]))
        carousel_items.append((display_party, paths))

    if len(qa_rows) != EXPECTED_SLIDE_COUNT or any(row["status"] != "PASS" for row in qa_rows):
        raise RuntimeError("Rendered-slide QA failed")

    contacts = OUTPUT_ROOT / "contact_sheets"
    profile._contact_sheet(cover_paths, contacts / "covers.jpg")
    profile._contact_sheet(raw_paths, contacts / "most_discussed_issues.jpg")
    profile._contact_sheet(share_paths, contacts / "more_than_average.jpg")
    profile._contact_sheet(per_td_paths, contacts / "more_per_td.jpg")
    profile._carousel_sheet(carousel_items, contacts / "five_slide_overview.jpg")

    qa_path = OUTPUT_ROOT / "qa_summary.csv"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(qa_rows[0]))
        writer.writeheader()
        writer.writerows(qa_rows)

    max_four_row_thickness_px = round((1210 * horizontal_bar.PLOT_HEIGHT / MIN_VISUAL_ROWS) * 0.72, 2)
    run_manifest = {
        "project_id": "party_issue_monthly_profile_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": source_run["period"],
        "readiness": source_run["readiness"],
        "sources": source_run["sources"],
        "source_metrics_run_manifest": str(SOURCE_ROOT / "run_manifest.json"),
        "source_metrics_project_id": source_run.get("project_id"),
        "calculation": source_run["calculation"],
        "presentation_labels": source_run["presentation_labels"],
        "chart_geometry": {
            "approved_variant": 2,
            "source_family": "Matplotlib final January commissioning",
            "source_commit": VARIANT_2_SOURCE_COMMIT,
            "source_run_number": VARIANT_2_SOURCE_RUN,
            "source_run_id": VARIANT_2_SOURCE_RUN_ID,
            "visual_media_dimensions": [1032, 1210],
            "outer_slide_dimensions": [1080, 1350],
            "plot_bottom_ratio": horizontal_bar.PLOT_BOTTOM,
            "plot_right_ratio": horizontal_bar.PLOT_RIGHT,
            "plot_height_ratio": horizontal_bar.PLOT_HEIGHT,
            "min_plot_left_ratio": horizontal_bar.MIN_PLOT_LEFT,
            "max_plot_left_ratio": horizontal_bar.MAX_PLOT_LEFT,
            "category_font_size_range": [horizontal_bar.MIN_CATEGORY_FONT_SIZE, horizontal_bar.MAX_CATEGORY_FONT_SIZE],
            "value_font_size_range": [horizontal_bar.MIN_VALUE_FONT_SIZE, horizontal_bar.MAX_VALUE_FONT_SIZE],
            "axis_font_size": horizontal_bar.AXIS_FONT_SIZE,
            "bar_height_ratio_for_7_rows": 0.62,
            "min_visual_rows": MIN_VISUAL_ROWS,
            "max_short_chart_bar_thickness_px": max_four_row_thickness_px,
            "short_chart_bar_thickness_policy": "1-3 row charts use a centered virtual 4-row stack",
        },
        "analytical_slide_visual_source": "Variant 2 — Matplotlib final January commissioning, run #203",
        "analytical_visual_variant": 2,
        "analytical_titles": ANALYTICAL_TITLES,
        "glossary_terms": [term for term, _ in V2_GLOSSARY],
        "party_asset_registry": "configs/reference/party_assets_v1.csv",
        "cover_title": COVER_TITLE,
        "cover_logo_geometry": {
            "square_size": [LOGO_SIZE, LOGO_SIZE],
            "top": LOGO_TOP,
            "centered": True,
            "border": {
                "enabled": True,
                "color": profile.ACCENT,
                "width_px": LOGO_BORDER_WIDTH,
                "position": "inside_square",
            },
            "artwork_scale_overrides": LOGO_SCALE_OVERRIDES,
        },
        "party_display_aliases": {"Independent": "Independents"},
        "parties": party_manifests,
        "qa": {"slide_count": len(qa_rows), "passed": len(qa_rows), "failed": 0},
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "period": PERIOD,
                "parties": len(party_manifests),
                "slides": len(qa_rows),
                "analytical_visual_variant": 2,
                "analytical_titles": ANALYTICAL_TITLES,
                "min_visual_rows": MIN_VISUAL_ROWS,
                "max_short_chart_bar_thickness_px": max_four_row_thickness_px,
                "output": str(OUTPUT_ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
